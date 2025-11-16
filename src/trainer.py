import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import torch.nn.functional as F
from queue import Full, Empty

from .config import config
from .trainer_utils import symlog, symexp, twohot_encode, initalize_critic, initalize_world_model, initialize_actor
from .encoder import ObservationEncoder
from .world_model import RSSMWorldModel

def train_world_model(data_queue, model_queue):
    """
    The terminology can get confusing.
    p(z|h,x) is the posterior from the perspective of the world model.
    """
    device = torch.device(config.general.device)
    print(f"Using device: {device}")

    actor = initialize_actor(device)
    critic = initalize_critic(device)
    encoder, world_model = initalize_world_model(device)

    # Combine parameters for the optimizer
    all_params = list(encoder.parameters()) + list(world_model.parameters())
    optimizer = optim.Adam(all_params, lr=1e-4, weight_decay=1e-6) # TODO FIX

    training_step = 0
    update_frequency = 100 # How often to send models to collector

    while training_step < config.train.num_time_steps:
            # Get a sample from our dataloader
            try:
                # pixels, states, actions, rewards, terminated = loader.sample()
                pixels, states, actions, rewards, terminated = data_queue.get()
                pixels = torch.from_numpy(pixels).to(device).unsqueeze(0)
                states = torch.from_numpy(states).to(device).unsqueeze(0)
                actions = torch.from_numpy(actions).to(device).unsqueeze(0)
                rewards = torch.from_numpy(rewards).to(device).unsqueeze(0)
                terminated = torch.from_numpy(terminated).to(device).unsqueeze(0)

            except Empty:
                continue

            # This is a placeholder for a proper dataloader and batching logic
            # The current implementation processes one episode at a time.
            # For real training, you'd batch multiple sequences.
            batch_idx = 0
            epoch = 0


            # Reset hidden states per trajectory and move to device
            world_model.h_prev = torch.zeros_like(world_model.h_prev).to(device)

            total_loss = 0
            # Accumulators for individual loss components for logging
            total_pred_loss_pixel = 0
            total_pred_loss_vector = 0
            total_reward_loss = 0
            total_pred_loss_continue = 0
            total_l_dyn = 0
            total_l_rep = 0

            states = symlog(states)  # symlog vector inputs before model input

            for t_step in range(pixels.shape[1]):
                obs_t = {"pixels": pixels[:, t_step], "state": states[:, t_step]}
                action_t = actions[:, t_step]
                reward_t = rewards[:, t_step]

                terminated_t = terminated[:, t_step]

                posterior_logits = encoder(obs_t) # This is z_t
                posterior_dist = dist.Categorical(logits=posterior_logits)
                
                # The world model now returns a dictionary of all its outputs
                wm_outputs = world_model(posterior_dist, action_t)
                obs_reconstruction = wm_outputs["obs_reconstruction"]
                reward_dist = wm_outputs["reward_logits"]
                continue_logits = wm_outputs["continue_logits"]
                prior_logits = wm_outputs["prior_logits"]
                h_z_joined = wm_outputs["h_z_joined"]

                # Distribution of mean pixel/observation values
                # Observation pixels are bernoulli, while observation vectors are gaussian
                pixel_probs = obs_reconstruction["pixels"]
                obs_pred = obs_reconstruction["state"]

                pixel_target = obs_t["pixels"]
                obs_target = obs_t["state"]
                obs_target = symlog(obs_target)  # loss in symlog space

                beta_dyn = config.train.beta_dyn
                beta_rep = config.train.beta_rep
                beta_pred = config.train.beta_pred

                # There are three loss terms:
                # 1. Prediction loss: -ln p(x|z,h) - ln(p(r|z,h)) + ln(p(c|z,h))
                # a. dynamics represetnation
                # -ln p(x|z,h) is trained with symlog squared loss
                pred_loss_vector = 1 / 2 * (obs_pred - obs_target) ** 2
                pred_loss_vector = pred_loss_vector.mean()

                bce_with_logits_loss_fn = nn.BCEWithLogitsLoss()
                # The decoder outputs logits, and the target should be in [0,1]
                pred_loss_pixel = bce_with_logits_loss_fn(
                    input=pixel_probs, target=pixel_target / 255.0
                )

                # b. reward predictor
                beta_range = torch.arange(
                    start=config.train.b_start,
                    end=config.train.b_end,
                    device=reward_t.device,
                )
                B = symexp(beta_range)
                reward_target = twohot_encode(reward_t, B)
                reward_loss_fn = nn.CrossEntropyLoss()
                reward_loss = reward_loss_fn(reward_dist, reward_target)

                # c. continue predictor
                # The target is 1 if we continue, 0 if we terminate.
                continue_target = (1.0 - terminated_t).unsqueeze(-1)
                pred_loss_continue = bce_with_logits_loss_fn(
                    continue_logits, continue_target
                )

                # Prediction loss is the sum of the individual losses
                l_pred = (
                    pred_loss_pixel
                    + pred_loss_vector
                    + reward_loss
                    + pred_loss_continue
                )

                # 2. Dynamics loss: max(1,KL) ; KL = KL[sg(q(z|h,x)) || p(z,h)]
                # 3. Representation Loss: max(1,KL) ; KL = KL[q(z|h,x) || sg(p(z|h))]
                # Log-likelihoods. Torch accepts logits

                # The "free bits" technique provides a minimum budget for the KL divergence.
                prior_dist = dist.Categorical(logits=prior_logits)
                free_bits = 1.0
                l_dyn_raw = dist.kl_divergence(
                    dist.Categorical(logits=posterior_dist.logits.detach()),prior_dist,).mean()
                l_dyn = torch.max(torch.tensor(free_bits, device=device), l_dyn_raw)

                l_rep_raw = dist.kl_divergence(
                    posterior_dist, dist.Categorical(logits=prior_dist.logits.detach()),).mean()
                l_rep = torch.max(torch.tensor(free_bits, device=device), l_rep_raw)

                
                # --- Dream Sequence for Actor-Critic ---
                # The actor is trained on the dreamed trajectories.
                # We start from the current state from the replay buffer.
                # Initialize lists to store the dreamed trajectory
                dreamed_rewards = []
                dreamed_continues = []
                dreamed_values = []

                # Start dreaming from the current state
                dream_h = h_z_joined.detach() # This is the input for actor/critic
                dream_z_embed = wm_outputs["z_embed"].detach() # This is for the recurrent dynamics
                action = actor()
                dream_action = action.detach()

                # Dream future steps for lambda returns
                for _ in range(config.train.num_dream_steps):
                    # 1. Step the dynamics model to get the next h and prior_z
                    dream_h_dyn, dream_prior_logits = world_model.step_dynamics(dream_z_embed, dream_action, dream_h)

                    # 2. Sample z from the prior
                    dream_prior_dist = dist.Categorical(logits=dream_prior_logits)
                    dream_z_sample = F.one_hot(dream_prior_dist.sample(), num_classes=config.models.d_hidden // 16).float()

                    # 3. Form the full state (h, z) for predictions
                    dream_h_z_joined = world_model.join_h_and_z(dream_h_dyn, dream_z_sample)

                    # 4. Predict expected reward, continue probability, and value for this dreamed state
                    reward_logits = world_model.reward_predictor(dream_h_z_joined)
                    reward_probs = F.softmax(reward_logits, dim=-1)
                    expected_reward = torch.sum(reward_probs * B, dim=-1)
                    dreamed_rewards.append(expected_reward)

                    dreamed_continues.append(world_model.continue_predictor(dream_h_z_joined))
                    dream_critic_logits = critic(dream_h_z_joined)
                    critic_probs = F.softmax(dream_critic_logits, dim=-1)
                    dreamed_values.append(torch.sum(critic_probs * B, dim=-1)) # The critic's value is the expectation

                    # 5. Get the next action from the actor for the next dream step
                    dream_action = actor(dream_h_z_joined)

                    # 6. Prepare inputs for the *next* loop iteration
                    dream_h = dream_h_dyn
                    dream_z_embed = world_model.z_embedding(dream_z_sample.view(dream_z_sample.size(0), -1))

                gamma = config.train.gamma
                lam = config.train.lam
                # Now that we have our dreamed states, we work backwards to get the return estimate
                # To do this, we can simply reverse the lists.
                dreamed_rewards.reverse()
                dreamed_continues.reverse()
                dreamed_values.reverse()

                # --- Lambda-Return Calculation ---
                # Bootstrap the return calculation with the value of the last dreamed state.
                lambda_returns = []
                next_lambda_return = dreamed_values[0]

                # Iterate backwards through the trajectory (which is forwards through our reversed lists)
                for i in range(len(dreamed_rewards)):
                    # The value and continue flag are for the *next* state (t+1)
                    # The reward is for the current state (t)
                    reward_t = dreamed_rewards[i]
                    continue_prob_t = torch.sigmoid(dreamed_continues[i]) # Sigmoid to get probability
                    value_t = dreamed_values[i]

                    # The recursive formula for lambda-returns
                    next_lambda_return = reward_t + gamma * continue_prob_t * (
                        (1 - lam) * value_t + lam * next_lambda_return
                    )
                    lambda_returns.append(next_lambda_return)

                # The returns are calculated backwards, so we reverse them to align with the actions
                lambda_returns.reverse()

                # Actor critic section
                critic_logits = critic(h_z_joined.detach()) # Use the state from the world model
                critic_probs = F.softmax(critic_logits, dim=-1)
                critic_target = twohot_encode(lambda_returns, B)

                critic_loss_fn = nn.CrossEntropyLoss()
                critic_loss = critic_loss_fn(critic_logits, critic_target.detach())

                wm_loss = beta_pred * l_pred + beta_dyn * l_dyn + beta_rep * l_rep

                # 

                total_loss += wm_loss + critic_loss + actor_loss

                # Accumulate individual losses
                total_pred_loss_pixel += pred_loss_pixel.item()
                total_pred_loss_vector += pred_loss_vector.item()
                total_reward_loss += reward_loss.item()
                total_pred_loss_continue += pred_loss_continue.item()
                total_l_dyn += l_dyn.item()
                total_l_rep += l_rep.item()

            # Perform backpropagation on the accumulated loss for the entire sequence
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            training_step += 1

            if batch_idx % 10 == 0:  # Print every 10 batches
                seq_len = pixels.shape[1]
                avg_loss = total_loss.item() / seq_len
                print(
                    f"Epoch {epoch + 1}, Batch {batch_idx}/{len(dataloader)}, Loss: {avg_loss:.4f}"
                )
                # make prints more useful; that is, show the loss for each term
                print(
                    f"  Pred_Pixel: {total_pred_loss_pixel / seq_len:.4f}",
                    f"  Pred_Vector: {total_pred_loss_vector / seq_len:.4f}",
                    f"  Reward: {total_reward_loss / seq_len:.4f}",
                    f"  Continue: {total_pred_loss_continue / seq_len:.4f}",
                    f"  Dyn: {total_l_dyn / seq_len:.4f}",
                    f"  Rep: {total_l_rep / seq_len:.4f}",
                )

            # Periodically send updated models to the collector
            if training_step % update_frequency == 0:
                print(f"Trainer: Sending model updates at step {training_step}.")
                models_to_send = {
                    'actor': {k: v.cpu() for k, v in actor.state_dict().items()},
                    'encoder': {k: v.cpu() for k, v in encoder.state_dict().items()},
                    'world_model': {k: v.cpu() for k, v in world_model.state_dict().items()},
                }
                try:
                    # Clear queue to ensure collector gets the latest version
                    while not model_queue.empty():
                        model_queue.get_nowait()
                    model_queue.put_nowait(models_to_send)
                except Full:
                    print("Trainer: Model queue was full. Skipping update.")
                    pass

    # Save final models
    torch.save(encoder.state_dict(), config.general.encoder_path)
    torch.save(world_model.state_dict(), config.general.rssm_path)