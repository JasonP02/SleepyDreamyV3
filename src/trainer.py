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

class Loader:
    def __init__(self, batch_size, device):
        """
        Function to pull random data sample from our simulation for training. Supports batching
        Sequences are continuous from the start time.
        """
        self.bsz = batch_size
        self.device = device

    def sample(self):
        return None


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
    loader = Loader(batch_size=config.train.batch_size, device=device)

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
            world_model.z_prev = torch.zeros_like(world_model.z_prev).to(device)

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

                # Actor critic section
                critic_logits = critic(h_z_joined.detach()) # Use the state from the world model
                critic_probs = F.softmax(critic_logits, dim=-1)
                expected_return = critic

                loss = beta_pred * l_pred + beta_dyn * l_dyn + beta_rep * l_rep

                total_loss += loss

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