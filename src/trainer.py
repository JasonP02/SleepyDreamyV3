import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import torch.nn.functional as F
from queue import Full, Empty

from .config import config
from .trainer_utils import (
    symlog,
    symexp,
    twohot_encode,
)
from .encoder import ObservationEncoder, ThreeLayerMLP
from .world_model import RSSMWorldModel

class WorldModelTrainer:
    def __init__(
        self,
        config,
        data_queue,
    ):
        self.device = torch.device(config.general.device)
        self.actor = self.initialize_actor()
        self.critic = self.initalize_critic()
        self.encoder = ObservationEncoder(
            mlp_config=config.models.encoder.mlp,
            cnn_config=config.models.encoder.cnn,
            d_hidden=config.models.d_hidden,
        ).to(self.device)
        self.world_model = RSSMWorldModel(
            models_config=config.models,
            env_config=config.environment,
            batch_size=1,  # For inference, batch size is 1
            b_start=config.train.b_start,
            b_end=config.train.b_end,
        ).to(self.device)

        self.wm_params = list(self.encoder.parameters()) + list(self.world_model.parameters())
        self.wm_optimizer = optim.Adam(self.wm_params, lr=config.train.wm_lr, weight_decay=config.train.wm_lr)

        self.max_train_steps = config.train.num_time_steps
        self.train_step = 0
        self.data_queue = data_queue

    def get_data_from_queue(self):
        try:
            # pixels, states, actions, rewards, terminated = loader.sample()
            pixels, states, actions, rewards, terminated = self.data_queue.get()
            self.pixels = torch.from_numpy(pixels).to(self.device).unsqueeze(0)
            self.states = torch.from_numpy(states).to(self.device).unsqueeze(0)
            self.actions = torch.from_numpy(actions).to(self.device).unsqueeze(0)
            self.rewards = torch.from_numpy(rewards).to(self.device).unsqueeze(0)
            self.terminated = torch.from_numpy(terminated).to(self.device).unsqueeze(0)
        except Empty:
            pass

    def train_models(self):
        while self.train_step < self.max_train_steps:
            self.get_data_from_queue()
            



    def initialize_actor(self):
        d_in = (config.models.d_hidden * config.models.rnn.n_blocks) + (
            config.models.d_hidden
            * (config.models.d_hidden // config.models.encoder.mlp.latent_categories)
        )
        return ThreeLayerMLP(
            d_in=d_in,
            d_hidden=config.models.d_hidden,
            d_out=config.environment.n_actions,
        ).to(self.device)

    def initalize_critic(self):
        d_in = (config.models.d_hidden * config.models.rnn.n_blocks) + (
            config.models.d_hidden
            * (config.models.d_hidden // config.models.encoder.mlp.latent_categories)
        )
        return ThreeLayerMLP(
            d_in=d_in,
            d_hidden=config.models.d_hidden,
            d_out=config.train.b_end - config.train.b_start,
        ).to(self.device)



def get_world_model_loss(
    obs_reconstruction,
    obs_t,
    reward_t,
    terminated_t,
    prior_logits,
    reward_dist,
    continue_logits,
    posterior_dist,
    device,
    B,
):
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
        pred_loss_pixel + pred_loss_vector + reward_loss + pred_loss_continue
    )

    # 2. Dynamics loss: max(1,KL) ; KL = KL[sg(q(z|h,x)) || p(z,h)]
    # 3. Representation Loss: max(1,KL) ; KL = KL[q(z|h,x) || sg(p(z|h))]
    # Log-likelihoods. Torch accepts logits

    # The "free bits" technique provides a minimum budget for the KL divergence.
    prior_dist = dist.Categorical(logits=prior_logits)
    free_bits = 1.0
    l_dyn_raw = dist.kl_divergence(
        dist.Categorical(logits=posterior_dist.logits.detach()),
        prior_dist,
    ).mean()
    l_dyn = torch.max(torch.tensor(free_bits, device=device), l_dyn_raw)

    l_rep_raw = dist.kl_divergence(
        posterior_dist,
        dist.Categorical(logits=prior_dist.logits.detach()),
    ).mean()
    l_rep = torch.max(torch.tensor(free_bits, device=device), l_rep_raw)

    return beta_pred * l_pred + beta_dyn * l_dyn + beta_rep * l_rep


def dream_sequence(
    initial_h, initial_z_embed, world_model, actor, num_dream_steps, n_actions, d_hidden
):
    """
    Generates a sequence of dreamed states and actions starting from an initial state.
    """
    dreamed_hs = []
    dreamed_actions_logits = []
    dreamed_actions_sampled = []

    # Start dreaming from the provided initial state
    dream_h = initial_h
    dream_z_embed = initial_z_embed

    for _ in range(num_dream_steps):
        dreamed_hs.append(dream_h)
        action_logits = actor(dream_h)
        dreamed_actions_logits.append(action_logits)

        action_dist = torch.distributions.Categorical(logits=action_logits)
        action_sample = action_dist.sample()
        dreamed_actions_sampled.append(action_sample)
        action_onehot = F.one_hot(action_sample, num_classes=n_actions).float()

        # 1. Step the dynamics model to get the next h and prior_z
        dream_h_dyn, dream_prior_logits = world_model.step_dynamics(
            dream_z_embed, action_onehot, dream_h
        )

        # 2. Sample z from the prior
        dream_prior_dist = dist.Categorical(logits=dream_prior_logits)
        dream_z_sample_indices = dream_prior_dist.sample()
        dream_z_sample = F.one_hot(
            dream_z_sample_indices, num_classes=d_hidden // 16
        ).float()

        # 3. Form the full state (h, z) for the next iteration's predictions
        dream_h = world_model.join_h_and_z(dream_h_dyn, dream_z_sample)
        dream_z_embed = world_model.z_embedding(
            dream_z_sample.view(dream_z_sample.size(0), -1)
        )

    # Stack the collected dreamed states and actions
    return (
        torch.stack(dreamed_hs),
        torch.stack(dreamed_actions_logits),
        torch.stack(dreamed_actions_sampled),
    )


def calculate_lambda_returns(
    dreamed_rewards, dreamed_values, dreamed_continues, gamma, lam, num_dream_steps
):
    """
    Calculates lambda-returns for a dreamed trajectory.
    """
    lambda_returns = []
    next_lambda_return = dreamed_values[-1]

    # Iterate backwards through the trajectory
    for i in reversed(range(num_dream_steps)):
        reward_t = dreamed_rewards[i]
        continue_prob_t = torch.sigmoid(dreamed_continues[i])
        value_t = dreamed_values[i]

        next_lambda_return = reward_t + gamma * continue_prob_t * (
            (1 - lam) * value_t + lam * next_lambda_return
        )
        lambda_returns.append(next_lambda_return)

    # The returns are calculated backwards, so we reverse them
    return torch.stack(lambda_returns).flip(0)

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
    optimizer = optim.Adam(all_params, lr=1e-4, weight_decay=1e-6)  # TODO FIX

    training_step = 0
    update_frequency = 100  # How often to send models to collector

    while training_step < config.train.num_time_steps:
        # Get a sample from our dataloader

        # This is a placeholder for a proper dataloader and batching logic
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

            posterior_logits = encoder(obs_t)  # This is z_t
            posterior_dist = dist.Categorical(logits=posterior_logits)

            # The world model now returns a dictionary of all its outputs
            wm_outputs = world_model(posterior_dist, action_t)
            obs_reconstruction = wm_outputs["obs_reconstruction"]
            reward_dist = wm_outputs["reward_logits"]
            continue_logits = wm_outputs["continue_logits"]
            prior_logits = wm_outputs["prior_logits"]
            h_z_joined = wm_outputs["h_z_joined"]

            # b. reward predictor
            beta_range = torch.arange(
            start=config.train.b_start,
            end=config.train.b_end,
            device=reward_t.device,
            )
            B = symexp(beta_range)

            wm_loss = get_world_model_loss(
                obs_reconstruction,
                obs_t,
                reward_t,
                terminated_t,
                prior_logits,
                reward_dist,
                continue_logits,
                posterior_dist,
                device,
                B,
            )

            # --- Dream Sequence for Actor-Critic ---
            dreamed_hs, dreamed_actions_logits, dreamed_actions_sampled = dream_sequence(
                initial_h=h_z_joined.detach(),
                initial_z_embed=wm_outputs["z_embed"].detach(),
                world_model=world_model,
                actor=actor,
                num_dream_steps=config.train.num_dream_steps,
                n_actions=config.environment.n_actions,
                d_hidden=config.models.d_hidden,
            )

            # Predict rewards and values for the dreamed states
            dreamed_rewards_logits = world_model.reward_predictor(dreamed_hs)
            dreamed_rewards_probs = F.softmax(dreamed_rewards_logits, dim=-1)
            dreamed_rewards = torch.sum(dreamed_rewards_probs * B, dim=-1)

            dreamed_continues = world_model.continue_predictor(dreamed_hs)

            dreamed_values_logits = critic(dreamed_hs)
            dreamed_values_probs = F.softmax(dreamed_values_logits, dim=-1)
            dreamed_values = torch.sum(dreamed_values_probs * B, dim=-1)

            lambda_returns = calculate_lambda_returns(
                dreamed_rewards, dreamed_values, dreamed_continues, config.train.gamma, config.train.lam, config.train.num_dream_steps
            )

            # --- Critic Loss ---
            # The critic loss is the cross-entropy between the critic's predicted
            # value distribution and the two-hot encoded lambda-return target.
            # We can compute this for the entire dream sequence at once.

            # Reshape inputs for batch processing with CrossEntropyLoss.
            # Input (dreamed_values_logits) should be (N, C) where C is number of classes.
            # Target (critic_targets) should be (N, C) for soft targets.
            # N = num_dream_steps * batch_size
            dreamed_values_logits_flat = dreamed_values_logits.view(-1, dreamed_values_logits.size(-1))
            lambda_returns_flat = lambda_returns.reshape(-1)
            critic_targets = twohot_encode(lambda_returns_flat, B)
            
            critic_loss_fn = nn.CrossEntropyLoss()
            critic_loss = critic_loss_fn(dreamed_values_logits_flat, critic_targets)

            # Actor Loss: Policy gradient with lambda returns as advantage
            advantage = (lambda_returns - dreamed_values).detach()
            action_dist = torch.distributions.Categorical(logits=dreamed_actions_logits)
            log_probs = action_dist.log_prob(dreamed_actions_sampled)
            
            # Reinforce algorithm: log_prob * advantage
            actor_loss = -torch.mean(log_probs * advantage)

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
                "actor": {k: v.cpu() for k, v in actor.state_dict().items()},
                "encoder": {k: v.cpu() for k, v in encoder.state_dict().items()},
                "world_model": {
                    k: v.cpu() for k, v in world_model.state_dict().items()
                },
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


def get_s_using_ema(lambda_returns):
    pass
