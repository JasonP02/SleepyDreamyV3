import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import torch.nn.functional as F
try:
    import psutil
except ImportError:
    psutil = None
import os
from queue import Full, Empty

from .config import config
from .trainer_utils import (
    symlog,
    symexp,
    twohot_encode,
    initialize_actor,
    initialize_critic,
)
from .encoder import ObservationEncoder, ThreeLayerMLP
from .world_model import RSSMWorldModel

class WorldModelTrainer:
    def __init__(
        self,
        config,
        data_queue,
        model_queue
    ):
        self.device = torch.device(config.general.device)
        self.model_update_frequency = 10 # fix later
        self.n_dream_steps = config.train.num_dream_steps
        b_start = config.train.b_start
        b_end = config.train.b_end
        beta_range = torch.arange(
        start=b_start,
        end=b_end,
        device=self.device,
        )
        self.B = symexp(beta_range)

        self.actor = initialize_actor(self.device)
        self.critic = initialize_critic(self.device)
        self.encoder = ObservationEncoder(
            mlp_config=config.models.encoder.mlp,
            cnn_config=config.models.encoder.cnn,
            d_hidden=config.models.d_hidden,
        ).to(self.device)
        self.world_model = RSSMWorldModel(
            models_config=config.models,
            env_config=config.environment,
            batch_size=config.train.batch_size,
            b_start=b_start,
            b_end=b_end,
        ).to(self.device)

        self.wm_params = list(self.encoder.parameters()) + list(self.world_model.parameters())
        # TODO: Add learning rate from config
        self.wm_optimizer = optim.Adam(self.wm_params, lr=1e-4, weight_decay=1e-6)

        self.max_train_steps = config.train.num_time_steps
        self.train_step = 0
        self.data_queue = data_queue
        self.model_queue = model_queue
        self.d_hidden = config.models.d_hidden
        self.n_actions = config.environment.n_actions

    def print_memory_usage(self, tag=""):
        if not config.general.debug_memory:
            return

        if psutil:
            process = psutil.Process(os.getpid())
            ram_usage = process.memory_info().rss / 1024 / 1024  # in MB
            print(f"[{tag}] RAM Usage: {ram_usage:.2f} MB")
        else:
            print(f"[{tag}] RAM Usage: (psutil not installed)")

        if self.device.type == "cuda":
            vram_usage = torch.cuda.memory_allocated(self.device) / 1024 / 1024
            print(f"[{tag}] VRAM Usage (CUDA): {vram_usage:.2f} MB")
        elif self.device.type == "mps":
            vram_usage = torch.mps.current_allocated_memory() / 1024 / 1024
            print(f"[{tag}] VRAM Usage (MPS): {vram_usage:.2f} MB")


    def get_data_from_queue(self):
        try:
            # pixels, states, actions, rewards, terminated = loader.sample()
            pixels, states, actions, rewards, terminated = self.data_queue.get()
            self.pixels = torch.from_numpy(pixels).to(self.device).float().unsqueeze(0).permute(0, 1, 4, 2, 3)
            self.states = torch.from_numpy(states).to(self.device).unsqueeze(0)
            self.states = symlog(self.states) # vector states are symlog transformed
            self.actions = torch.from_numpy(actions).to(self.device).unsqueeze(0)
            self.rewards = torch.from_numpy(rewards).to(self.device).unsqueeze(0)
            self.terminated = torch.from_numpy(terminated).to(self.device).unsqueeze(0)
        except Empty:
            pass

    def train_models(self):
        while self.train_step < self.max_train_steps:
            self.print_memory_usage(f"Step {self.train_step} Start")
            self.get_data_from_queue() # TODO: Implement batching with this

            # Reset hidden states per trajectory and move to self.device
            self.world_model.h_prev = torch.zeros_like(self.world_model.h_prev).to(self.device)

            total_loss = 0

            for t_step in range(self.pixels.shape[1]):
                obs_t = {"pixels": self.pixels[:, t_step], "state": self.states[:, t_step]}
                action_t = self.actions[:, t_step]
                reward_t = self.rewards[:, t_step]
                terminated_t = self.terminated[:, t_step]

                posterior_logits = self.encoder(obs_t)  # This is z_t
                posterior_dist = dist.Categorical(logits=posterior_logits) 

                (
                    obs_reconstruction,
                    reward_dist,
                    continue_logits,
                    h_z_joined,
                    prior_logits,
                ) = self.world_model(posterior_dist, action_t)

                # Updating loss of encoder and world model
                wm_loss = self.update_wm_loss(
                    obs_reconstruction,
                    obs_t,
                    reward_dist,
                    reward_t,
                    terminated_t,
                    continue_logits,
                    posterior_dist,
                    prior_logits
                    )
                
                # --- Dream Sequence for Actor-Critic ---
                (
                    dreamed_recurrent_states,
                    dreamed_actions_logits,
                    dreamed_actions_sampled
                ) = self.dream_sequence(
                    h_z_joined,
                    self.world_model.z_embedding(posterior_dist.probs.view(1, -1)),
                    self.n_dream_steps
                    )

                # Predict rewards and values for the dreamed states
                dreamed_rewards_logits = self.world_model.reward_predictor(dreamed_recurrent_states)
                dreamed_rewards_probs = F.softmax(dreamed_rewards_logits, dim=-1)
                dreamed_rewards = torch.sum(dreamed_rewards_probs * self.B, dim=-1)

                dreamed_continues = self.world_model.continue_predictor(dreamed_recurrent_states)

                dreamed_values_logits = self.critic(dreamed_recurrent_states)
                dreamed_values_probs = F.softmax(dreamed_values_logits, dim=-1)
                dreamed_values = torch.sum(dreamed_values_probs * self.B, dim=-1)

                lambda_returns = self.calculate_lambda_returns(
                    dreamed_rewards, dreamed_values, dreamed_continues, config.train.gamma, config.train.lam, self.n_dream_steps
                )

                dreamed_values_logits_flat = dreamed_values_logits.view(-1, dreamed_values_logits.size(-1))
                lambda_returns_flat = lambda_returns.reshape(-1)
                critic_targets = twohot_encode(lambda_returns_flat, self.B)
                
                # critic_loss_fn = nn.CrossEntropyLoss()
                # critic_loss = critic_loss_fn(dreamed_values_logits_flat, critic_targets)
                critic_loss = -torch.mean(torch.sum(critic_targets * F.log_softmax(dreamed_values_logits_flat, dim=-1), dim=-1))
                actor_loss, critic_loss = self.update_actor_critic_losses(
                    dreamed_values_logits,
                    dreamed_values,
                    lambda_returns,
                    dreamed_actions_logits,
                    dreamed_actions_sampled
                )


                total_loss += wm_loss + critic_loss + actor_loss

            # Perform backpropagation on the accumulated loss for the entire sequence
            # optimizer.zero_grad()
            # total_loss.backward()
            # optimizer.step()
            self.train_step += 1


            # Periodically send updated models to the collector
            if self.train_step % self.model_update_frequency == 0:
                self.print_memory_usage(f"Step {self.train_step} End")
                self.send_models_to_collector(self.train_step)

        torch.save(self.encoder.state_dict(), config.general.encoder_path)
        torch.save(self.world_model.state_dict(), config.general.rssm_path)

    def update_actor_critic_losses(
            self,
            dreamed_values_logits,
            dreamed_values,
            lambda_returns,
            dreamed_actions_logits,
            dreamed_actions_sampled
    ):
        
        dreamed_values_logits_flat = dreamed_values_logits.view(-1, dreamed_values_logits.size(-1)) lambda_returns_flat = lambda_returns.reshape(-1)
        critic_targets = twohot_encode(lambda_returns_flat, self.B)
        
        critic_loss_fn = nn.CrossEntropyLoss()
        critic_loss = critic_loss_fn(dreamed_values_logits_flat, critic_targets)

        # Actor Loss: Policy gradient with lambda returns as advantage
        advantage = (lambda_returns - dreamed_values).detach()
        action_dist = torch.distributions.Categorical(logits=dreamed_actions_logits)
        log_probs = action_dist.log_prob(dreamed_actions_sampled)
        
        # Reinforce algorithm: log_prob * advantage
        actor_loss = -torch.mean(log_probs * advantage)

        return actor_loss, critic_loss


    def dream_sequence(
        self,
        initial_h,
        initial_z_embed,
        num_dream_steps
    ):
        """
        Generates a sequence of dreamed states and actions starting from an initial state.
        """
        dreamed_recurrent_states = []
        dreamed_actions_logits = []
        dreamed_actions_sampled = []

        # Start dreaming from the provided initial state
        dream_h = initial_h
        dream_z_embed = initial_z_embed

        for _ in range(num_dream_steps):
            dreamed_recurrent_states.append(dream_h)
            action_logits = self.actor(dream_h)
            dreamed_actions_logits.append(action_logits)

            action_dist = torch.distributions.Categorical(logits=action_logits)
            action_sample = action_dist.sample()
            dreamed_actions_sampled.append(action_sample)
            action_onehot = F.one_hot(action_sample, num_classes=self.n_actions).float()

            # 1. Step the dynamics model to get the next h and prior_z
            dream_h_dyn, dream_prior_logits = self.world_model.step_dynamics(
                dream_z_embed, action_onehot, dream_h
            )

            # 2. Sample z from the prior
            dream_prior_dist = dist.Categorical(logits=dream_prior_logits)
            dream_z_sample_indices = dream_prior_dist.sample()
            dream_z_sample = F.one_hot(
                dream_z_sample_indices, num_classes=self.d_hidden // 16
            ).float()

            # 3. Form the full state (h, z) for the next iteration's predictions
            dream_h = self.world_model.join_h_and_z(dream_h_dyn, dream_z_sample)
            dream_z_embed = self.world_model.z_embedding(
                dream_z_sample.view(dream_z_sample.size(0), -1)
            )

        # Stack the collected dreamed states and actions
        return (
            torch.stack(dreamed_recurrent_states),
            torch.stack(dreamed_actions_logits),
            torch.stack(dreamed_actions_sampled),
        )


    def calculate_lambda_returns(
        self, dreamed_rewards, dreamed_values, dreamed_continues, gamma, lam, num_dream_steps
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

    def update_wm_loss(
        self,
        obs_reconstruction,
        obs_t,
        reward_dist,
        reward_t,
        terminated_t,
        continue_logits,
        posterior_dist,
        prior_logits,
        
    ):
        # Observation pixels are bernoulli, while observation vectors are gaussian
        pixel_probs = obs_reconstruction["pixels"]
        obs_pred = symlog(obs_reconstruction["state"])

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
        # Resize target to match output
        pixel_target = F.interpolate(pixel_target, size=pixel_probs.shape[-2:], mode="bilinear")
        pred_loss_pixel = bce_with_logits_loss_fn(
            input=pixel_probs, target=pixel_target / 255.0
        )

        reward_target = twohot_encode(reward_t, self.B)
        # reward_loss_fn = nn.CrossEntropyLoss()
        # reward_loss = reward_loss_fn(reward_dist, reward_target)
        reward_loss = -torch.mean(torch.sum(reward_target * F.log_softmax(reward_dist, dim=-1), dim=-1))

        # c. continue predictor
        # The target is 1 if we continue, 0 if we terminate.
        continue_target = (1.0 - terminated_t.float()).unsqueeze(-1)
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
        l_dyn = torch.max(torch.tensor(free_bits, device=self.device), l_dyn_raw)

        l_rep_raw = dist.kl_divergence(
            posterior_dist,
            dist.Categorical(logits=prior_dist.logits.detach()),
        ).mean()
        l_rep = torch.max(torch.tensor(free_bits, device=self.device), l_rep_raw)

        return beta_pred * l_pred + beta_dyn * l_dyn + beta_rep * l_rep

    def send_models_to_collector(self, training_step):
        print(f"Trainer: Sending model updates at step {training_step}.")
        models_to_send = {
            "actor": {k: v.cpu() for k, v in self.actor.state_dict().items()},
            "encoder": {k: v.cpu() for k, v in self.encoder.state_dict().items()},
            "world_model": {
                k: v.cpu() for k, v in self.world_model.state_dict().items()
            },
        }
        try:
            # Clear queue to ensure collector gets the latest version
            while not self.model_queue.empty():
                self.model_queue.get_nowait()
            self.model_queue.put_nowait(models_to_send)
        except Full:
            print("Trainer: Model queue was full. Skipping update.")
            pass

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


def train_world_model(data_queue, model_queue):
    """
    Entry point for the training process.
    """
    trainer = WorldModelTrainer(config, data_queue, model_queue)
    trainer.train_models()
            pass
