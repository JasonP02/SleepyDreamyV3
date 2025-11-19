import gymnasium as gym
import numpy as np
from gymnasium.wrappers import AddRenderObservation
import h5py
import torch
import torch.nn.functional as F
from queue import Empty

from .config import config
from .world_model import RSSMWorldModel 
from .encoder import ObservationEncoder
from .trainer_utils import initialize_actor


def create_env_with_vision():
    base_env = gym.make(config.environment.environment_name, render_mode="rgb_array")

    env = AddRenderObservation(
        base_env, render_only=False, render_key="pixels", obs_key="state"
    )
    return env

def collect_experiences(data_queue, model_queue):
    """
    Continuously collects experiences from the environment and puts them on a queue.
    This function acts as the "Producer".
    """
    env = create_env_with_vision()
    device = "cpu" # Collector runs on CPU
    actor = initialize_actor(device=device)
    encoder = ObservationEncoder(
        mlp_config=config.models.encoder.mlp,
        cnn_config=config.models.encoder.cnn,
        d_hidden=config.models.d_hidden,
    ).to(device)
    world_model = RSSMWorldModel(
        models_config=config.models,
        env_config=config.environment,
        batch_size=1,  # For inference, batch size is 1
        b_start=b_start,
        b_end=b_end,
    ).to(device)

    # Move models to eval mode
    actor.eval()
    encoder.eval()
    world_model.eval()

    episode_count = 0

    while True:  # Run indefinitely
        try:
            # Check for updated models from the trainer
            new_model_states = model_queue.get_nowait()
            actor.load_state_dict(new_model_states['actor'])
            encoder.load_state_dict(new_model_states['encoder'])
            world_model.load_state_dict(new_model_states['world_model'])
            actor.eval()
            encoder.eval()
            world_model.eval()
            print("Collector: Updated models from trainer.")
        except Empty:
            pass

        episode_count += 1
        print(f"Collecting episode {episode_count}...")
        obs, info = env.reset()

        # Reset world model state for new episode
        h = torch.zeros(1, config.models.d_hidden * config.models.rnn.n_blocks, device=device)
        z_probs = torch.zeros(1, config.models.d_hidden, config.models.d_hidden // 16, device=device)
        action = torch.zeros(1, config.environment.n_actions, device=device)

        episode_pixels, episode_vec_obs, episode_actions, episode_rewards, episode_terminated = (
            [], [], [], [], []
        )

        while True:
            # 1. Preprocess observation
            pixel_obs_t = torch.from_numpy(obs['pixels']).to(device).float().permute(2, 0, 1).unsqueeze(0)
            vec_obs_t = torch.from_numpy(obs['state']).to(device).float().unsqueeze(0)
            current_obs_dict = {"pixels": pixel_obs_t, "state": vec_obs_t}

            # 2. Run inference with no gradients
            with torch.no_grad():
                # a. Encode observation to get posterior z_t
                posterior_logits = encoder(current_obs_dict)
                posterior_dist = torch.distributions.Categorical(logits=posterior_logits)
                z_indices = posterior_dist.sample()
                z_onehot = F.one_hot(z_indices, num_classes=config.models.d_hidden // 16).float()
                # Use the "soft" one-hot for consistency with training
                z_sample = z_onehot + (posterior_dist.probs - posterior_dist.probs.detach())
                z_flat = z_sample.view(1, -1)

                # b. Update recurrent state h_t
                z_embed = world_model.z_embedding(z_flat)
                h, _ = world_model.step_dynamics(z_embed, action, h) # We don't need prior_logits here

                # c. Get action from actor
                actor_input = world_model.join_h_and_z(h, z_sample)
                action_dist = torch.distributions.Categorical(logits=actor(actor_input))
                action = action_dist.sample()

            # Convert action tensor to a numpy int for the environment
            action_np = action.item()
            # Convert action to one-hot for the next world model step
            action = F.one_hot(action, num_classes=config.environment.n_actions).float()

            # Execute action in environment
            obs, reward, terminated, truncated, info = env.step(action_np)

            episode_pixels.append(obs["pixels"])
            episode_vec_obs.append(obs["state"])
            episode_actions.append(action.cpu().numpy()) # Store one-hot action
            episode_rewards.append(reward)
            episode_terminated.append(terminated)

            if terminated or truncated:
                break

        # Process and package the data for the queue
        pixels_np = np.array(episode_pixels, dtype=np.uint8)
        vec_obs_np = np.array(episode_vec_obs, dtype=np.float32)
        actions_np = np.squeeze(np.array(episode_actions, dtype=np.float32))
        rewards_np = np.array(episode_rewards, dtype=np.float32)
        terminated_np = np.array(episode_terminated, dtype=bool)

        # Put the complete episode data as a tuple of numpy arrays onto the queue
        data_queue.put((pixels_np, vec_obs_np, actions_np, rewards_np, terminated_np))
