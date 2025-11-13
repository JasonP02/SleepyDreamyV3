import gymnasium as gym
import numpy as np
from gymnasium.wrappers import AddRenderObservation
import h5py
import torch
import torch.nn.functional as F

from .config import config

def create_env_with_vision():
    base_env = gym.make(config.environment.environment_name, render_mode="rgb_array")
    
    env = AddRenderObservation(
        base_env, 
        render_only=False,
        render_key='pixels',
        obs_key='state'
    )
    
    return env

def collect_bootstrapping_examples():
    env = create_env_with_vision()

    with h5py.File(config.general.env_bootstrapping_samples, 'w') as f:
        grp = f.create_group("episodes")
        f.attrs['n_episodes'] = config.train.num_bootstrap_episodes

        for episode_idx in range(config.train.num_bootstrap_episodes):
            print(f"Collecting episode {episode_idx + 1}/{config.train.num_bootstrap_episodes}")
            obs, info = env.reset()
            episode_pixels, episode_vec_obs, episode_actions, episode_rewards = [], [], [], []

            while True:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)

                episode_pixels.append(obs['pixels'])
                episode_vec_obs.append(obs['state'])
                episode_actions.append(action)
                episode_rewards.append(reward)

                if terminated or truncated:
                    break

            # Process and write this episode's data directly to the file
            pixels_np = np.array(episode_pixels)
            pixels_tensor = torch.from_numpy(pixels_np).permute(0, 3, 1, 2).float()
            resized = F.interpolate(pixels_tensor, size=config.models.encoder.cnn.target_size, mode='bilinear')
            resized_np = resized.permute(0, 2, 3, 1).numpy()

            ep_grp = grp.create_group(str(episode_idx))
            ep_grp.create_dataset('pixels', data=resized_np, dtype='uint8')
            ep_grp.attrs['n_steps'] = len(resized_np)
            ep_grp.create_dataset('vec_obs', data=np.array(episode_vec_obs), dtype='float32')
            ep_grp.create_dataset('action', data=np.array(episode_actions), dtype='int64')
            ep_grp.create_dataset('reward', data=np.array(episode_rewards), dtype='float32')

    env.close()

if __name__ == '__main__':
    collect_bootstrapping_examples()
