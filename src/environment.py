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
    episodes = []

    for episode in range(config.train.num_bootstrap_episodes):
        obs, info = env.reset()
        episode_pixels = []
        episode_vec_obs = []
        episode_actions = []
        episode_rewards = []
        
        while True:
            episode_number = len(episode_pixels)
            if episode_number % 100 == 0:
                print(f"Episode {episode_number}")
            
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_pixels.append(obs['pixels'])  # (H, W, C) numpy array
            episode_vec_obs.append(obs['state'])
            episode_actions.append(action)
            episode_rewards.append(reward)
            
            if terminated or truncated:
                break
        
        pixels_np = np.array(episode_pixels)  # (T, H, W, C)
        pixels_tensor = torch.from_numpy(pixels_np).permute(0, 3, 1, 2)  # (T, C, H, W)
        resized = F.interpolate(pixels_tensor, size=config.models.encoder.cnn.target_size, 
                                mode='bilinear')  # (T, C, 64, 64)
        resized = resized.permute(0, 2, 3, 1).numpy()  # (T, 64, 64, C)
        
        episodes.append({
            'pixels': resized,
            'vec_obs': np.array(episode_vec_obs),
            'action': np.array(episode_actions),
            'reward': np.array(episode_rewards)
        })
    
    with h5py.File(config.general.env_bootstrapping_samples, 'w') as f:
        grp = f.create_group("episodes")

        for i, ep in enumerate(episodes):
            ep_grp = grp.create_group(str(i))
            ep_grp['pixels'] = np.array(ep['pixels'])
            ep_grp.attrs['n_steps'] = len(ep['pixels'])
            ep_grp['vec_obs'] = np.array(ep['vec_obs'])
            ep_grp['action'] = np.array(ep['action'])
            ep_grp['reward'] = np.array(ep['reward'])

        f.attrs['n_episodes'] = len(episodes)
    
    env.close()

if __name__ == '__main__':
    collect_bootstrapping_examples()
