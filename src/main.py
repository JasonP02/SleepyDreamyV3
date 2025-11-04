import gymnasium as gym
import torch
import os

from .environment import create_env_with_vision, collect_bootstrapping_examples
from .world_model import RSSMWorldModel
from .world_model_trainer import train_world_model
from .config import config

def main():
    env = create_env_with_vision()

    checkpoint_path = config.general.world_model_path
    if not os.path.exists(config.general.env_bootstrapping_samples):
        print("Bootstrap samples not found. Collecting now...")
        collect_bootstrapping_examples()

    if config.general.train_world_model or not os.path.exists(checkpoint_path):
        print("Training world model...")
        train_world_model()

    print("Loading trained world model...")
    world_model = RSSMWorldModel(
        mlp_config=config.models.encoder.mlp,
        cnn_config=config.models.encoder.cnn,
        env_config=config.environment,
        gru_config=config.models.rnn,
        batch_size=1 # For inference, batch size is 1
    )
    world_model.load_state_dict(torch.load(checkpoint_path))
    print("World model loaded successfully.")

    for episode in range(config.train.num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        step_count = 0
        total_reward = 0

        # Reset world model hidden state at the start of each episode
        world_model.h_prev.zero_()

        while True:
            action = env.action_space.sample()
            action_onehot = torch.zeros(1, config.environment.n_actions)
            action_onehot[0, action] = 1.0
            obs, reward, terminated, truncated, info = env.step(action)

            # Convert numpy arrays to torch tensors and fix format
            pixels = torch.from_numpy(obs['pixels']).float()
            # If pixels is (H, W, C), convert to (N, C, H, W)
            if pixels.dim() == 3:  # (H, W, C)
                pixels = pixels.permute(2, 0, 1).unsqueeze(0)  # -> (1, C, H, W)

            obs_tensor = {
                'pixels': pixels,
                'state': torch.from_numpy(obs['state']).float().unsqueeze(0)  # Add batch dim
            }

            print(action)
            
            world_model(obs_tensor, action_onehot)

            step_count += 1
            episode_reward += reward
            total_reward += reward

            if terminated or truncated or step_count >= config.train.num_episodes:
                break
    
    env.close()

if __name__ == "__main__":
    main()