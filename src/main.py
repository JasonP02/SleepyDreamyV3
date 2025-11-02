import gymnasium as gym
import torch
from .lunar_lander import create_lunar_lander_with_vision
from .world_model import RSSMWorldModel
from .config import config

def main():
    env = create_lunar_lander_with_vision()
    obs, info = env.reset()

    print(obs.keys())

    world_model = RSSMWorldModel(
        mlp_config=config.models.encoder.mlp,
        cnn_config=config.models.encoder.cnn,
        env_config=config.environment,
        gru_config=config.models.rnn,
        batch_size=config.train.batch_size
    )

    for episode in range(config.train.num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        step_count = 0
        total_reward = 0

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