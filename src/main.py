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

        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            world_model(obs)

            step_count += 1
            episode_reward += reward
            total_reward += reward

            if terminated or truncated or step_count >= config.train.num_episodes:
                break
    
    env.close()

if __name__ == "__main__":
    main()