import gymnasium as gym
import torch
from util import load_config

def main():
    config = load_config()

    # Loading gym and accessing the 
    env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=False)
    obs, info = env.reset()

    episode_over = False
    total_reward = 0

    while not episode_over:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        episode_over = terminated or truncated

    # Box observation space (continuous values)
    print(f"Observation space: {env.observation_space}")  # Box with 4 values
    # Box([-4.8, -inf, -0.418, -inf], [4.8, inf, 0.418, inf])
    print(f"Sample observation: {env.observation_space.sample()}")  # Random valid observation
    obs = env.observation_space.sample()
    print(obs.shape)
    import matplotlib.pyplot as plt
    plt.imshow(obs)            # HWC uint8 works directly
    plt.show()

    print(f"Episode finished! Total reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    main()