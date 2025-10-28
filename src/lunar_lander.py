import gymnasium as gym
import numpy as np
from gymnasium.wrappers import AddRenderObservation


def create_lunar_lander_with_vision():
    base_env = gym.make("LunarLander-v3", render_mode="rgb_array")
    
    # Add render observations to include vision data
    # This will create a dict observation with both 'pixels' (vision) and 'state' (normal)
    env = AddRenderObservation(
        base_env, 
        render_only=False,  # Keep both render and original observations
        render_key='pixels',  # Key for vision observations
        obs_key='state'  # Key for normal observations
    )
    
    return env


def demonstrate_observations():
    """
    Demonstrates the different observation types available in the wrapped environment.
    """
    env = create_lunar_lander_with_vision()
    
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    
    # Reset environment
    obs, info = env.reset(seed=42)
    
    print("\nObservation keys:", obs.keys())
    print("State observation shape:", obs['state'].shape)
    print("Pixels observation shape:", obs['pixels'].shape)
    
    # Take a few steps to show observations
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"\nStep {i+1}:")
        print(f"  Reward: {reward}")
        print(f"  State observation: {obs['state']}")
        print(f"  Pixels observation shape: {obs['pixels'].shape}")
        
        if terminated or truncated:
            break
    
    env.close()


if __name__ == "__main__":
    demonstrate_observations()