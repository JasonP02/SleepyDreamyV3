import gymnasium as gym
import numpy as np
from gymnasium.wrappers import AddRenderObservation

from .config import config


def create_env_with_vision():
    base_env = gym.make(config.environment.env_name, render_mode="rgb_array")
    
    # Add render observations to include vision data
    # This will create a dict observation with both 'pixels' (vision) and 'state' (normal)
    env = AddRenderObservation(
        base_env, 
        render_only=False,  # Keep both render and original observations
        render_key='pixels',  # Key for vision observations
        obs_key='state'  # Key for normal observations
    )
    
    return env


def collect_bootstrapping_examples():
    
