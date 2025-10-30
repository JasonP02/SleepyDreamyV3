import gymnasium as gym
import torch
from lunar_lander import create_lunar_lander_with_vision
from world_model import RSSMWorldModel
from config import config

def main():
    print(config.environment.frame_size)

    env = create_lunar_lander_with_vision()
    obs, info = env.reset()

    print(obs.keys())
    print(obs['state'].shape)
    print(obs['pixels'].shape)
    
    
    # world_model = RSSMWorldModel()
    # world_model(obs)

if __name__ == "__main__":
    main()