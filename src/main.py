import gymnasium as gym
import torch
from util import load_config
from lunar_lander import create_lunar_lander_with_vision
from world_model import RSSMWorldModel

def main():
    config = load_config()

    env = create_lunar_lander_with_vision()
    obs, info = env.reset()
    print(obs.keys())
    print(obs['state'])
    print(obs['pixels'][1][1:10])
    
    world_model = RSSMWorldModel()
    world_model(obs)

if __name__ == "__main__":
    main()