"""
Takes in collection of observation action pairs from a pretrained policy
Batches them
Trains a world model conditioned on those observations & actions
This is a bootstrapping script for integration with online actor/critic learning later
"""
import os
import torch
import torch.nn.functional as F

from .world_model import RSSMWorldModel
from .config import config
from .environment import create_env_with_vision, collect_bootstrapping_examples

def main():
    # collect experiences:
    if os.path.exists(config.general.env_bootstrapping_samples):
        pass
    else:
        collect_bootstrapping_examples()
    env = create_env_with_vision()
    obs, info = env.reset()




if __name__ == "__main__":
    main()