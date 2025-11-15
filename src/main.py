import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
import multiprocessing as mp

from .trainer import train_world_model
from .environment import collect_experiences


def main():
    """
    Initalizes networks, w&b, threads
    """
    parser = argparse.ArgumentParser(
        description="Training and inference interface for DreamerV3."
    )
    parser.add_argument(
        "mode",
        help="Specify 'train' to train a world model, and 'deploy' for interence on a pretrained checkpoint",
    )
    parser.add_argument(
        "--train_steps", help="Number of environment steps to use for training"
    )

    args = parser.parse_args()

    if args.mode == "train":
        # set up threads for experience collection and training concurrently
        experience_loop = mp.Process(target=collect_experiences)
        trainer_loop = mp.Process(target=train_world_model)
    elif args.mode == "deploy":
        pass  # TODO


if __name__ == "__main__":
    main()
