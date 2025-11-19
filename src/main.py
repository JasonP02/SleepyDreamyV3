import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
import multiprocessing as mp

from .config import config
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
    parser.add_argument(
        "--debug_memory", action="store_true", help="Enable memory profiling prints"
    )

    args = parser.parse_args()

    if args.mode == "train":
        if args.debug_memory:
            config.general.debug_memory = True
        # set up threads for experience collection and training concurrently
        print("Starting experience collection and training processes...")

        # Create a queue to pass data from collector to trainer
        # maxsize prevents the collector from running too far ahead and using up all the memory.
        data_queue = mp.Queue(maxsize=config.train.batch_size * 5)
        model_queue = mp.Queue(maxsize=1)
        experience_loop = mp.Process(target=collect_experiences, args=(data_queue,model_queue))
        trainer_loop = mp.Process(target=train_world_model, args=(data_queue,model_queue))

        experience_loop.start()
        trainer_loop.start()

        experience_loop.join()
        trainer_loop.join()

        print("Both processes have finished.")
    elif args.mode == "deploy":
        pass  # TODO


if __name__ == "__main__":
    main()
