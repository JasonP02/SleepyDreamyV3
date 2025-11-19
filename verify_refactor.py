import torch
from src.config import config
from src.trainer import WorldModelTrainer
from src.trainer_utils import initialize_world_model
import multiprocessing as mp

def test_init():
    print("Testing initialization...")
    device = torch.device("cpu")
    
    # Test direct init from utils
    print("Initializing world model from utils...")
    encoder, world_model = initialize_world_model(device, batch_size=2)
    print("Encoder and World Model initialized successfully.")
    
    # Test trainer init
    print("Initializing WorldModelTrainer...")
    data_queue = mp.Queue()
    model_queue = mp.Queue()
    trainer = WorldModelTrainer(config, data_queue, model_queue)
    print("WorldModelTrainer initialized successfully.")
    print("Done.")

if __name__ == "__main__":
    test_init()
