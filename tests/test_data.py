import torch
from src.world_model_trainer import TrajectoryDataset
from src.config import config
import os

def test_trajectory_dataset_getitem():
    """
    Tests that the TrajectoryDataset can successfully load an item
    and that the item has the correct structure and shapes.
    """
    dataset_path = config.general.env_bootstrapping_samples
    sequence_length = 50

    # Ensure the bootstrap file exists
    assert os.path.exists(dataset_path), f"Bootstrap file not found at {dataset_path}. Please generate it first."

    dataset = TrajectoryDataset(dataset_path, sequence_length=sequence_length)
    
    # Get a single item
    sample = dataset[0]

    # Check that all keys are present
    expected_keys = {"pixels", "state", "action", "reward", "terminated"}
    assert set(sample.keys()) == expected_keys, f"Sample keys are incorrect. Got {sample.keys()}"

    # Check shapes
    assert sample['pixels'].shape == (sequence_length, 3, 64, 64)
    assert sample['state'].shape == (sequence_length, config.environment.n_observations)
    assert sample['action'].shape == (sequence_length, config.environment.n_actions)
