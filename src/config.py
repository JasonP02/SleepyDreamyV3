# config.py
from pydantic import BaseModel
from typing import Tuple
import torch


def get_default_device():
    """Checks for available hardware accelerators."""
    if torch.cuda.is_available():
        # This works for both NVIDIA (CUDA) and AMD (ROCm)
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class GeneralConfig(BaseModel):
    device: str = get_default_device()
    encoder_path: str = "encoder.pt"
    rssm_path: str = "rssm.pt"
    train_world_model: bool = True
    env_bootstrapping_samples: str = "bootstrap_trajectorires.h5"


class EnvironmentConfig(BaseModel):
    environment_name: str = "LunarLander-v3"
    n_actions: int = 4
    n_observations: int = 8


class CNNEncoderConfig(BaseModel):
    stride: int = 2
    activation: str = "sigmoid"
    target_size: Tuple[int, int] = (64, 64)
    kernel_size: int = 2
    padding: int = 0
    input_channels: int = 3  # RGB
    num_layers: int = 4  # number of convolutional layers
    final_feature_size: int = 4  # output is final_feature_size x final_feature_size


class MLPEncoderConfig(BaseModel):
    hidden_dim_ratio: int = 16
    n_layers: int = 3
    latent_categories: int = 16  # Number of categories per latent variable


class GRUConfig(BaseModel):
    n_blocks: int = 8


class EncoderConfig(BaseModel):
    cnn: CNNEncoderConfig = CNNEncoderConfig()
    mlp: MLPEncoderConfig = MLPEncoderConfig()


class ModelsConfig(BaseModel):
    d_hidden: int = 256
    encoder: EncoderConfig = EncoderConfig()
    rnn: GRUConfig = GRUConfig()


class TrainConfig(BaseModel):
    num_bootstrap_episodes: int = 100
    num_episodes: int = 100
    num_bootstrap_epochs: int = 3
    sequence_length: int = 50
    batch_size: int = 1
    beta_dyn: float = 0.99
    beta_rep: float = 0.99
    beta_pred: float = 0.99
    b_start: int = -20
    b_end: int = 21


class Config(BaseModel):
    general: GeneralConfig = GeneralConfig()
    environment: EnvironmentConfig = EnvironmentConfig()
    models: ModelsConfig = ModelsConfig()
    train: TrainConfig = TrainConfig()


# Default configuration instance
config = Config()
