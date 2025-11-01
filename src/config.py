# config.py
from pydantic import BaseModel
from typing import Tuple

class GeneralConfig(BaseModel):
    env: str = "car_racing"
    device: str = "cuda"

class EnvironmentConfig(BaseModel):
    n_actions: int = 4

class CNNEncoderConfig(BaseModel):
    stride: int = 2
    activation: str = "sigmoid"
    target_size: Tuple[int, int] = (64, 64)
    kernel_size: int = 2
    padding: int = 0
    input_channels: int = 3 # RGB
    num_layers: int = 4  # number of convolutional layers
    final_feature_size: int = 4  # output is final_feature_size x final_feature_size

class MLPEncoderConfig(BaseModel):
    hidden_dim_ratio: int = 16
    n_layers: int = 3
    d_hidden: int = 124
    d_out: int = 1024

class GRUConfig(BaseModel):
    n_blocks: int = 8

class EncoderConfig(BaseModel):
    cnn: CNNEncoderConfig = CNNEncoderConfig()
    mlp: MLPEncoderConfig = MLPEncoderConfig()
    rnn: GRUConfig = GRUConfig()

class ModelsConfig(BaseModel):
    encoder: EncoderConfig = EncoderConfig()

class Config(BaseModel):
    general: GeneralConfig = GeneralConfig()
    environment: EnvironmentConfig = EnvironmentConfig()
    models: ModelsConfig = ModelsConfig()

# Default configuration instance
config = Config()
