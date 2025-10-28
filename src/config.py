# config.py
from pydantic import BaseModel
from typing import Tuple

class GeneralConfig(BaseModel):
    env: str = "car_racing"
    device: str = "cuda"

class EnvironmentConfig(BaseModel):
    frame_size: Tuple[int, int] = (150, 100)

class CNNEncoderConfig(BaseModel):
    stride: int = 2
    activation: str = "sigmoid"

class MLPEncoderConfig(BaseModel):
    n_layers: int = 3
    d_hidden: int = 124
    d_out: int = 16

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
