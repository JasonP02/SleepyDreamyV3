import torch
import pytest

from src.config import config
from src.world_model import RSSMWorldModel

def test_rssm():
    rssm = RSSMWorldModel(
        mlp_config=config.models.encoder.mlp,
        cnn_config=config.models.encoder.cnn,
        env_config=config.environment,
        gru_config=config.models.rnn,
        batch_size=config.train.batch_size
    )

    # Create batch of images with proper shape (batch, channels, height, width)
    batch_size = config.train.batch_size
    image = torch.rand((batch_size, 3, 400, 400))
    obs = torch.rand((batch_size, config.environment.n_observations))

    x = {"state": obs, "pixels": image}
    a = torch.rand((batch_size, config.environment.n_actions))
    out = rssm(x=x, a=a)
    print(out.shape)
    print(out)

