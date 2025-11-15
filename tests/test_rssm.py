import torch

from src.config import config
from src.world_model import RSSMWorldModel


def test_rssm():
    rssm = RSSMWorldModel(
        models_config=config.models,
        env_config=config.environment,
        batch_size=config.train.batch_size,
        b_start=-20,
        b_end=21,
    )

    # Create batch of images with proper shape (batch, channels, height, width)
    batch_size = config.train.batch_size
    image = torch.rand((batch_size, 3, 400, 400))
    obs = torch.rand((batch_size, config.environment.n_observations))

    x = {"state": obs, "pixels": image}
    a = torch.rand((batch_size, config.environment.n_actions))
    out = rssm(x=x, a=a)
    print(out)
