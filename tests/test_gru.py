import torch

from src.world_model import GatedRecurrentUnit
from src.config import config

def test_gru():
    """
    Tests shape outputs of the gated recurrent unit
    """
    d_hidden = config.models.d_hidden
    n_actions = config.environment.n_actions
    d_in = d_hidden + n_actions
    bsz = 1000
    z = torch.rand((bsz, d_hidden))  # encoded state
    a = torch.rand((bsz, n_actions))  # action

    gru = GatedRecurrentUnit(d_in=d_in, d_hidden=d_hidden)
    out = gru(z, a)
    assert out.shape == (bsz, d_hidden)
