import pytest
import torch

from src.world_model import GatedRecurrentUnit
from src.config import config

def test_gru():
    """
    Tests shape outputs of the gated recurrent unit
    """
    d_hidden = config.models.encoder.mlp.d_hidden
    n_actions = config.environment.n_actions
    d_in = d_hidden + n_actions
    bsz = 1000
    in_shape = (bsz, d_in)
    x = torch.rand(in_shape)

    gru = GatedRecurrentUnit(d_in=d_in, d_hidden=d_hidden)
    out = gru(x)
    assert out.shape == (bsz, d_hidden)
