import pytest
import torch

from src.trainer_utils import twohot_encode, symexp, symlog


def test_twohot_encode():
    B = torch.arange(start=-20, end=21)
    B = symexp(B)

    bsz = 15
    x = torch.rand(bsz)

    pred = twohot_encode(torch.tensor(x), B)
    print(pred)
