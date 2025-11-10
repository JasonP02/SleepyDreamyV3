import torch


def symlog(x):
    out = torch.sign(x) * (torch.log(torch.abs(x)) + 1)
    return out


def symexp(x):
    out = torch.sign(x) * (torch.exp(torch.abs(x)) - 1)
    return out
