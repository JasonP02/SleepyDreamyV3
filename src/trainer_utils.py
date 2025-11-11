import torch


def symlog(x):
    out = torch.sign(x) * (torch.log(torch.abs(x)) + 1)
    return out


def symexp(x):
    out = torch.sign(x) * (torch.exp(torch.abs(x)) - 1)
    return out


# The network is trained on twohot encoded targets8,28, a generalization of onehot encoding to
# continuous values. The twohot encoding of a scalar is a vector with |B| entries that are all 0 except
# at the indices k and k + 1 of the two bins closest to the encoded scalar. The two entries sum up
# to 1, with linearly higher weight given to the bin that is closer to the encoded continuous number.
# The network is then trained to minimize the categorical cross entropy loss for classification with
# soft targets
def twohot_encode(x, B):
    # B is a vector of exponentially spaced values
    # x is a scalar
    # We want to find the two values of B that are closest to x
    out = torch.zeros_like(B)

    # 1. if x is outside of B, we just set x=b
    if x < B[0]:
        out[0] = 1.0
        return out
    elif x > B[-1]:
        out[-1] = 0.0
        return out

    # 2. if x is inside B, then we need to search over it via bisection
    indices = torch.searchsorted(sorted_sequence=B, input=x)
    print(indices)
    out = indices

    return out
