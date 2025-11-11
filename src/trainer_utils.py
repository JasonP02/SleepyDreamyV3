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
    # x is a vector of size (batch)
    # We want to find the two values of B that are closest to x
    out = torch.zeros(x.shape[0], B.shape[0])

    # 1. if x is outside of B, we just set x=b
    # if x smaller than b[0], we want out[batch,0] to be 1
    # x < B[0] returns a boolean tensor where 1 indicates x < B[0] for that batch value
    # We can then push in this tensor to
    mask_left = x < B[0]
    mask_right = x > B[-1]
    print(out.shape)
    out[:, 0] = mask_left
    out[:, -1] = mask_right

    # 2. if x is inside B, then we need to search over it via bisection
    # We can safely take the found index, and bin to the left as our k, and k+1 bins
    index = torch.searchsorted(sorted_sequence=B, input=x)

    bin1 = B[index]

    zero_mask = index == 0:
    out[index] = 1

    bin2 = B[index - 1]

    out[index] = (x - bin2) / (bin1 - bin2)
    out[index - 1] = (bin1 - x) / (bin1 - bin2)

    return out
