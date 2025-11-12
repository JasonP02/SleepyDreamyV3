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
    # x is a tensor of size (batch)
    # We want to find the two values of B that are closest to x
    x = x.unsqueeze(-1)  # (batch, 1)
    B = B.unsqueeze(0)   # (1, n_bins)

    # 1. Handle values outside the range of B
    # Clamp values to be within the range of B
    x_clamped = torch.clamp(x, B.min(), B.max())

    # 2. Find the index of the right bin
    # `torch.searchsorted` gives the index of the bucket where the value would be inserted.
    # This corresponds to the right bin.
    right_bin_indices = torch.searchsorted(B.squeeze(0), x_clamped.squeeze(-1)).unsqueeze(-1)

    # Ensure left bin indices are not less than 0
    left_bin_indices = torch.clamp(right_bin_indices - 1, 0)

    # 3. Get the values of the left and right bins
    bin_left = torch.gather(B.expand_as(x), 1, left_bin_indices)
    bin_right = torch.gather(B.expand_as(x), 1, right_bin_indices)

    # 4. Calculate weights
    # Avoid division by zero if bin_left and bin_right are the same
    denom = bin_right - bin_left
    denom[denom == 0] = 1.0
    weight_right = (x_clamped - bin_left) / denom
    weight_left = 1.0 - weight_right

    # 5. Create the two-hot encoded tensor
    weights = torch.zeros(x.shape[0], B.shape[1], device=x.device)
    weights.scatter_(1, left_bin_indices, weight_left)
    weights.scatter_add_(1, right_bin_indices, weight_right)
    return weights
