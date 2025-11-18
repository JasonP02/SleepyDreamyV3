import torch
from .encoder import ObservationEncoder, ThreeLayerMLP
from .world_model import RSSMWorldModel
from .config import config


def symlog(x):
    out = torch.sign(x) * (torch.log(torch.abs(x) + 1))
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
    # B is a 1D tensor of bin edges.
    # x is a 1D tensor of values to encode.

    # Clamp values to be within the range of B
    x = torch.clamp(x, B.min(), B.max())

    # Find the index of the bin to the right of each value
    # The result of searchsorted is the index of the first element in B that is >= x
    right_bin_indices = torch.searchsorted(B, x)

    # The left bin is the one before it. Clamp at 0 for safety.
    left_bin_indices = torch.clamp(right_bin_indices - 1, 0)

    # Handle cases where right_bin_indices might be out of bounds if x matches B.max()
    right_bin_indices = torch.clamp(right_bin_indices, 0, len(B) - 1)

    # Get the values of the left and right bin edges
    bin_left = B[left_bin_indices]
    bin_right = B[right_bin_indices]

    # Calculate weights
    # Avoid division by zero if a value falls exactly on a bin edge
    denom = bin_right - bin_left
    denom[denom == 0] = 1.0

    weight_right = (x - bin_left) / denom
    weight_left = 1.0 - weight_right

    # Create the two-hot encoded tensor
    batch_size = x.size(0)
    n_bins = B.size(0)
    weights = torch.zeros(batch_size, n_bins, device=x.device)

    # Use scatter to place the weights in the correct locations
    weights.scatter_(1, left_bin_indices.unsqueeze(1), weight_left.unsqueeze(1))
    # Use scatter_add_ for the right bin in case left and right indices are the same
    weights.scatter_add_(1, right_bin_indices.unsqueeze(1), weight_right.unsqueeze(1))

    return weights


def initalize_world_model(device):


