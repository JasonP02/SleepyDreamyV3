import torch
import torch.nn as nn

class RSSMWorldModel:
    """
    World model architecture from Dreamerv3
    * 
    """
    def __init__(
            self,
    ):
        self.encoder = CarRacingEncoder()

class CarRacingEncoder(nn.Module):

    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 stride,
                 padding
                 ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel,
                               out_channels=out_channel,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding)
