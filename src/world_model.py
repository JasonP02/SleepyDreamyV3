import torch
import torch.nn as nn
import torch.nn.functional as F

class RSSMWorldModel:
    """
    World model architecture from Dreamerv3
    * 
    """
    def __init__(
            self,
    ):
        self.encoder = ObservationEncoder()

class ObservationEncoder(nn.Module):
    def __init__(self):
        self.MLP = ObservationMLPEncoder()
        self.CNN = ObservationCNNEncoder()

    def forward(self, x):
        # x is passed as a dict of ['state', 'pixels']
        image_obs = x['pixels']
        vec_obs = x['state']
        x1 = self.CNN(image_obs)
        x2 = self.MLP(vec_obs)
        x = torch.cat((x1,x2)) # concatenate the outputs
        return F.softmax(input=x, dim=0) # output a probability distribution (z)
        



class ObservationCNNEncoder(nn.Module):
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

class ObservationMLPEncoder(nn.Module):
    def __init(self):
        pass