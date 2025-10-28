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
    """
    Observations are compressed to 6x6 or 4x4, flattened, and joined to the MLP encoding
    The input image is (400, 600, 3)
    """
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 stride,
                 padding,
                 d_hidden,
                 ):
        super().__init__()
        self.n_channels = d_hidden / 16
        self.codes_per_latent = d_hidden / 16

        self.conv1 = nn.Conv2d(in_channels=in_channel,
                               out_channels=out_channel,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding)

class ObservationMLPEncoder(nn.Module):
    """
    Observations are encoded with 3-layer MLP
    Input state is a vector of size 8
    """
    def __init__(self,
               d_in,
               d_hidden,
               d_out,
               ):
        super().__init__()
        self.w1 = nn.Linear(d_in, d_hidden)
        self.w2 = nn.Linear(d_hidden, d_hidden)
        self.w3 = nn.Linear(d_hidden, d_out)
        self.b1 = nn.init.zeros_(d_hidden)
        self.b2 = nn.init.zeros_(d_hidden)
        self.b3 = nn.init.zeros_(d_hidden)
        
        self.act = F.relu # TODO review this
    
    def forward(self, x):
        h = self.act(self.w1 @ x + self.b1)
        h = self.act(self.w2 @ h + self.b2)
        h = self.fact(self.w3 @ h + self.b3)
        return h



