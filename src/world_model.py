import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import ObservationEncoder

class RSSMWorldModel(nn.Module):
    """
    World model architecture from Dreamerv3
    * 
    """
    def __init__(
            self,
            mlp_config,
            cnn_config,
    ):
        super().__init__()
        d_hidden = mlp_config.d_hidden
        
        self.encoder = ObservationEncoder(mlp_config=mlp_config, cnn_config=cnn_config)
        self.sequence_model = GatedRecurrentUnit(d_in=1, d_hidden=d_hidden)

        self.h_prev = torch.zeros(d_hidden)
        self.dynamics_predictor = DynamicsPredictor()

    def forward(self, x):
        h = self.sequence_model(x=x, h_prev=self.h_prev)


class DynamicsPredictor(nn.Module):
    """
    Upscales GRU to d_hidden ** 2 / 16
    Breaks the hidden state into a distribution, and set of bins 
    Takes logits over the bins (final dimension)
    """
    def __init__(self,
                 d_in,
                 d_hidden
                ):
        super().__init__()
        d_out = d_hidden**2 / 16
        self.latents = d_hidden

        self.layers = nn.Sequential(
            nn.Linear(d_in, d_hidden, bias=True),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out)
        )

    def forward(self, x):
        out = self.layers(x)
        out = out.view(out.shape[0], self.latents, self.latents//16)

        out =  F.softmax(input=out, dim=2) # output a probability distribution (z)
        return out

class GatedRecurrentUnit(nn.Module):
    """
    The GRU is the paper's recurrent model for 'dreaming' ahead
    Takes:
        h_{t-1}: Previous hidden state
        z_{t-1}: Previous stochastic representation
        z_{t-1}: Previous *action*
    Returns:
        h_{t}: Current hidden state
    z: output of encoder "ObservationEncoder" class
    a: action sampled from policy: $a_{t} ~ \pi(a_{t} | s_{t})
    
    From the paper: 
    "The input to the GRU is a linear embedding of the *sampled latent z_t*, 
    the action a_t, and the recurrent state"

    """
    def __init__(self,
                 d_in,
                 d_hidden,
                 ):
        super().__init__()
        self.d_hidden = d_hidden
        self.W_ir = nn.Linear(d_in, d_hidden, bias=True)
        self.W_hr = nn.Linear(d_hidden, d_hidden, bias=True)
        self.W_iz = nn.Linear(d_in, d_hidden, bias=True)
        self.W_hz = nn.Linear(d_hidden, d_hidden, bias=True)
        self.W_in = nn.Linear(d_in, d_hidden, bias=True)
        self.W_hn = nn.Linear(d_hidden, d_hidden, bias=True)

    def forward(self, x, h_prev=None):
        batch_size = x.shape[0]
        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.d_hidden, device=x.device, dtype=x.dtype)
    
        r = torch.sigmoid(self.W_ir(x) + self.W_hr(h_prev))
        z = torch.sigmoid(self.W_iz(x) + self.W_hz(h_prev))
        n = torch.tanh(self.W_in(x) + r * self.W_hn(h_prev))
        h = (1 - z) * n + z * h_prev
        return h




