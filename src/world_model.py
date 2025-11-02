import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from .encoder import ObservationEncoder

class RSSMWorldModel(nn.Module):
    """
    World model architecture from Dreamerv3, called a 'Recurrent State Space Model'

    1. Encode scene (image, observation vector) -> Returns binned distribution of states 'z'
    2. Pass encoded states into GRU
    3. Estimate encoded distribution, hat{z}, a learned prior of the encoded representation
    """
    def __init__(
            self,
            mlp_config,
            cnn_config,
            env_config,
            gru_config,
            batch_size
    ):
        super().__init__()
        n_gru_blocks = gru_config.n_blocks
        self.d_hidden = mlp_config.d_hidden
        n_actions = env_config.n_actions
        
        # Transforms scene (image, obs) to latent distribution "z" for sampling
        self.encoder = ObservationEncoder(mlp_config=mlp_config, cnn_config=cnn_config)
        self.latents = mlp_config.d_hidden

        # Models the recurrent represetnation of the enviornment
        # The authors use 8 blocks concurrently, which is an efficient way of creating a pseudo-larger layer
        self.blocks = []
        self.n_blocks = gru_config.n_blocks
        for _ in range(self.n_blocks):
            self.blocks.append(GatedRecurrentUnit(d_in=self.d_hidden+n_actions, d_hidden=self.d_hidden))

        # Predicts the latent distribution \hat{z} from the sequence model
        self.dynamics_predictor = DynamicsPredictor(d_in=self.d_hidden*n_gru_blocks, d_hidden=self.d_hidden)

        self.h_prev = torch.zeros(batch_size, self.d_hidden * n_gru_blocks)
        self.h_prev_blocks = torch.split(self.h_prev, self.d_hidden, dim=-1)
        self.z_prev = torch.zeros((batch_size, self.d_hidden, int(self.d_hidden/16)))
        
        # Linear layer to project categorical sample to embedding dimension
        # Takes 2D categorical samples and projects to d_hidden for GRU input
        self.z_embedding = nn.Linear(int(self.d_hidden**2/16), self.d_hidden)

    def forward(self, x, a):
        """
        1. Get distribution of representations (z) from image/obs
        2. Use 'straight-through gradients' to sample from z
        3. Pass sampled representation, z, into the GRU and get the hidden state: h
        4. Pass state, h, into dynamics predictor to get the learned representation: z^
        """
        # 1. Get distribution
        z_dist = self.encoder(x)  # (batch_size, latents, bins_per_latent)
        
        # 2. Apply straight-through method
        sampling_distribution = dist.Categorical(probs=z_dist)
        z_indices = sampling_distribution.sample()  # (batch_size, latents)
        z_onehot = F.one_hot(z_indices, num_classes=self.d_hidden // 16).float()
        z_sample = z_onehot + (z_dist - z_dist.detach())
        bsz = z_onehot.shape[0]
        z_flat = z_sample.view(bsz, -1)
        z_embed = self.z_embedding(z_flat)

        # 3. Pass representation into GRU
        outputs = []
        for i, block in enumerate(self.blocks):
            h_i = block(z_embed, a, self.h_prev_blocks[i])
            outputs.append(h_i)
            
        h = torch.cat(outputs, dim=-1)

        # Predict the distribution of z (called z^)
        dynamics_prediction = self.dynamics_predictor(h)

        self.h_prev = h
        self.z_prev = z_dist
        return dynamics_prediction

class GatedRecurrentUnit(nn.Module):
    """
    The GRU is the paper's recurrent model for 'dreaming' ahead
    Takes:
        h_{t-1}: Previous hidden state
        z_{t-1}: Previous stochastic representation
        z_{t-1}: Previous *action*
    Returns:
        h_{t}: Current hidden state/16
    z: output of encoder "ObservationEncoder" class
    a: action sampled from policy: $a_{t} ~ pi(a_{t} | s_{t})
    
    From the paper: 
    "The input to the GRU is a linear embedding of the *sampled latent z_t*, 
    the action a_t, and the recurrent state"

    Authors use block-diagonal weights with 8 blocks.

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

    def forward(self, z, a, h_prev=None):
        """
        and the previous action, a_{t-1}
        'x' contains a sampled vector from the distribution z_{t-1},

        z has shape: (B, d_hidden) since it is *after sampling*
        Thus, the shape of x is 
        a has shape: (B, 4)
        """

        batch_size = z.shape[0]
        x = torch.cat((z,a), dim=1) # Join x and a in the hidden state axis
        if h_prev is None:

            h_prev = torch.zeros(batch_size, self.d_hidden, device=x.device, dtype=x.dtype)
    
        r = torch.sigmoid(self.W_ir(x) + self.W_hr(h_prev))
        z = torch.sigmoid(self.W_iz(x) + self.W_hz(h_prev))
        n = torch.tanh(self.W_in(x) + r * self.W_hn(h_prev))
        h = (1 - z) * n + z * h_prev
        return h

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
        d_out = int(d_hidden**2 / 16)
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
