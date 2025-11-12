import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from .encoder import ObservationEncoder
from .decoder import ObservationDecoder


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
        batch_size,
        b_start,
        b_end,
    ):
        super().__init__()
        self.d_hidden = mlp_config.d_hidden
        # Encoder
        self.encoder = ObservationEncoder(mlp_config=mlp_config, cnn_config=cnn_config)

        # GatedRecurrentUnit | Uses 8 blocks to make a pseudo-large network
        self.blocks = []
        self.n_blocks = gru_config.n_blocks
        gru_d_in = self.d_hidden + env_config.n_actions
        for _ in range(self.n_blocks):
            self.blocks.append(
                GatedRecurrentUnit(d_in=gru_d_in, d_hidden=self.d_hidden)
            )

        # Outputs prior distribution \hat{z} from the sequence model
        n_gru_blocks = gru_config.n_blocks
        self.dynamics_predictor = DynamicsPredictor(
            d_in=self.d_hidden * n_gru_blocks, d_hidden=self.d_hidden
        )

        self.n_latents = mlp_config.d_hidden
        # Initalizing network params for t=0 ; h_0 is the zero matrix
        self.h_prev = torch.zeros(batch_size, self.d_hidden * n_gru_blocks)
        self.h_prev_blocks = torch.split(self.h_prev, self.d_hidden, dim=-1)
        self.z_prev = torch.zeros((batch_size, self.d_hidden, int(self.d_hidden / 16)))

        # Linear layer to project categorical sample to embedding dimension
        self.z_embedding = nn.Linear(
            self.d_hidden * (self.d_hidden // 16), self.d_hidden
        )

        # Takes 2D categorical samples and projects to d_hidden for GRU input
        h_z_dim = (self.d_hidden * n_gru_blocks) + (
            self.d_hidden * (self.d_hidden // mlp_config.latent_categories)
        )

        # Rewards use two-hot encoding
        reward_out = abs(b_start - b_end)
        self.reward_predictor = nn.Linear(h_z_dim, reward_out)
        self.continue_predictor = nn.Linear(h_z_dim, 1)

        # Decoder. Outputs distribution of mean predictions for pixel/vetor observations
        self.decoder = ObservationDecoder(
            mlp_config=mlp_config,
            cnn_config=cnn_config,
            env_config=env_config,
            gru_config=gru_config,
        )

    def forward(self, x, a):
        """
        1. Get distribution of representations (z) from image/obs
        2. Use 'straight-through gradients' to sample from z
        3. Pass sampled representation, z, into the GRU and get the hidden state: h
        4. Pass state, h, into dynamics predictor to get the learned representation: z^
        """
        # 1. Get distribution z
        posterior_logits = self.encoder(x)  # (batch_size, latents, bins_per_latent)
        posterior_dist = dist.Categorical(logits=posterior_logits)

        # 2. Apply straight-through method (sample while keeping gradients)
        z_indices = posterior_dist.sample()  # (batch_size, latents)
        z_onehot = F.one_hot(z_indices, num_classes=self.d_hidden // 16).float()

        z_sample = z_onehot + (posterior_dist.probs - posterior_dist.probs.detach())
        bsz = z_onehot.shape[0]
        z_flat = z_sample.view(bsz, -1)
        z_embed = self.z_embedding(z_flat)

        # 3. Pass representation into GRU
        outputs = []
        for i, block in enumerate(self.blocks):
            h_i = block(z_embed, a, self.h_prev_blocks[i])
            outputs.append(h_i)

        h = torch.cat(outputs, dim=-1)

        # 4. Get leraned representation \hat{z}
        prior_logits = self.dynamics_predictor(h)
        prior_dist = dist.Categorical(logits=prior_logits)

        self.h_prev = h
        self.z_prev = (
            posterior_dist.probs
        )  # Store probabilities for next step if needed

        # The reward, continue, decoder all take the same input
        h_z_joined = self.join_h_and_z(h, z_sample)

        # Decode the prediction and hidden state back into the original space
        obs_reconstruction = self.decoder(h_z_joined)
        reward_logits = self.reward_predictor(h_z_joined)
        continue_logits = self.continue_predictor(h_z_joined)

        # Reward is categorical over bins. We return logits for CrossEntropyLoss.
        reward_dist = reward_logits

        # Continue is categorical/bernoulli.
        continue_dist = dist.Bernoulli(logits=continue_logits)

        return (
            obs_reconstruction,
            posterior_dist,
            prior_dist,
            reward_dist,
            continue_logits,
        )

    def join_h_and_z(self, h, z):
        z_flat = z.view(z.size(0), -1)
        return torch.cat([h, z_flat], dim=-1)


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
    a: action sampled from policy: a_{t} ~ pi(a_{t} | s_{t})

    "The input to the GRU is a linear embedding of the *sampled latent z_t*,
    the action a_t, and the recurrent state"
    """

    def __init__(
        self,
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
        batch_size = z.shape[0]
        x = torch.cat((z, a), dim=1)  # Join x and a in the hidden state axis
        if h_prev is None:
            h_prev = torch.zeros(
                batch_size, self.d_hidden, device=x.device, dtype=x.dtype
            )

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

    def __init__(self, d_in, d_hidden):
        super().__init__()
        d_out = int(d_hidden**2 / 16)
        self.n_latents = d_hidden

        self.layers = nn.Sequential(
            nn.Linear(d_in, d_hidden, bias=True),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x):
        out = self.layers(x)
        out = out.view(out.shape[0], self.n_latents, self.n_latents // 16)

        return out
