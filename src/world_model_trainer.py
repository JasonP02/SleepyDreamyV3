import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from .config import config
from .trainer_utils import symlog
from .world_model import RSSMWorldModel


class TrajectoryDataset(Dataset):
    def __init__(self, h5_path, sequence_length: int = 50):
        self.h5_path = h5_path
        self.sequence_length = sequence_length

        with h5py.File(self.h5_path, "r") as f:
            self.n_episodes = f.attrs["n_episodes"]
            self.ep_lengths = [
                f[f"episodes/{i}"].attrs["n_steps"]
                for i in range(self.n_episodes)  # type:ignore
            ]

        self.cumulative_lengths = np.cumsum(
            [cum_len - self.sequence_length + 1 for cum_len in self.ep_lengths]
        )

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx):
        # Find which episode this index belongs to
        ep_idx = np.searchsorted(self.cumulative_lengths, idx, side="right")

        # Find the start index within that episode
        start_in_ep = idx
        if ep_idx > 0:
            start_in_ep = idx - self.cumulative_lengths[ep_idx - 1]

        end_in_ep = start_in_ep + self.sequence_length

        with h5py.File(self.h5_path, "r") as f:
            ep_group = f[f"episodes/{ep_idx}"]

            pixels = torch.from_numpy(ep_group["pixels"][start_in_ep:end_in_ep]).float()
            vec_obs = torch.from_numpy(
                ep_group["vec_obs"][start_in_ep:end_in_ep]
            ).float()
            action = torch.from_numpy(ep_group["action"][start_in_ep:end_in_ep]).long()

            # Pixels are (T, H, W, C), need to be (T, C, H, W)
            pixels = pixels.permute(0, 3, 1, 2)

            # Actions need to be one-hot encoded
            action_onehot = F.one_hot(
                action, num_classes=config.environment.n_actions
            ).float()

        return {"pixels": pixels, "state": vec_obs, "action": action_onehot}


def train_world_model():
    """
    The terminology can get confusing.
    p(z|h,x) is the posterior from the perspective of the world model.

    """
    world_model = RSSMWorldModel(
        mlp_config=config.models.encoder.mlp,
        cnn_config=config.models.encoder.cnn,
        env_config=config.environment,
        gru_config=config.models.rnn,
        decoder_config=config.models.decoder,
        batch_size=config.train.batch_size,
    )
    optimizer = optim.Adam(world_model.parameters(), lr=1e-4)

    dataset = TrajectoryDataset(
        config.general.env_bootstrapping_samples, sequence_length=50
    )
    dataloader = DataLoader(
        dataset, batch_size=config.train.batch_size, shuffle=True, num_workers=4
    )

    beta_dyn = config.train.beta_dyn
    beta_rep = config.train.beta_rep
    beta_pred = config.train.beta_pred

    for batch in dataloader:
        # Reset hidden states per trajectory
        world_model.h_prev.zero_()
        world_model.z_prev.zero_()

        pixels = batch["pixels"]  # (batch_size, sequence_length, 3, 64, 64)
        states = batch["state"]  # (batch_size, sequence_length, 8)
        states = symlog(states)
        actions = batch["action"]  # (batch_size, sequence_length, 4)

        for t in range(pixels.shape[1]):
            obs_t = {"pixels": pixels[:, t], "state": states[:, t]}
            action_t = actions[:, t]

            (
                obs_reconstruction,
                posterior_dist,
                prior_dist,
                reward_dist,
                continue_dist,
            ) = world_model(obs_t, action_t)

            # Distribution of mean pixel/observation values
            # Observation pixels are bernoulli, while observation vectors are gaussian
            pixel_dist = obs_reconstruction["pixels"]
            state_dist = obs_reconstruction["state"]
            pixel_sample = pixel_dist.sample()
            state_sample = state_dist.sample()

            pixel_target = obs_t[
                "pixels"
            ]  # pixel target is sigmoid activation over pixels
            obs_target = obs_t["state"]  # obs target is a distribution of mean values

            # There are three loss terms:
            # 1. Prediction loss: -ln p(x|z,h) - ln(p(r|z,h)) + ln(p(c|z,h))
            # -ln p(x|z,h) is trained with symlog squared loss

            # 2. Dynamics loss: max(1,KL) ; KL = KL[sg(q(z|h,x)) || p(z,h)]
            # 3. Representation Loss: max(1,KL) ; KL = KL[q(z|h,x) || sg(p(z|h))]
            # Log-likelihoods. Torch accepts logits

            term_1 = posterior_dist.probs  # q(z|h,x)
            term_2 = prior_dist.probs  # p(z|h)

            l_dyn_term = (
                -torch.distributions.kl_divergence(term_1.detach(), term_2) * beta_dyn
            )
            l_rep_term = (
                -torch.distributions.kl_divergence(term_1, term_2.detach()) * beta_rep
            )

            loss = l_pred + l_dyn_term + l_rep_term

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # test

            print(f"Loss: {loss.item()}")


if __name__ == "__main__":
    train_world_model()
