import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from .config import config
from .trainer_utils import symlog, symexp, twohot_encode
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
            reward = torch.from_numpy(ep_group["reward"][start_in_ep:end_in_ep]).float()

        # The 'continue' flag is 1 if the episode is not terminated.
        # We check if the *next* step is terminal. The last step in a sequence
        # is always considered terminal for the purpose of the continue predictor.
        terminated = torch.zeros(self.sequence_length, dtype=torch.float32)
        if end_in_ep >= self.ep_lengths[ep_idx]:
            terminated[-1] = 1.0  # Episode ends within this sequence

        # Pixels are (T, H, W, C), need to be (T, C, H, W)
        pixels = pixels.permute(0, 3, 1, 2)

        # Actions need to be one-hot encoded
        action_onehot = F.one_hot(
            action, num_classes=config.environment.n_actions
        ).float()

        return {
            "pixels": pixels,
            "state": vec_obs,
            "action": action_onehot,
            "reward": reward,
            "terminated": terminated,
        }


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
        batch_size=config.train.batch_size,
        b_start=config.train.b_start,
        b_end=config.train.b_end,
    )
    optimizer = optim.Adam(world_model.parameters(), lr=1e-4)
    bsz = config.train.batch_size

    dataset_path = config.general.env_bootstrapping_samples
    dataset = TrajectoryDataset(dataset_path, sequence_length=50)

    dataloader = DataLoader(dataset, batch_size=bsz, shuffle=True)


    for batch in dataloader:
        # Reset hidden states per trajectory
        world_model.h_prev.zero_()
        world_model.z_prev.zero_()

        pixels = batch["pixels"]  # (batch_size, sequence_length, 3, 64, 64)
        states = batch["state"]  # (batch_size, sequence_length, 8)
        states = symlog(states)  # symlog vector inputs before model input
        actions = batch["action"]  # (batch_size, sequence_length, 4)
        rewards = batch["reward"]
        terminated = batch["terminated"]

        for t in range(pixels.shape[1]):
            obs_t = {"pixels": pixels[:, t], "state": states[:, t]}
            action_t = actions[:, t]
            reward_t = rewards[:, t]
            terminated_t = terminated[:, t]
            (
                obs_reconstruction,
                posterior_dist,
                prior_dist,
                reward_dist,
                continue_logits,
            ) = world_model(obs_t, action_t)

            # Distribution of mean pixel/observation values
            # Observation pixels are bernoulli, while observation vectors are gaussian
            pixel_probs = obs_reconstruction["pixels"]
            obs_pred = obs_reconstruction["state"]

            pixel_target = obs_t["pixels"]
            obs_target = obs_t["state"]
            obs_target = symlog(obs_target)  # loss in symlog space

            beta_dyn = config.train.beta_dyn
            beta_rep = config.train.beta_rep
            beta_pred = config.train.beta_pred

            # There are three loss terms:
            # 1. Prediction loss: -ln p(x|z,h) - ln(p(r|z,h)) + ln(p(c|z,h))
            # a. dynamics represetnation
            # -ln p(x|z,h) is trained with symlog squared loss
            pred_loss_vector = 1 / 2 * (obs_pred - obs_target) ** 2
            pred_loss_vector = pred_loss_vector.mean()

            bce_with_logits_loss_fn = nn.BCEWithLogitsLoss()
            # The decoder outputs logits, and the target should be in [0,1]
            pred_loss_pixel = bce_with_logits_loss_fn(input=pixel_probs, target=pixel_target / 255.0)

            # b. reward predictor 
            beta_range = torch.arange(
                start=config.train.b_start, end=config.train.b_end, device=reward_t.device
            )
            B = symexp(beta_range)
            reward_target = twohot_encode(reward_t, B)
            reward_loss_fn = nn.CrossEntropyLoss()
            reward_loss = reward_loss_fn(reward_dist, reward_target)

            # c. continue predictor
            # The target is 1 if we continue, 0 if we terminate.
            continue_target = (1.0 - terminated_t).unsqueeze(-1)
            pred_loss_continue = bce_with_logits_loss_fn(continue_logits.squeeze(-1), continue_target.squeeze(-1))

            # Prediction loss is the sum of the individual losses
            l_pred = pred_loss_pixel + pred_loss_vector + reward_loss + pred_loss_continue

            # 2. Dynamics loss: max(1,KL) ; KL = KL[sg(q(z|h,x)) || p(z,h)]
            # 3. Representation Loss: max(1,KL) ; KL = KL[q(z|h,x) || sg(p(z|h))]
            # Log-likelihoods. Torch accepts logits

            term_1 = posterior_dist.probs  # q(z|h,x)
            term_2 = prior_dist.probs  # p(z|h)
            
            # Note: The paper uses "free bits" here, which is a max(1, kl) term.
            # This is a simpler implementation without it for now.
            l_dyn = torch.distributions.kl_divergence(
                torch.distributions.Categorical(logits=posterior_dist.logits.detach()), prior_dist
            ).mean()
            l_rep = torch.distributions.kl_divergence(
                posterior_dist, torch.distributions.Categorical(logits=prior_dist.logits.detach())
            ).mean()

            loss = beta_pred * l_pred + beta_dyn * l_dyn + beta_rep * l_rep

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Loss: {loss.item()}")


if __name__ == "__main__":
    train_world_model()
