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
    def __init__(self, h5_path, sequence_length: int):
        self.h5_path = h5_path
        self.sequence_length = sequence_length

        self.indices = []
        with h5py.File(self.h5_path, "r") as f:
            self.n_episodes = f.attrs["n_episodes"]
            for ep_idx in range(self.n_episodes):
                ep_len = f[f"episodes/{ep_idx}"].attrs["n_steps"]
                if ep_len >= self.sequence_length:
                    for start_idx in range(ep_len - self.sequence_length + 1):
                        self.indices.append((ep_idx, start_idx))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ep_idx, start_in_ep = self.indices[idx]
        end_in_ep = start_in_ep + self.sequence_length

        with h5py.File(self.h5_path, "r") as f:
            ep_group = f[f"episodes/{ep_idx}"]
            ep_len = ep_group.attrs["n_steps"]

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
        if end_in_ep >= ep_len:
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
    device = torch.device(config.general.device)
    print(f"Using device: {device}")

    world_model = RSSMWorldModel(
        mlp_config=config.models.encoder.mlp,
        cnn_config=config.models.encoder.cnn,
        env_config=config.environment,
        gru_config=config.models.rnn,
        batch_size=config.train.batch_size,
        b_start=config.train.b_start,
        b_end=config.train.b_end,
    )
    world_model.to(device)
    optimizer = optim.Adam(world_model.parameters(), lr=1e-4, weight_decay=1e-6)
    bsz = config.train.batch_size

    dataset_path = config.general.env_bootstrapping_samples
    dataset = TrajectoryDataset(
        dataset_path, sequence_length=config.train.sequence_length
    )

    dataloader = DataLoader(dataset, batch_size=bsz, shuffle=True)

    num_epochs = config.train.num_bootstrap_epochs
    for epoch in range(num_epochs):
        print(f"--- Epoch {epoch + 1}/{num_epochs} ---")
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            pixels = batch["pixels"].to(
                device
            )  # (batch_size, sequence_length, 3, 64, 64)
            states = batch["state"].to(device)  # (batch_size, sequence_length, 8)
            actions = batch["action"].to(device)  # (batch_size, sequence_length, 4)
            rewards = batch["reward"].to(device)
            terminated = batch["terminated"].to(device)

            # Reset hidden states per trajectory and move to device
            world_model.h_prev = torch.zeros_like(world_model.h_prev).to(device)
            world_model.z_prev = torch.zeros_like(world_model.z_prev).to(device)

            total_loss = 0
            # Accumulators for individual loss components for logging
            total_pred_loss_pixel = 0
            total_pred_loss_vector = 0
            total_reward_loss = 0
            total_pred_loss_continue = 0
            total_l_dyn = 0
            total_l_rep = 0

            states = symlog(states)  # symlog vector inputs before model input

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
                pred_loss_pixel = bce_with_logits_loss_fn(
                    input=pixel_probs, target=pixel_target / 255.0
                )

                # b. reward predictor
                beta_range = torch.arange(
                    start=config.train.b_start,
                    end=config.train.b_end,
                    device=reward_t.device,
                )
                B = symexp(beta_range)
                reward_target = twohot_encode(reward_t, B)
                reward_loss_fn = nn.CrossEntropyLoss()
                reward_loss = reward_loss_fn(reward_dist, reward_target)

                # c. continue predictor
                # The target is 1 if we continue, 0 if we terminate.
                continue_target = (1.0 - terminated_t).unsqueeze(-1)
                pred_loss_continue = bce_with_logits_loss_fn(
                    continue_logits, continue_target
                )

                # Prediction loss is the sum of the individual losses
                l_pred = (
                    pred_loss_pixel
                    + pred_loss_vector
                    + reward_loss
                    + pred_loss_continue
                )

                # 2. Dynamics loss: max(1,KL) ; KL = KL[sg(q(z|h,x)) || p(z,h)]
                # 3. Representation Loss: max(1,KL) ; KL = KL[q(z|h,x) || sg(p(z|h))]
                # Log-likelihoods. Torch accepts logits

                # The "free bits" technique provides a minimum budget for the KL divergence.
                free_bits = 1.0
                l_dyn_raw = torch.distributions.kl_divergence(
                    torch.distributions.Categorical(
                        logits=posterior_dist.logits.detach()
                    ),
                    prior_dist,
                ).mean()
                l_dyn = torch.max(torch.tensor(free_bits, device=device), l_dyn_raw)

                l_rep_raw = torch.distributions.kl_divergence(
                    posterior_dist,
                    torch.distributions.Categorical(logits=prior_dist.logits.detach()),
                ).mean()
                l_rep = torch.max(torch.tensor(free_bits, device=device), l_rep_raw)

                loss = beta_pred * l_pred + beta_dyn * l_dyn + beta_rep * l_rep

                total_loss += loss

                # Accumulate individual losses
                total_pred_loss_pixel += pred_loss_pixel.item()
                total_pred_loss_vector += pred_loss_vector.item()
                total_reward_loss += reward_loss.item()
                total_pred_loss_continue += pred_loss_continue.item()
                total_l_dyn += l_dyn.item()
                total_l_rep += l_rep.item()

            # Perform backpropagation on the accumulated loss for the entire sequence
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:  # Print every 10 batches
                seq_len = pixels.shape[1]
                avg_loss = total_loss.item() / seq_len
                print(
                    f"Epoch {epoch + 1}, Batch {batch_idx}/{len(dataloader)}, Loss: {avg_loss:.4f}"
                )
                # make prints more useful; that is, show the loss for each term
                print(
                    f"  Pred_Pixel: {total_pred_loss_pixel / seq_len:.4f}",
                    f"  Pred_Vector: {total_pred_loss_vector / seq_len:.4f}",
                    f"  Reward: {total_reward_loss / seq_len:.4f}",
                    f"  Continue: {total_pred_loss_continue / seq_len:.4f}",
                    f"  Dyn: {total_l_dyn / seq_len:.4f}",
                    f"  Rep: {total_l_rep / seq_len:.4f}",
                )

        # Save the world model at the end of each epoch
        print(f"--- End of Epoch {epoch + 1}, saving model... ---")
        torch.save(world_model.state_dict(), config.general.world_model_path)


if __name__ == "__main__":
    train_world_model()
