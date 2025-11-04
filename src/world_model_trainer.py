import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as dist
import h5py
from torch.utils.data import Dataset, DataLoader
from .config import config
from .world_model import RSSMWorldModel
import numpy as np

class TrajectoryDataset(Dataset):
    def __init__(self, h5_path, sequence_length=50):
        self.h5_path = h5_path
        self.sequence_length = sequence_length
        
        with h5py.File(self.h5_path, 'r') as f:
            self.n_episodes = f.attrs['n_episodes']
            self.ep_lengths = [f[f'episodes/{i}'].attrs['n_steps'] for i in range(self.n_episodes)]
        
        self.cumulative_lengths = np.cumsum([l - self.sequence_length + 1 for l in self.ep_lengths])

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx):
        # Find which episode this index belongs to
        ep_idx = np.searchsorted(self.cumulative_lengths, idx, side='right')
        
        # Find the start index within that episode
        start_in_ep = idx
        if ep_idx > 0:
            start_in_ep = idx - self.cumulative_lengths[ep_idx - 1]
            
        end_in_ep = start_in_ep + self.sequence_length

        with h5py.File(self.h5_path, 'r') as f:
            ep_group = f[f'episodes/{ep_idx}']
            
            pixels = torch.from_numpy(ep_group['pixels'][start_in_ep:end_in_ep]).float()
            vec_obs = torch.from_numpy(ep_group['vec_obs'][start_in_ep:end_in_ep]).float()
            action = torch.from_numpy(ep_group['action'][start_in_ep:end_in_ep]).long()
            
            # Pixels are (T, H, W, C), need to be (T, C, H, W)
            pixels = pixels.permute(0, 3, 1, 2)
            
            # Actions need to be one-hot encoded
            action_onehot = F.one_hot(action, num_classes=config.environment.n_actions).float()

        return {
            'pixels': pixels,
            'state': vec_obs,
            'action': action_onehot
        }

def train_world_model():
    world_model = RSSMWorldModel(
        mlp_config=config.models.encoder.mlp,
        cnn_config=config.models.encoder.cnn,
        env_config=config.environment,
        gru_config=config.models.rnn,
        decoder_config=config.models.decoder,
        batch_size=config.train.batch_size
    )
    optimizer = optim.Adam(world_model.parameters(), lr=1e-4)

    dataset = TrajectoryDataset(config.general.env_bootstrapping_samples, sequence_length=50)
    dataloader = DataLoader(dataset, batch_size=config.train.batch_size, shuffle=True, num_workers=4)

    for batch in dataloader:
        pixels = batch['pixels'] # (batch_size, sequence_length, 3, 64, 64)
        states = batch['state'] # (batch_size, sequence_length, 8)
        actions = batch['action'] # (batch_size, sequence_length, 4)

        for t in range(pixels.shape[1]):
            world_model.h_prev.zero_()
            world_model.z_prev.zero_()

            obs_t = {'pixels': pixels[:, t], 'state': states[:, t]}
            action_t = actions[:, t]

            x_reconstructed = world_model(obs_t, action_t)

            

if __name__ == "__main__":
    train_world_model()