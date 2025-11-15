from threading import Thread
import torch
import os
import torch.nn.functional as F

from .environment import create_env_with_vision
from .world_model import RSSMWorldModel
from .encoder import ObservationEncoder
from .config import config
from .encoder import ThreeLayerMLP


def check_if_models_exist(encoder_path, rssm_path):
    if not os.path.exists(encoder_path) or not os.path.exists(rssm_path):
        print("Trained model files not found. Please run the training script first.")
        print(f"Expected encoder at: {encoder_path}")
        print(f"Expected RSSM at: {rssm_path}")
        return False
    return True

def load_models(encoder_path, rssm_path, device):
    encoder = ObservationEncoder(
        mlp_config=config.models.encoder.mlp,
        cnn_config=config.models.encoder.cnn,
        d_hidden=config.models.d_hidden,
    ).to(device)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))

    world_model = RSSMWorldModel(
        models_config=config.models,
        env_config=config.environment,
        batch_size=1,  # For inference, batch size is 1
        b_start=config.train.b_start,
        b_end=config.train.b_end,
    )
    world_model.load_state_dict(torch.load(rssm_path, map_location=device))
    world_model.to(device)

    actor_critic_d_in = (config.models.d_hidden * config.models.rnn.n_blocks) + (
        config.models.d_hidden
        * (config.models.d_hidden // config.models.encoder.mlp.latent_categories)
    )

    actor = ThreeLayerMLP(
        d_in=actor_critic_d_in,
        d_hidden=config.models.actor.d_hidden,
        d_out=config.environment.n_actions,
    ).to(device)

    critic = ThreeLayerMLP(
        d_in=actor_critic_d_in,
        d_hidden=config.models.critic.d_hidden,
        d_out=1,
    ).to(device)

    return encoder, world_model, actor, critic

def main():
    device = torch.device(config.general.device)
    print(f"Using device: {device}")

    env = create_env_with_vision()
    encoder_path = config.general.encoder_path
    rssm_path = config.general.rssm_path

    if not check_if_models_exist(encoder_path, rssm_path):
        return

    print("Loading trained encoder and world model...")
    encoder, world_model, actor, critic = load_models(encoder_path, rssm_path, device)

    print("Models loaded successfully.")

    for episode in range(config.train.num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        step_count = 0
        total_reward = 0

        for dream_step in range(config.train.num_dream_steps):
            action = actor(obs)
            


        # Reset world model hidden state at the start of each episode
        world_model.h_prev = torch.zeros_like(world_model.h_prev).to(device)

        while True:
            action = env.action_space.sample()
            action_onehot = F.one_hot(
                torch.tensor([action]), num_classes=config.environment.n_actions
            ).float().to(device)
            obs, reward, terminated, truncated, info = env.step(action)

            # Convert numpy arrays to torch tensors and fix format
            pixels = torch.from_numpy(obs["pixels"]).float().to(device)
            # If pixels is (H, W, C), convert to (N, C, H, W)
            if pixels.dim() == 3:  # (H, W, C)
                pixels = pixels.permute(2, 0, 1).unsqueeze(0)  # -> (1, C, H, W)

            obs_tensor = {
                "pixels": pixels,
                "state": torch.from_numpy(obs["state"])
                .float().to(device)
                .unsqueeze(0),  # Add batch dim,
            }

            posterior_logits = encoder(obs_tensor)
            posterior_dist = torch.distributions.Categorical(logits=posterior_logits)
            world_model(posterior_dist, action_onehot)

            step_count += 1
            episode_reward += float(reward)
            total_reward += float(reward)

            if terminated or truncated or step_count >= config.train.num_episodes:
                break
    env.close()


if __name__ == "__main__":
    main()
