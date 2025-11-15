import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist

from .config import config
from .trainer_utils import symlog, symexp, twohot_encode
from .encoder import ObservationEncoder
from .world_model import RSSMWorldModel

class Loader:
    def __init__(self, batch_size, device):
        """
        Function to pull random data sample from our simulation for training. Supports batching
        Sequences are continuous from the start time.
        """
        self.bsz = batch_size
        self.device = device

    def sample(self):
        pass


def train_world_model():
    """
    The terminology can get confusing.
    p(z|h,x) is the posterior from the perspective of the world model.
    """
    device = torch.device(config.general.device)
    print(f"Using device: {device}")

    encoder = ObservationEncoder(
        mlp_config=config.models.encoder.mlp,
        cnn_config=config.models.encoder.cnn,
        d_hidden=config.models.d_hidden,
    ).to(device)

    world_model = RSSMWorldModel(
        models_config=config.models,
        env_config=config.environment,
        batch_size=config.train.batch_size,
        b_start=config.train.b_start,
        b_end=config.train.b_end,
    ).to(device)

    # Combine parameters for the optimizer
    all_params = list(encoder.parameters()) + list(world_model.parameters())
    optimizer = optim.Adam(all_params, lr=1e-4, weight_decay=1e-6)
    bsz = config.train.batch_size
    loader = Loader()

    t = 0
    while t < config.train.num_time_steps
            # Get a sample from our dataloader
            pixels, states, actions, rewards, terminated = loader.sample(batch_size, device)

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

                posterior_logits = encoder(obs_t)
                posterior_dist = dist.Categorical(logits=posterior_logits)
                (
                    obs_reconstruction,
                    _, # posterior_logits are already available
                    prior_logits,
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
                prior_dist = dist.Categorical(logits=prior_logits)
                free_bits = 1.0
                l_dyn_raw = dist.kl_divergence(
                    dist.Categorical(
                        logits=posterior_dist.logits.detach()
                    ),
                    prior_dist,
                ).mean()
                l_dyn = torch.max(torch.tensor(free_bits, device=device), l_dyn_raw)

                l_rep_raw = dist.kl_divergence(
                    posterior_dist,
                    dist.Categorical(logits=prior_dist.logits.detach()),
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
        # Save both encoder and world model
        torch.save(encoder.state_dict(), config.general.encoder_path)
        torch.save(world_model.state_dict(), config.general.rssm_path)


if __name__ == "__main__":
    train_world_model()
