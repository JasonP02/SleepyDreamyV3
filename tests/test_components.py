import torch
from src.config import config
from src.encoder import ObservationEncoder

def test_encoder_forward_pass_and_shape():
    """
    Tests that the ObservationEncoder can perform a forward pass
    and that the output shape is correct.
    """
    batch_size = 4
    
    # Create dummy input data
    dummy_pixels = torch.randn(batch_size, 3, 64, 64)
    dummy_state = torch.randn(batch_size, config.environment.n_observations)
    dummy_input = {"pixels": dummy_pixels, "state": dummy_state}

    # Initialize the encoder
    encoder = ObservationEncoder(
        mlp_config=config.models.encoder.mlp,
        cnn_config=config.models.encoder.cnn
    )

    # Perform a forward pass
    try:
        logits = encoder(dummy_input)
    except Exception as e:
        assert False, f"Encoder forward pass failed with an exception: {e}"

    # Check the output shape
    expected_shape = (batch_size, config.models.encoder.mlp.d_hidden, config.models.encoder.mlp.d_hidden // config.models.encoder.mlp.latent_categories)
    assert logits.shape == expected_shape, f"Expected output shape {expected_shape}, but got {logits.shape}"

    print("Encoder forward pass and shape test passed!")
