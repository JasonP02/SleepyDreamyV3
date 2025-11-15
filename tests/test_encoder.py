import torch
from src.encoder import (
    ThreeLayerMLP,
    ObservationCNNEncoder,
    ObservationEncoder,
)

# Batch sizes are 128


def test_encoder_mlp():
    """
    We encode the observation vectors with a 3-layer MLP
    """
    d_in = 8
    d_hidden = 128
    d_out = 128
    model = ThreeLayerMLP(
        d_in=d_in,
        d_hidden=d_hidden,
        d_out=d_out,
    )
    bsz = 128
    x_shape = (bsz, d_in)  # batch, vector
    sample_data = torch.rand(x_shape)
    out = model(sample_data)
    assert out.shape == (bsz, d_out)


def test_encoder_cnn():
    """
    We encode the observation image with a 2-stride CNN
    Checks the output shape based on input image

    """
    image = torch.randn((128, 3, 100, 100))
    model = ObservationCNNEncoder(
        target_size=(64, 64),
        in_channels=3,
        kernel_size=2,
        stride=2,
        padding=0,
        d_hidden=1024,
    )

    out = model(image)
    assert out.shape == (128, 512, 4, 4)


def test_observation_encoder():
    """
    Test of the full encoder pipeline
    It should output a probability distribution for each input observation
    """
    from src.config import config
    import torch.nn.functional as F

    encoder = ObservationEncoder(
        mlp_config=config.models.encoder.mlp,
        cnn_config=config.models.encoder.cnn,
        d_hidden=config.models.d_hidden,
    )

    image = torch.randn((128, 3, 100, 100))
    vector = torch.randn((128, 8))
    input_dict = {"pixels": image, "state": vector}
    logits = encoder(input_dict)

    # The output should be logits for a categorical distribution.
    # To check if it's a valid distribution, we can apply softmax and check if it sums to 1.
    probabilities = F.softmax(logits, dim=2)
    distributions = torch.sum(input=probabilities, dim=2)

    assert torch.allclose(distributions, torch.ones_like(distributions), atol=1e-5)
