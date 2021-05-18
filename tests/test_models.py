import pytest


torch = pytest.importorskip("torch")

from src.models import PretrainedUNet, UNet


def test_unet_output_shape_matches_input_spatial_size():
    model = UNet(in_channels=1, out_channels=2, batch_norm=True, upscale_mode="bilinear")
    inputs = torch.randn(2, 1, 128, 128)
    outputs = model(inputs)
    assert outputs.shape == (2, 2, 128, 128)


def test_pretrained_unet_can_initialize_without_downloading_weights():
    model = PretrainedUNet(
        in_channels=1,
        out_channels=2,
        batch_norm=True,
        upscale_mode="bilinear",
        pretrained=False,
    )
    inputs = torch.randn(1, 1, 128, 128)
    outputs = model(inputs)
    assert outputs.shape == (1, 2, 128, 128)
