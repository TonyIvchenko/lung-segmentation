import pytest


torch = pytest.importorskip("torch")

from src import checkpoints


def test_build_model_rejects_unknown_name():
    with pytest.raises(ValueError):
        checkpoints.build_model("unknown")


def test_infer_model_config_uses_checkpoint_args_when_auto():
    checkpoint = {"args": {"model": "unet", "batch_norm": True, "upscale_mode": "nearest"}}
    model_name, batch_norm, upscale_mode = checkpoints._infer_model_config(
        checkpoint=checkpoint,
        model_name=None,
        batch_norm=False,
        upscale_mode=None,
    )

    assert model_name == "unet"
    assert batch_norm is True
    assert upscale_mode == "nearest"
