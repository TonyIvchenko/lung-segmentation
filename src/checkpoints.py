from pathlib import Path

import torch

from .models import PretrainedUNet, UNet


def build_model(model_name, batch_norm=False, upscale_mode="bilinear", pretrained=True):
    if model_name == "unet":
        return UNet(
            in_channels=1,
            out_channels=2,
            batch_norm=batch_norm,
            upscale_mode=upscale_mode,
        )

    if model_name == "pretrained-unet":
        return PretrainedUNet(
            in_channels=1,
            out_channels=2,
            batch_norm=batch_norm,
            upscale_mode=upscale_mode,
            pretrained=pretrained,
        )

    raise ValueError(f"Unknown model name: {model_name}")


def save_checkpoint(path, model, args=None, metrics=None, history=None):
    payload = {"model": model.state_dict()}
    if args is not None:
        payload["args"] = args
    if metrics is not None:
        payload["metrics"] = metrics
    if history is not None:
        payload["history"] = history

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path, model_name, device, batch_norm=False, upscale_mode="bilinear"):
    checkpoint = torch.load(path, map_location=device)
    model = build_model(
        model_name,
        batch_norm=batch_norm,
        upscale_mode=upscale_mode,
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model, checkpoint
