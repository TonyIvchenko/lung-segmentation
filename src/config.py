from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TrainConfig:
    data_folder: Path = Path("input/dataset")
    splits: Path = Path("splits.pk")
    output: Path = Path("models/unet-cli.pt")
    history_output: Optional[Path] = None
    model: str = "pretrained-unet"
    epochs: int = 50
    batch_size: int = 4
    learning_rate: float = 5e-4
    image_size: int = 512
    num_workers: int = 0
    seed: int = 42
    upscale_mode: str = "bilinear"
    batch_norm: bool = False
    pretrained_encoder: bool = True
    cpu: bool = False

    def to_dict(self):
        return asdict(self)
