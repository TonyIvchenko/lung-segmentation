from pathlib import Path

import numpy as np
import pytest
from PIL import Image


torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from src.data import ComposePair, LungDataset, Resize


def _write_png(path: Path, values: np.ndarray):
    Image.fromarray(values.astype(np.uint8)).save(path)


def test_dataset_builds_tensor_and_binary_mask(tmp_path):
    origins = tmp_path / "images"
    masks = tmp_path / "masks"
    origins.mkdir()
    masks.mkdir()

    _write_png(origins / "sample.png", np.array([[0, 255], [128, 64]], dtype=np.uint8))
    _write_png(masks / "sample_mask.png", np.array([[0, 255], [200, 100]], dtype=np.uint8))

    dataset = LungDataset(
        [("sample", "sample_mask")],
        origins,
        masks,
        transforms=ComposePair([Resize((2, 2))]),
        mask_threshold=128,
    )

    origin, mask = dataset[0]

    assert origin.shape == (1, 2, 2)
    assert mask.shape == (2, 2)
    assert mask.dtype == torch.long
    assert set(mask.flatten().tolist()) <= {0, 1}


def test_dataset_raises_for_missing_files(tmp_path):
    origins = tmp_path / "images"
    masks = tmp_path / "masks"
    origins.mkdir()
    masks.mkdir()

    dataset = LungDataset([("missing", "missing_mask")], origins, masks)

    with pytest.raises(FileNotFoundError):
        _ = dataset[0]
