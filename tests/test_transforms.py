import pytest
from PIL import Image


pytest.importorskip("torchvision")

from src.data import Crop, Pad, RandomHorizontalFlipPair, RandomVerticalFlipPair


def _dummy_pair(size=(8, 8)):
    origin = Image.new("L", size)
    mask = Image.new("L", size)
    return origin, mask


def test_pad_accepts_zero_padding():
    origin, mask = _dummy_pair()
    transformed_origin, transformed_mask = Pad(max_padding=0)((origin, mask))
    assert transformed_origin.size == origin.size
    assert transformed_mask.size == mask.size


def test_crop_accepts_zero_shift():
    origin, mask = _dummy_pair()
    transformed_origin, transformed_mask = Crop(max_shift=0)((origin, mask))
    assert transformed_origin.size == origin.size
    assert transformed_mask.size == mask.size


def test_flip_pairs_preserve_size():
    origin, mask = _dummy_pair()
    for transform in (RandomHorizontalFlipPair(p=1.0), RandomVerticalFlipPair(p=1.0)):
        out_origin, out_mask = transform((origin, mask))
        assert out_origin.size == origin.size
        assert out_mask.size == mask.size
