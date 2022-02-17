import pytest

from scripts import make_splits


def test_origin_name_from_mask_with_default_suffix():
    assert make_splits.origin_name_from_mask("case001_mask", "_mask") == "case001"


def test_origin_name_from_mask_with_empty_suffix():
    assert make_splits.origin_name_from_mask("case001", "") == "case001"


def test_origin_name_from_mask_raises_for_suffix_mismatch():
    with pytest.raises(ValueError):
        make_splits.origin_name_from_mask("case001", "_mask")


def test_parse_args_accepts_custom_naming_options(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "make_splits.py",
            "--mask-suffix",
            "",
            "--image-ext",
            ".jpg",
            "--mask-ext",
            ".png",
            "--strict-pairs",
        ],
    )
    args = make_splits.parse_args()
    assert args.mask_suffix == ""
    assert args.image_ext == ".jpg"
    assert args.mask_ext == ".png"
    assert args.strict_pairs is True
