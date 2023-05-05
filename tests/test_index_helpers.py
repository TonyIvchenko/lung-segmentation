from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "index_helpers.py"
SPEC = spec_from_file_location("index_helpers", MODULE_PATH)
index_helpers = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(index_helpers)


def _sample_payload():
    return {
        "images_folder": "images",
        "masks_folder": "masks",
        "pairs": [
            ["alpha", "alpha"],
            ["beta", "beta"],
            ["beta", "beta_alt"],
            ["gamma", "gamma"],
        ],
        "missing_masks": ["delta", "omega"],
    }


def test_index_pairs():
    assert len(index_helpers.index_pairs(_sample_payload())) == 4


def test_index_missing_masks():
    assert index_helpers.index_missing_masks(_sample_payload()) == ["delta", "omega"]


def test_index_pair_count():
    assert index_helpers.index_pair_count(_sample_payload()) == 4


def test_index_missing_count():
    assert index_helpers.index_missing_count(_sample_payload()) == 2


def test_index_origin_names():
    assert index_helpers.index_origin_names(_sample_payload()) == ["alpha", "beta", "beta", "gamma"]


def test_index_mask_names():
    assert index_helpers.index_mask_names(_sample_payload()) == ["alpha", "beta", "beta_alt", "gamma"]


def test_index_has_duplicate_origins():
    assert index_helpers.index_has_duplicate_origins(_sample_payload()) is True


def test_index_has_duplicate_masks():
    assert index_helpers.index_has_duplicate_masks(_sample_payload()) is False


def test_index_same_stem_pairs():
    assert index_helpers.index_same_stem_pairs(_sample_payload()) == [["alpha", "alpha"], ["beta", "beta"], ["gamma", "gamma"]]


def test_index_mismatched_stem_pairs():
    assert index_helpers.index_mismatched_stem_pairs(_sample_payload()) == [["beta", "beta_alt"]]
