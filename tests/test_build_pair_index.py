import pytest
import json

from scripts import build_pair_index


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_build_pairs_matches_identical_stems(tmp_path):
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    _touch(images / "case001.png")
    _touch(images / "case002.png")
    _touch(masks / "case001.png")
    _touch(masks / "case003.png")

    pairs, missing = build_pair_index.build_pairs(images, masks)

    assert pairs == [["case001", "case001"]]
    assert missing == ["case003"]


def test_build_pairs_supports_mask_suffix(tmp_path):
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    _touch(images / "case001.png")
    _touch(masks / "case001_mask.png")

    pairs, missing = build_pair_index.build_pairs(images, masks, mask_suffix="_mask")

    assert pairs == [["case001", "case001_mask"]]
    assert missing == []


def test_build_pairs_strict_mode_raises_on_missing_masks(tmp_path):
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    _touch(images / "case001.png")
    _touch(masks / "case002.png")

    with pytest.raises(ValueError):
        build_pair_index.build_pairs(images, masks, strict=True)


def test_main_writes_pair_index_payload(tmp_path, monkeypatch):
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    output = tmp_path / "pairs.json"
    _touch(images / "case001.png")
    _touch(masks / "case001.png")

    monkeypatch.setattr(
        "sys.argv",
        [
            "build_pair_index.py",
            "--images-folder",
            str(images),
            "--masks-folder",
            str(masks),
            "--output",
            str(output),
        ],
    )
    build_pair_index.main()

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["pairs"] == [["case001", "case001"]]
    assert payload["missing_masks"] == []
