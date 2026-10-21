from pathlib import Path

from genesis.dataset import IMAGE_EXTENSIONS, build_dataset


def make_image(path, size=300, color=(200, 40, 40)):
    from PIL import Image

    Image.new("RGB", (size, size), color).save(path)
    return path


def test_build_dataset_crops_captions_and_metadata(tmp_path):
    src = tmp_path / "raw"
    out = tmp_path / "dataset"
    src.mkdir()
    for i in range(1, 4):
        make_image(src / f"shot_{i:02d}.png")

    meta = build_dataset(src, out, trigger="genesis_identity", size=256, caption_suffix="a person")

    assert meta["count"] == 3
    assert meta["trigger"] == "genesis_identity"
    for name in meta["images"]:
        image_file = out / name
        assert image_file.exists()
        caption = (out / (image_file.stem + ".txt")).read_text(encoding="utf-8").strip()
        assert caption.startswith("genesis_identity")
        assert "a person" in caption
    assert (out / "metadata.json").is_file()
    stored = out / "metadata.json"
    from pathlib import Path

    data = Path(stored).read_text(encoding="utf-8")
    assert "created" in data


def test_build_dataset_skips_tiny_images(tmp_path):
    src = tmp_path / "raw2"
    out = tmp_path / "dataset2"
    src.mkdir()
    make_image(src / "good.jpg", size=400)
    make_image(src / "tiny.jpg", size=120)

    meta = build_dataset(src, out, size=256, min_side=256)

    assert [Path(p).name for p in meta["images"]] == ["good.jpg"]
    assert meta["skipped"] and meta["skipped"][0]["file"] == "tiny.jpg"


def test_extensions_cover_common_camera_formats():
    assert ".jpg" in IMAGE_EXTENSIONS and ".png" in IMAGE_EXTENSIONS
    assert ".txt" not in IMAGE_EXTENSIONS
