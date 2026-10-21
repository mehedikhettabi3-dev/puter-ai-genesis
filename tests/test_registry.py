import pytest

from genesis.config import MODEL_REGISTRY, UnknownModel, list_models, pick_model


def test_registry_has_video_and_image_models():
    videos = [s for s in MODEL_REGISTRY.values() if s.kind == "video"]
    images = [s for s in MODEL_REGISTRY.values() if s.kind == "image"]
    assert len(videos) >= 4
    assert len(images) >= 3
    assert {"wan-t2v-1.3b", "ltx-video-2b", "cogvideox-2b"} <= {s.id for s in videos}
    assert "sdxl-base" in {s.id for s in images}


def test_every_missing_licence_metadata_is_rejected():
    for spec in MODEL_REGISTRY.values():
        assert spec.license, f"{spec.id} missing licence"
        assert spec.license_url.startswith("http"), f"{spec.id} missing licence url"
        assert isinstance(spec.commercial, bool)
        assert spec.vram_gb > 0


def test_list_models_filtered_and_sorted():
    videos = list_models("video")
    assert all(m["kind"] == "video" for m in videos)
    vram = [m["vram_gb"] for m in videos]
    assert vram == sorted(vram)

    everything = list_models()
    assert all(kind in ("video", "image") for kind in {m["kind"] for m in everything})


def test_pick_model_respects_vram_budget():
    spec = pick_model("video", vram_gb=9)
    assert spec.vram_gb <= 9
    assert pick_model("image", preferred="sd15").id == "sd15"
    with pytest.raises(UnknownModel):
        pick_model("image", preferred="nope")
