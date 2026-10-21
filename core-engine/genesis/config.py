"""Open-weights model registry. Every model must carry licence metadata."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

HF = "https://huggingface.co/"


@dataclass(frozen=True)
class ModelSpec:
    id: str
    kind: str  # "video" | "image"
    repo: str
    pipeline: str
    license: str
    license_url: str
    commercial: bool
    vram_gb: float
    width: int = 832
    height: int = 480
    frames: Optional[int] = None
    fps: Optional[int] = None
    steps: int = 30
    guidance: float = 5.0
    notes: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


MODEL_REGISTRY: dict[str, ModelSpec] = {
    spec.id: spec
    for spec in [
        # ------------------------------------------------------------ video
        ModelSpec(
            id="wan-t2v-1.3b",
            kind="video",
            repo="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            pipeline="WanPipeline",
            license="Apache-2.0",
            license_url=HF + "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            commercial=True,
            vram_gb=8,
            width=832,
            height=480,
            frames=49,
            fps=16,
            steps=30,
            guidance=5.0,
            notes="Best quality/VRAM ratio for 8-12 GB cards. Runs with model CPU offload.",
        ),
        ModelSpec(
            id="wan-t2v-14b",
            kind="video",
            repo="Wan-AI/Wan2.1-T2V-14B-Diffusers",
            pipeline="WanPipeline",
            license="Apache-2.0",
            license_url=HF + "Wan-AI/Wan2.1-T2V-14B-Diffusers",
            commercial=True,
            vram_gb=24,
            width=1280,
            height=720,
            frames=49,
            fps=16,
            steps=30,
            guidance=5.0,
            notes="Flagship open T2V model; needs 24 GB or 4-bit + offload.",
        ),
        ModelSpec(
            id="ltx-video-2b",
            kind="video",
            repo="Lightricks/LTX-Video",
            pipeline="LTXPipeline",
            license="LTXV Open Weights License",
            license_url="https://huggingface.co/Lightricks/LTX-Video/blob/main/License.txt",
            commercial=True,
            vram_gb=10,
            width=704,
            height=480,
            frames=97,
            fps=24,
            steps=40,
            guidance=3.0,
            notes="Fastest realistic T2V; commercial use allowed below the licence revenue cap.",
        ),
        ModelSpec(
            id="cogvideox-2b",
            kind="video",
            repo="THUDM/CogVideoX-2b",
            pipeline="CogVideoXPipeline",
            license="Apache-2.0",
            license_url=HF + "THUDM/CogVideoX-2b",
            commercial=True,
            vram_gb=9,
            width=720,
            height=480,
            frames=49,
            fps=8,
            steps=50,
            guidance=6.0,
            notes="Compact T2V tuned for 8-12 GB cards.",
        ),
        ModelSpec(
            id="cogvideox-5b",
            kind="video",
            repo="THUDM/CogVideoX-5b",
            pipeline="CogVideoXPipeline",
            license="CogVideoX License",
            license_url="https://huggingface.co/THUDM/CogVideoX-5b/blob/main/LICENSE",
            commercial=True,
            vram_gb=16,
            width=720,
            height=480,
            frames=49,
            fps=8,
            steps=50,
            guidance=6.0,
            notes="Higher fidelity CogVideoX; 16 GB recommended.",
        ),
        ModelSpec(
            id="hunyuanvideo",
            kind="video",
            repo="tencent/HunyuanVideo",
            pipeline="HunyuanVideoPipeline",
            license="Tencent Hunyuan Community License",
            license_url="https://github.com/Tencent/HunyuanVideo/blob/main/LICENSE",
            commercial=False,
            vram_gb=45,
            width=1280,
            height=720,
            frames=129,
            fps=24,
            steps=50,
            guidance=6.0,
            notes="Cinematic quality; licence excludes some territories and needs a 40 GB+ GPU.",
        ),
        # ------------------------------------------------------------ image
        ModelSpec(
            id="flux-dev",
            kind="image",
            repo="black-forest-labs/FLUX.1-dev",
            pipeline="FluxPipeline",
            license="FLUX.1 [dev] Non-Commercial License",
            license_url="https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md",
            commercial=False,
            vram_gb=12,
            width=1024,
            height=1024,
            steps=28,
            guidance=3.5,
            notes="Best prompt adherence in open weights; non-commercial licence.",
        ),
        ModelSpec(
            id="sd3.5-large",
            kind="image",
            repo="stabilityai/stable-diffusion-3.5-large",
            pipeline="StableDiffusion3Pipeline",
            license="Stability AI Community License",
            license_url="https://huggingface.co/stabilityai/stable-diffusion-3.5-large/blob/main/LICENSE.md",
            commercial=True,
            vram_gb=12,
            width=1024,
            height=1024,
            steps=28,
            guidance=4.5,
            notes="Commercial use below the licence revenue threshold.",
        ),
        ModelSpec(
            id="sdxl-base",
            kind="image",
            repo="stabilityai/stable-diffusion-xl-base-1.0",
            pipeline="StableDiffusionXLPipeline",
            license="CreativeML Open RAIL++-M",
            license_url=HF + "stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md",
            commercial=True,
            vram_gb=7,
            width=1024,
            height=1024,
            steps=30,
            guidance=7.0,
            notes="Backbone for identity LoRAs and IP-Adapter face lock.",
        ),
        ModelSpec(
            id="sd15",
            kind="image",
            repo="stable-diffusion-v1-5/stable-diffusion-v1-5",
            pipeline="StableDiffusionPipeline",
            license="CreativeML Open RAIL-M",
            license_url=HF + "stable-diffusion-v1-5/stable-diffusion-v1-5",
            commercial=True,
            vram_gb=4,
            width=512,
            height=512,
            steps=30,
            guidance=7.5,
            notes="Smallest base; used by the built-in LoRA trainer.",
        ),
    ]
}


class UnknownModel(KeyError):
    pass


def list_models(kind: Optional[str] = None) -> list[dict]:
    specs = list(MODEL_REGISTRY.values())
    if kind:
        specs = [s for s in specs if s.kind == kind]
    return [s.as_dict() for s in sorted(specs, key=lambda s: (s.kind, s.vram_gb))]


def get_model(model_id: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[model_id]
    except KeyError:
        raise UnknownModel(model_id) from None


def pick_model(kind: str, vram_gb: Optional[float] = None, preferred: Optional[str] = None) -> ModelSpec:
    if preferred:
        spec = get_model(preferred)
        if spec.kind != kind:
            raise ValueError(f"{preferred} is a {spec.kind} model, not {kind}")
        return spec
    candidates = sorted((s for s in MODEL_REGISTRY.values() if s.kind == kind), key=lambda s: s.vram_gb)
    if not candidates:
        raise UnknownModel(kind)
    if vram_gb is None:
        return candidates[0]
    for spec in candidates:
        if spec.vram_gb <= vram_gb:
            return spec
    return candidates[0]
