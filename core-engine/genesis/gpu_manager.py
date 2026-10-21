"""Hardware detection and memory planning (pure logic + lazy torch import)."""
from __future__ import annotations

from typing import Optional


def detect_hardware() -> dict:
    info = {
        "backend": "cpu",
        "device": "cpu",
        "vram_gb": 0.0,
        "torch": None,
        "torch_cuda": None,
        "bfloat16": False,
    }
    try:
        import torch  # noqa: F401
    except Exception as exc:  # torch not installed: CPU dev mode
        info["reason"] = f"PyTorch not installed ({exc.__class__.__name__})"
        return info
    import torch

    info["torch"] = torch.__version__
    info["torch_cuda"] = torch.version.cuda
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            backend="cuda",
            device=props.name,
            vram_gb=round(props.total_memory / (1024**3), 2),
            bfloat16=bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)()),
            gpu_count=torch.cuda.device_count(),
        )
    else:
        info["reason"] = "torch.cuda.is_available() is False (no CUDA GPU)"
    return info


def plan_for(vram_gb: float, available_vram_gb: float, backend: str = "cuda", bfloat16: bool = True) -> dict:
    """Decide dtype / offload / tiling for a model on a given GPU."""
    plan = {
        "ok": True,
        "dtype": "bfloat16" if bfloat16 else "float16",
        "cpu_offload": False,
        "quantize": None,
        "vae_tiling": False,
        "reason": "",
    }
    if backend != "cuda":
        return {**plan, "ok": False, "reason": "No CUDA GPU detected. A CUDA GPU is required for diffusion generation."}
    if available_vram_gb <= 0:
        return {**plan, "ok": False, "reason": "CUDA reported, but no usable VRAM."}
    if available_vram_gb < vram_gb:
        plan["vae_tiling"] = True
        plan["cpu_offload"] = True
        if available_vram_gb < vram_gb * 0.75:
            plan["quantize"] = "4bit"
        if available_vram_gb < vram_gb * 0.45:
            plan["ok"] = False
            plan["reason"] = (
                f"Model needs ~{vram_gb:.0f} GB (quantized/offloaded); "
                f"only {available_vram_gb:.1f} GB VRAM is available."
            )
    return plan


def plan_for_model(spec_vram_gb: float, hardware: Optional[dict] = None) -> dict:
    hw = hardware or detect_hardware()
    return plan_for(spec_vram_gb, float(hw.get("vram_gb") or 0.0), str(hw.get("backend")), bool(hw.get("bfloat16")))
