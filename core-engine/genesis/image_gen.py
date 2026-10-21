"""Text-to-image generation on open weights (Flux / SD3.5 / SDXL via diffusers)."""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Callable, Optional

from .config import ModelSpec
from .video_gen import GenerationCancelled


class ImageGenerator:
    """Loads one image pipeline and generates png files. GPU required."""

    def __init__(self, spec: ModelSpec, plan: dict):
        self.spec = spec
        self.plan = plan
        self.pipe = None

    def load(self):
        if self.pipe is not None:
            return self.pipe
        if not self.plan.get("ok"):
            raise RuntimeError(self.plan.get("reason") or "GPU plan rejects this model")
        import diffusers

        torch = _torch()
        dtype = getattr(torch, self.plan.get("dtype", "bfloat16"), torch.float16)
        cls = getattr(diffusers, self.spec.pipeline, None)
        if cls is None:
            raise RuntimeError(f"diffusers has no '{self.spec.pipeline}'. Update diffusers.")
        pipe = cls.from_pretrained(self.spec.repo, torch_dtype=dtype)
        if self.plan.get("cpu_offload") and hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        self.pipe = pipe
        return pipe

    def unload(self) -> None:
        self.pipe = None

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: Optional[int] = None,
        height: Optional[int] = None,
        steps: Optional[int] = None,
        guidance: Optional[float] = None,
        seed: Optional[int] = None,
        ip_adapter_image: Optional[str] = None,
        ip_adapter_scale: float = 0.7,
        output_path: Optional[str | Path] = None,
        progress_cb: Optional[Callable[[float, str], None]] = None,
        cancelled: Optional[Callable[[], bool]] = None,
    ) -> dict:
        import PIL.Image

        pipe = self.load()
        torch = _torch()

        width = int(width or self.spec.width)
        height = int(height or self.spec.height)
        steps = int(steps or self.spec.steps)
        guidance = float(self.spec.guidance if guidance is None else guidance)

        generator = torch.Generator(device="cpu")
        if seed is None:
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        generator.manual_seed(int(seed))

        if ip_adapter_image:
            face = PIL.Image.open(ip_adapter_image).convert("RGB")
            try:
                pipe.load_ip_adapter(
                    "h94/IP-Adapter",
                    subfolder="sdxl_models",
                    weight_name="ip-adapter-plus-face_sdxl_vit-h.safetensors",
                )
                pipe.set_ip_adapter_scale(float(ip_adapter_scale))
            except Exception as exc:
                raise RuntimeError(f"IP-Adapter face lock unavailable for this pipeline: {exc}") from exc

        kwargs = dict(
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=guidance,
            num_inference_steps=steps,
            generator=generator,
        )
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt
        if ip_adapter_image:
            kwargs["ip_adapter_image"] = face

        try:
            supports_callback = "callback_on_step_end" in inspect.signature(pipe.__call__).parameters
        except (TypeError, ValueError):
            supports_callback = False
        if supports_callback:

            def on_step_end(inner_pipe, step_index, _timestep, callback_kwargs):
                if cancelled is not None and cancelled():
                    inner_pipe._interrupt = True
                if progress_cb is not None:
                    progress_cb(min(0.95, (step_index + 1) / max(1, steps) * 0.95), f"denoise {step_index + 1}/{steps}")
                return callback_kwargs

            kwargs["callback_on_step_end"] = on_step_end

        output = pipe(**kwargs)
        if cancelled is not None and cancelled():
            raise GenerationCancelled("cancelled by user")

        images = getattr(output, "images", None)
        if not images:
            raise RuntimeError("pipeline returned no images")
        final_path = Path(output_path or f"image_{seed}.png")
        final_path.parent.mkdir(parents=True, exist_ok=True)
        images[0].save(final_path)
        if progress_cb is not None:
            progress_cb(1.0, "done")
        return {
            "path": str(final_path),
            "width": width,
            "height": height,
            "seed": seed,
            "model": self.spec.id,
            "license": self.spec.license,
            "face_lock": bool(ip_adapter_image),
        }


def _torch():
    import torch

    return torch
