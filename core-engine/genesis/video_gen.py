"""Text-to-video generation on open weights (Wan / LTX / CogVideoX via diffusers)."""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Callable, Optional

from .config import ModelSpec


class GenerationCancelled(RuntimeError):
    pass


def _torch():
    import torch

    return torch


class VideoGenerator:
    """Loads one video pipeline and generates mp4 files. GPU required."""

    def __init__(self, spec: ModelSpec, plan: dict):
        self.spec = spec
        self.plan = plan
        self.pipe = None

    # ------------------------------------------------------------------ load
    def load(self):
        if self.pipe is not None:
            return self.pipe
        if not self.plan.get("ok"):
            raise RuntimeError(self.plan.get("reason") or "GPU plan rejects this model")
        import diffusers

        torch = _torch()
        dtype_name = self.plan.get("dtype", "bfloat16")
        dtype = getattr(torch, dtype_name, torch.float16)
        cls = getattr(diffusers, self.spec.pipeline, None)
        if cls is None:
            raise RuntimeError(
                f"diffusers has no '{self.spec.pipeline}'. diffusers>=0.34 is required "
                f"(installed: {getattr(diffusers, '__version__', 'unknown')})."
            )
        pipe = cls.from_pretrained(self.spec.repo, torch_dtype=dtype)
        if self.plan.get("cpu_offload") and hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        if self.plan.get("vae_tiling"):
            for helper in ("enable_vae_tiling", "enable_vae_slicing"):
                if hasattr(pipe, helper):
                    getattr(pipe, helper)()
        self.pipe = pipe
        return pipe

    def unload(self) -> None:
        self.pipe = None

    # -------------------------------------------------------------- generate
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: Optional[int] = None,
        height: Optional[int] = None,
        frames: Optional[int] = None,
        fps: Optional[int] = None,
        steps: Optional[int] = None,
        guidance: Optional[float] = None,
        seed: Optional[int] = None,
        output_path: Optional[str | Path] = None,
        progress_cb: Optional[Callable[[float, str], None]] = None,
        cancelled: Optional[Callable[[], bool]] = None,
    ) -> dict:
        pipe = self.load()
        torch = _torch()

        width = int(width or self.spec.width)
        height = int(height or self.spec.height)
        frames = int(frames or self.spec.frames or 49)
        fps = int(fps or self.spec.fps or 16)
        steps = int(steps or self.spec.steps)
        guidance = float(self.spec.guidance if guidance is None else guidance)

        generator = torch.Generator(device="cpu")
        if seed is None:
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        generator.manual_seed(int(seed))

        def on_step_end(inner_pipe, step_index, _timestep, callback_kwargs):
            if cancelled is not None and cancelled():
                inner_pipe._interrupt = True
            if progress_cb is not None:
                progress_cb(min(0.95, (step_index + 1) / max(1, steps) * 0.95), f"denoise {step_index + 1}/{steps}")
            return callback_kwargs

        kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            height=height,
            width=width,
            num_frames=frames,
            guidance_scale=guidance,
            num_inference_steps=steps,
            generator=generator,
        )
        try:
            supports_callback = "callback_on_step_end" in inspect.signature(pipe.__call__).parameters
        except (TypeError, ValueError):
            supports_callback = False
        if supports_callback:
            kwargs["callback_on_step_end"] = on_step_end

        output = pipe(**kwargs)
        if cancelled is not None and cancelled():
            raise GenerationCancelled("cancelled by user")

        frames_out = getattr(output, "frames", None)
        if not frames_out:
            raise RuntimeError("pipeline returned no frames")
        pil_frames = frames_out[0] if isinstance(frames_out[0], (list, tuple)) else frames_out

        from diffusers.utils import export_to_video

        final_path = Path(output_path or f"video_{seed}.mp4")
        final_path.parent.mkdir(parents=True, exist_ok=True)
        export_to_video(pil_frames, str(final_path), fps=fps)
        if progress_cb is not None:
            progress_cb(1.0, "done")
        return {
            "path": str(final_path),
            "frames": len(pil_frames),
            "fps": fps,
            "seconds": round(len(pil_frames) / fps, 2),
            "width": width,
            "height": height,
            "seed": seed,
            "model": self.spec.id,
            "license": self.spec.license,
        }
