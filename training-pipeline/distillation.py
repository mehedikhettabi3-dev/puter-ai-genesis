"""Distillation: merge an LCM-LoRA speed adapter so generation takes 4-8 steps.

    python training-pipeline/distillation.py --base stable-diffusion-xl-base-1.0 --out lora/sdxl-distilled

The core-engine applies the same trick at inference time via merge_adapters().
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

LCM_LORA = {
    "stable-diffusion-xl-base-1.0": "latent-consistency/lcm-lora-sdxl",
    "stabilityai/stable-diffusion-xl-base-1.0": "latent-consistency/lcm-lora-sdxl",
    "sdxl-base": "latent-consistency/lcm-lora-sdxl",
    "stable-diffusion-v1-5/stable-diffusion-v1-5": "latent-consistency/lcm-lora-sdv1-5",
    "sd15": "latent-consistency/lcm-lora-sdv1-5",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Genesis distillation (LCM adapter merge)")
    parser.add_argument("--base", default="stable-diffusion-v1-5/stable-diffusion-v1-5", help="repo id (or model spec id)")
    parser.add_argument("--out", required=True, help="output folder for the fused copy")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--guidance", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from genesis.config import get_model

    base_repo = LCM_LORA.get(args.base) or LCM_LORA.get(args.base.lower())
    if base_repo is None:
        from genesis.config import get_model

        try:
            spec = get_model(args.base)
            base_repo = LCM_LORA.get(spec.repo)
        except Exception:
            pass
    if not base_repo:
        print("supported bases:", ", ".join(LCM_LORA), file=sys.stderr)
        return 1

    from diffusers import LCMScheduler, StableDiffusionPipeline, StableDiffusionXLPipeline  # noqa: E402

    pipe_cls = StableDiffusionXLPipeline if "xl" in base_repo.lower() else StableDiffusionPipeline
    pipe = pipe_cls.from_pretrained(base_repo, torch_dtype=None)
    pipe.load_lora_weights(base_repo)
    pipe.fuse_lora(lora_scale=1.0)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pipe.save_pretrained(str(out))
    print(
        {"fused": str(out), "recommended_steps": args.steps, "recommended_guidance": args.guidance}
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
