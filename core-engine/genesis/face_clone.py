"""Identity pipeline (Face-Lock): dataset -> personal LoRA -> inference."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

from .dataset import build_dataset

TRAINING_ROOT = Path(__file__).resolve().parents[2] / "training-pipeline"


def prepare_identity(
    images_dir: str | Path,
    out_dir: Optional[str | Path] = None,
    trigger: str = "genesis_identity",
    caption_suffix: str = "a photo of a person, detailed face",
    size: int = 1024,
) -> dict:
    """Step 1: clean + caption the user's photos (20-50 images is enough)."""
    source = Path(images_dir)
    destination = Path(out_dir) if out_dir else source.parent / (source.name + "_dataset")
    return build_dataset(source, destination, trigger=trigger, size=size, caption_suffix=caption_suffix)


def train_identity(
    dataset_dir: str | Path,
    output_dir: str | Path,
    base_model: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
    max_steps: int = 600,
    rank: int = 8,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    cancelled: Optional[Callable[[], bool]] = None,
) -> dict:
    """Step 2: train the personal identity LoRA (GPU required).

    Runs training-pipeline/lora_trainer.py in a subprocess and forwards its
    JSON-lines progress ({"progress": 0..1, "loss": ...}).
    """
    script = TRAINING_ROOT / "lora_trainer.py"
    if not script.is_file():
        raise FileNotFoundError(f"Trainer script missing: {script}")
    command = [
        sys.executable,
        str(script),
        "--dataset", str(dataset_dir),
        "--output_dir", str(output_dir),
        "--base", base_model,
        "--max-steps", str(int(max_steps)),
        "--rank", str(int(rank)),
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
    summary: dict = {"steps": 0}
    tail: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        line = line.strip()
        if not line:
            continue
        tail.append(line)
        tail = tail[-20:]
        try:
            event = json.loads(line)
        except Exception:
            continue
        if progress_cb is not None and "progress" in event:
            progress_cb(float(event["progress"]), event.get("message", "training"))
        summary["steps"] = int(event.get("step", summary.get("steps", 0)))
        if cancelled is not None and cancelled():
            process.terminate()
            return {"cancelled": True}
    code = process.wait()
    if code != 0:
        raise RuntimeError("LoRA training failed:\n" + "\n".join(tail))
    weights = Path(output_dir) / "pytorch_lora_weights.safetensors"
    return {
        "lora": str(weights) if weights.is_file() else str(output_dir),
        "base_model": base_model,
        **summary,
    }
