"""Identity dataset builder: resize, square-crop, caption (Pillow only)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def build_dataset(
    src_dir: str | Path,
    out_dir: str | Path,
    trigger: str = "genesis_identity",
    size: int = 1024,
    caption_suffix: str = "",
    min_side: int = 256,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> dict:
    src = Path(src_dir)
    out = Path(out_dir)
    if not src.is_dir():
        raise FileNotFoundError(f"Dataset source not found: {src}")
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(f for f in src.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS)
    built: list[str] = []
    skipped: list[dict] = []
    out_name = ""

    for index, file in enumerate(files, start=1):
        try:
            with Image.open(file) as handle:
                image = handle.convert("RGB")
                if min(image.size) < min_side:
                    skipped.append({"file": file.name, "reason": f"too small ({image.width}x{image.height})"})
                    continue
                scale = size / min(image.size)
                new_size = (round(image.width * scale), round(image.height * scale))
                image = image.resize(new_size, Image.LANCZOS)
                left = (new_size[0] - size) // 2
                top = (new_size[1] - size) // 2
                image = image.crop((left, top, left + size, top + size))
                out_name = file.stem + ".jpg"
                image.save(out / out_name, quality=92)
        except Exception as exc:  # unreadable file: report, keep going
            skipped.append({"file": file.name, "reason": str(exc)})
            continue
        caption = f"{trigger}, {caption_suffix}".strip(", ") if caption_suffix else trigger
        (out / (Path(out_name).stem + ".txt")).write_text(caption, encoding="utf-8")
        built.append(out_name)
        if progress_cb:
            progress_cb(index / max(1, len(files)))

    metadata = {
        "trigger": trigger,
        "size": size,
        "count": len(built),
        "images": built,
        "skipped": skipped,
        "created": time.time(),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata
