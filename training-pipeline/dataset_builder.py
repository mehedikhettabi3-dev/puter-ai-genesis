"""CLI wrapper: build a LoRA-ready identity dataset from a folder of photos.

    python training-pipeline/dataset_builder.py <src_dir> [--out datasets/identity] [--size 1024]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

GEN = Path(__file__).resolve().parents[1] / "core-engine"
sys.path.insert(0, str(GEN))

from genesis.dataset import build_dataset  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Genesis dataset builder")
    parser.add_argument("src_dir")
    parser.add_argument("--out", default=None)
    parser.add_argument("--trigger", default="genesis_identity")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--caption", default="", help="extra caption after the trigger token")
    parser.add_argument("--min-side", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out or str(Path(args.src_dir).parent / (Path(args.src_dir).name + "_dataset"))
    meta = build_dataset(
        args.src_dir,
        out_dir,
        trigger=args.trigger,
        size=args.size,
        caption_suffix=args.caption,
        min_side=args.min_side,
    )
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
