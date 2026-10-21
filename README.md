# Everything-Genesis (puter-ai-genesis)

Privacy-first, self-hosted generative engine on **open weights** — video, image, and
identity (face-clone) generation with **zero external APIs**. This repository implements
the vision in `project_context.md`: a community rebuild around HeyPuter's
"everything available at all times" discipline (self-hosted browser engineering,
agentic reasoning, open-source evaluations, lived usage).

## Stack

| Layer            | What it does                                             | Entrance |
|------------------|----------------------------------------------------------|----------|
| `core-engine/genesis` | model registry, GPU planning, dataset builder, video/image/face engines | `python` package |
| `training-pipeline`    | LoRA identity trainer (`lora_trainer.py`), dataset CLI, LCM distillation | CLI scripts |
| `inference-server`     | FastAPI + WebSocket job API on open weights              | `python inference-server/server.py` |
| `web-ui`               | minimal dark studio (no CDN, no trackers)                | served at `/` |
| `crazy-features`       | `genesis/crazy.py` — style presets, adapter stacking, face-lock | — |

## Quick start

```bash
pip install -r requirements.txt          # CPU dev/test
python -m pytest tests/ -q               # 100% CPU, no torch needed

# On a CUDA machine (>=10 GB VRAM recommended):
pip install -e ".[gpu,server]"
python inference-server/server.py        # http://127.0.0.1:8500
```

CPU-only machines: `python inference-server/server.py --engine stub` (API + UI flow, no real media).

## API

```
GET  /api/health              engine + hardware report
GET  /api/models              open-weights registry (licence metadata included)
POST /api/generate            { kind: video|image, prompt, model?, style?, seed?... }
GET  /api/jobs/<id>           poll a job
POST /api/jobs/<id>/cancel    cooperative cancel (pipeline._interrupt)
WS   /api/ws/<id>             push progress
POST /api/dataset             build an identity dataset from a photo folder
POST /api/train               train a personal LoRA (identity)
```

## Cold-start decision table (from the vision doc)

- **8-12 GB VRAM** → video `wan-t2v-1.3b` / image `sdxl-base` (commercial-ok, Apache-2.0)
- **12-16 GB** → `ltx-video-2b` or `cogvideox-5b`, image `flux-dev` (non-commercial licence!)
- **24 GB+** → `wan-t2v-14b`; studio card families hire video domain separately
- Identity → `prepare_identity()` → `lora_trainer.py` (SD1.5 backbone) → merge via `merge_adapters()`
- Speed → LCM distillation (`distillation.py`) for 4-8 step preview passes

## Serving the Midox app

The Midox creative-studio talks to genesis when it is up at `http://127.0.0.1:8500`
(provider probe: `/api/health`). Until then the app keeps its offline procedural engine.
