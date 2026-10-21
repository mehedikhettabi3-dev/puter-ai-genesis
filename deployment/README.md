# Genesis on free GPU (Kaggle) — zero paid APIs

Runs the real open-weights engine on Kaggle's free GPU quota (T4 x2 / P100, ~30 h/week) and exposes it to **Midox Vedios App** through a Cloudflare quick tunnel. Nothing you paste is a mock: generation happens on the remote GPU, locally you only preview the result.

## Why this route
| Your machine | Genesis option |
|---|---|
| Intel UHD 620, no CUDA → cannot run Wan2.1 locally | Kaggle free GPU (P100 16GB / T4 x2 15GB each) |
| Kaggle notebooks are 12 h sessions | weights cached in `/kaggle/working/hf_cache` → next session warm |
| Laptop must reach the GPU server | `cloudflared` quick tunnel → public HTTPS URL |

Reference solutions this follows: the Wan2.1 Kaggle recipes (`kayas881/WanGP-Kaggle`, `darkon12/Wan2GP-Kaggle` — restart-kernel pattern, low-VRAM fork by DeepBeepMeep), `Isi-dev/Google-Colab_Notebooks` Wan2.1-1.3B free-T4 notebook (81 frames ≈ 31 min on T4), and the LTX ComfyUI Kaggle+tunnel YouTube guides. Critical bit they all share: **torch must be `2.6.0+cu124`** because Kaggle's older Tesla P100 (sm_60) has no kernels in torch ≥ 2.7 (`CUDA error: no kernel image ... sm_60`).

## Steps (once ≈ 10 min, then ~4 min per session)
1. Kaggle → Add **`deployment/kaggle_genesis.ipynb`** (Data ▸ Upload) or copy the cells.
2. Settings → Accelerator **GPU T4 x2** (or P100), Internet **ON**.
3. Optional: Add-ons ▸ Secrets → secret named `HF_TOKEN` (Hugging Face token) — avoids anonymous rate limits. Kaggle requires **phone verification** before GPU access.
4. Run cells 1→6. Cell 6 prints `GENESIS_PUBLIC_URL = https://….trycloudflare.com`.
5. Midox Vedios App → **Settings ▸ Genesis server URL** → paste the URL → Save. Models list switches to `GenesisProvider` remote and jobs run on Kaggle.

## Quota etiquette (from Kaggle's own guide)
- Enable the GPU accelerator only while running; use "Stop/toggle accelerator" after you finish.
- Interactive sessions waste quota while idle — hit Stop session when leaving the laptop.
- Download big weights once; the `/kaggle/working` cache carries them to the next session.
- If pip/git install of the package is slow, `git clone https://github.com/mehedikhettabi3-dev/puter-ai-genesis` and `pip install -e .` instead.

## Troubleshooting
- `no kernel image` → torch not 2.6.0; rerun the pinned torch line from Cell 1.
- `diffusers/transformers` import error → pin to a known-good pair: `diffusers==0.34.0 transformers==4.51.3`.
- Tunnel URL changes every session (normal for quick tunnels) → rerun Cell 6, update the URL in the app Settings.
- 401/429 from Hugging Face → add the `HF_TOKEN` secret.

## Repo basics (local CPU dev)
```
pip install -r requirements.txt        # CPU dev stack (pytest, fastapi, httpx...)
python inference-server/server.py --engine stub           # stub mode, http://127.0.0.1:8500
python inference-server/server.py --engine diffusers      # real engine (needs CUDA GPU)
python -m pytest tests -q
```
Model registry, VRAM plans, licences: see `core-engine/genesis/config.py` and `gpu_manager.py`. Video generation target: `Wan2.1-T2V-1.3B (diffusers)` first, LTX-2B/CogVideoX next; distillation via `training-pipeline/distillation.py`.
