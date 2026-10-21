"""Everything-Genesis inference server: FastAPI + WebSocket, zero cloud APIs.

Run (GPU machine):   python inference-server/server.py
Run (CPU dev mode):  python inference-server/server.py --engine stub
Open:                http://127.0.0.1:8500
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT / "core-engine"), str(ROOT / "inference-server")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from fastapi import FastAPI, HTTPException, WebSocket  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from genesis.config import get_model, list_models, pick_model  # noqa: E402
from genesis.dataset import build_dataset  # noqa: E402
from genesis.gpu_manager import detect_hardware, plan_for_model  # noqa: E402
from jobs import TERMINAL, JobStore  # noqa: E402


# --------------------------------------------------------------------- engines
class BaseEngine:
    requires_gpu = True

    def hardware(self) -> dict:
        return detect_hardware()

    def run(self, job: dict, progress_cb, cancelled, outputs_dir: Path) -> dict:  # pragma: no cover
        raise NotImplementedError


class DiffusersEngine(BaseEngine):
    """Real generation on open weights. Needs torch + diffusers + a CUDA GPU."""

    requires_gpu = True

    def __init__(self) -> None:
        self._video: dict[str, object] = {}
        self._image: dict[str, object] = {}

    def run(self, job, progress_cb, cancelled, outputs_dir: Path) -> dict:
        from genesis.crazy import apply_style

        params = job["params"]
        kind = job["kind"]
        spec = pick_model(kind, preferred=params.get("model") or None)
        plan = plan_for_model(spec.vram_gb, self.hardware())
        if not plan.get("ok"):
            raise RuntimeError(plan.get("reason") or "GPU too small for this model")
        prompt, negative = apply_style(
            params.get("prompt", ""), params.get("style"), params.get("negative_prompt", "")
        )
        seed = params.get("seed")
        target = Path(outputs_dir) / f"{kind}_{job['id']}.{('mp4' if kind == 'video' else 'png')}"

        def cb(fraction, message):
            progress_cb(float(fraction), message)

        if kind == "video":
            from genesis.video_gen import VideoGenerator

            key = spec.id
            if key not in self._video:
                self._video[key] = VideoGenerator(spec, plan)
            result = self._video[key].generate(
                prompt=prompt,
                negative_prompt=negative,
                width=params.get("width"),
                height=params.get("height"),
                frames=params.get("frames"),
                fps=params.get("fps"),
                steps=params.get("steps"),
                guidance=params.get("guidance"),
                seed=seed,
                output_path=target,
                progress_cb=cb,
                cancelled=cancelled,
            )
        elif kind == "image":
            from genesis.image_gen import ImageGenerator

            key = spec.id
            if key not in self._image:
                self._image[key] = ImageGenerator(spec, plan)
            result = self._image[key].generate(
                prompt=prompt,
                negative_prompt=negative,
                width=params.get("width"),
                height=params.get("height"),
                steps=params.get("steps"),
                guidance=params.get("guidance"),
                seed=seed,
                ip_adapter_image=params.get("face_image"),
                output_path=target,
                progress_cb=cb,
                cancelled=cancelled,
            )
        else:
            raise ValueError(f"unsupported kind: {kind}")

        result["url"] = "/outputs/" + target.name
        # Free VRAM between jobs (single worker queue).
        if kind == "video" and hasattr(self._video.get(spec.id), "unload"):
            self._video[spec.id].unload()
        if kind == "image" and hasattr(self._image.get(spec.id), "unload"):
            self._image[spec.id].unload()
        return result


class StubEngine(BaseEngine):
    """CPU dev mode: verifies the API/UI flow without pretending to generate."""

    requires_gpu = False

    def hardware(self) -> dict:
        return {**detect_hardware(), "engine": "stub"}

    def run(self, job, progress_cb, cancelled, outputs_dir: Path) -> dict:
        for step in range(3):
            if cancelled():
                raise RuntimeError("cancelled")
            progress_cb((step + 1) / 3, f"stub step {step + 1}/3")
            time.sleep(0.3)
        return {
            "stub": True,
            "path": None,
            "url": None,
            "message": "Stub engine (CPU dev mode). Install the GPU stack to generate real media: pip install -e .[gpu]",
        }


# ------------------------------------------------------------------------ app
class GenerateRequest(BaseModel):
    kind: str = Field(pattern="^(video|image)$")
    prompt: str
    model: str | None = None
    negative_prompt: str = ""
    style: str | None = None
    width: int | None = None
    height: int | None = None
    frames: int | None = None
    fps: int | None = None
    steps: int | None = None
    guidance: float | None = None
    seed: int | None = None
    face_image: str | None = None


def create_app(engine: BaseEngine | None = None, output_dir: str | Path | None = None) -> FastAPI:
    engine = engine or DiffusersEngine()
    store = JobStore()
    outputs = Path(output_dir or ROOT / "outputs")
    outputs.mkdir(parents=True, exist_ok=True)
    cancel_events: dict[str, threading.Event] = {}
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="genesis")

    app = FastAPI(title="Everything-Genesis", version="0.1.0", docs_url="/api/docs")

    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def progress_callback(job_id: str):
        def cb(fraction: float, message: str = "") -> None:
            store.update(job_id, progress=round(max(0.0, min(1.0, fraction)), 4), message=str(message))

        return cb

    def work(job_id: str) -> None:
        store.update(job_id, status="running")
        event = threading.Event()
        cancel_events[job_id] = event
        try:
            result = engine.run(
                store.get(job_id), progress_callback(job_id), event.is_set, outputs
            )
            store.update(job_id, status="completed", progress=1.0, result=result)
        except Exception as exc:  # worker errors land in the job, not the server
            if event.is_set():
                store.update(job_id, status="cancelled")
            else:
                store.update(job_id, status="failed", error=f"{exc.__class__.__name__}: {exc}")
                traceback.print_exc()
        finally:
            cancel_events.pop(job_id, None)

    def guard_gpu() -> None:
        if not engine.requires_gpu:
            return
        hardware = engine.hardware()
        if hardware.get("backend") != "cuda":
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "GPU required",
                    "hardware": hardware,
                    "hint": "Run this server on a CUDA machine (or Kaggle/Colab notebook), "
                    "or start CPU dev mode with --engine stub.",
                },
            )

    # ------------------------------------------------------------------ routes
    @app.get("/api/health")
    def health():
        return {
            "ok": True,
            "name": "Everything-Genesis",
            "version": "0.1.0",
            "engine": engine.__class__.__name__,
            "requires_gpu": engine.requires_gpu,
            "hardware": engine.hardware(),
        }

    @app.get("/api/models")
    def models():
        return {"models": list_models()}

    @app.post("/api/generate")
    def generate(request: GenerateRequest):
        if not request.prompt.strip():
            raise HTTPException(400, "prompt is empty")
        if request.model:
            try:
                get_model(request.model)  # unknown ids fail early
            except Exception:
                raise HTTPException(400, f"unknown model: {request.model}") from None
        guard_gpu()
        job = store.create(request.kind, request.model_dump())
        executor.submit(work, job["id"])
        return {"job_id": job["id"], "status": job["status"]}

    @app.post("/api/train")
    def train(payload: dict):
        dataset_dir = payload.get("dataset_dir")
        if not dataset_dir or not Path(dataset_dir).is_dir():
            raise HTTPException(400, "dataset_dir missing (build one with genesis.face_clone.prepare_identity)")
        guard_gpu()
        out_name = payload.get("name") or "identity_lora"
        job = store.create("face_lora", {**payload, "output_dir": str(ROOT / "lora" / out_name)})

        def run_training(job_id: str) -> None:
            store.update(job_id, status="running")
            event = threading.Event()
            cancel_events[job_id] = event
            try:
                from genesis.face_clone import train_identity

                result = train_identity(
                    dataset_dir,
                    job["params"]["output_dir"],
                    base_model=payload.get("base_model", "stable-diffusion-v1-5/stable-diffusion-v1-5"),
                    max_steps=int(payload.get("max_steps", 600)),
                    progress_cb=progress_callback(job_id),
                    cancelled=event.is_set,
                )
                store.update(job_id, status="completed", progress=1.0, result=result)
            except Exception as exc:
                if event.is_set():
                    store.update(job_id, status="cancelled")
                else:
                    store.update(job_id, status="failed", error=f"{exc.__class__.__name__}: {exc}")
            finally:
                cancel_events.pop(job_id, None)

        executor.submit(run_training, job["id"])
        return {"job_id": job["id"], "status": job["status"]}

    @app.get("/api/jobs")
    def jobs_list():
        return {"jobs": store.list()}

    @app.get("/api/jobs/{job_id}")
    def jobs_get(job_id: str):
        job = store.get(job_id)
        if not job:
            raise HTTPException(404, "job not found")
        return job

    @app.post("/api/jobs/{job_id}/cancel")
    def jobs_cancel(job_id: str):
        event = cancel_events.get(job_id)
        if event:
            event.set()
        job = store.cancel(job_id)
        if not job:
            raise HTTPException(404, "job not found")
        return job

    @app.post("/api/dataset")
    def dataset(payload: dict):
        images_dir = payload.get("images_dir")
        if not images_dir:
            raise HTTPException(400, "images_dir required")
        try:
            meta = build_dataset(
                images_dir,
                payload.get("out_dir") or str(ROOT / "datasets" / "identity"),
                trigger=payload.get("trigger", "genesis_identity"),
                size=int(payload.get("size", 1024)),
            )
        except Exception as exc:
            raise HTTPException(400, str(exc)) from None
        return meta

    @app.exception_handler(HTTPException)
    async def http_error(_request, exc: HTTPException):
        detail = exc.detail if isinstance(exc.detail, dict) else {"error": exc.detail}
        return JSONResponse(status_code=exc.status_code, content=detail)

    @app.websocket("/api/ws/{job_id}")
    async def ws_job(websocket: WebSocket, job_id: str):
        import asyncio

        await websocket.accept()
        try:
            while True:
                job = store.get(job_id)
                if job is None:
                    await websocket.send_json({"error": "job not found"})
                    break
                await websocket.send_json(job)
                if job["status"] in TERMINAL:
                    break
                await asyncio.sleep(0.25)
        finally:
            try:
                await websocket.close()
            except Exception:
                pass

    # ----------------------------------------------------------------- static
    app.mount("/outputs", StaticFiles(directory=str(outputs)), name="outputs")
    web_ui = ROOT / "web-ui"
    if web_ui.is_dir():
        app.mount("/", StaticFiles(directory=str(web_ui), html=True), name="ui")
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Everything-Genesis inference server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8500)
    parser.add_argument("--engine", choices=["diffusers", "stub"], default="diffusers")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    engine = StubEngine() if args.engine == "stub" else DiffusersEngine()
    hardware = engine.hardware()
    print(f"[genesis] backend={hardware.get('backend')} device={hardware.get('device')}")
    if engine.requires_gpu and hardware.get("backend") != "cuda":
        print("[genesis] WARNING: no CUDA GPU — realtime generation will be refused; use --engine stub for dev mode.")

    import uvicorn

    uvicorn.run(create_app(engine, args.output_dir), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
