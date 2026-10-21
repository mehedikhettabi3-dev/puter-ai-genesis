import time

from fastapi.testclient import TestClient

from server import StubEngine, create_app


def make_client(tmp_path):
    return TestClient(create_app(engine=StubEngine(), output_dir=tmp_path / "outputs"))


def test_health_reports_stub_engine_and_hardware(tmp_path):
    client = make_client(tmp_path)
    health = client.get("/api/health").json()
    assert health["ok"] is True
    assert health["engine"] == "StubEngine"
    assert health["requires_gpu"] is False
    assert "backend" in health["hardware"]


def test_models_route_lists_open_weights(tmp_path):
    client = make_client(tmp_path)
    models = client.get("/api/models").json()["models"]
    ids = {m["id"] for m in models}
    assert {"wan-t2v-1.3b", "sdxl-base"} <= ids
    for m in models:
        assert m["license"] and m["license_url"]


def test_stub_generate_completes(tmp_path):
    client = make_client(tmp_path)
    created = client.post(
        "/api/generate", json={"kind": "video", "prompt": "neon medina at night"}
    ).json()
    job_id = created["job_id"]
    assert created["status"] == "queued"

    deadline = time.time() + 8
    job = None
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.15)
    assert job["status"] == "completed", job
    assert job["result"]["stub"] is True
    assert 0.99 <= job["progress"] <= 1.0
    # video kind renders no media in stub mode
    assert job["result"]["message"].startswith("Stub engine")


def test_generate_validates_input(tmp_path):
    client = make_client(tmp_path)
    assert client.post("/api/generate", json={"kind": "video", "prompt": "   "}).status_code == 400
    res = client.post("/api/generate", json={"kind": "audio", "prompt": "x"})
    assert res.status_code == 422  # pydantic kind pattern
    res = client.post("/api/generate", json={"kind": "video", "prompt": "x", "model": "ghost"})
    assert res.status_code == 400


def test_real_engine_requires_gpu(tmp_path):
    from server import DiffusersEngine

    with TestClient(create_app(engine=DiffusersEngine(), output_dir=tmp_path / "o")) as client:
        res = client.post("/api/generate", json={"kind": "video", "prompt": "x"})
        assert res.status_code == 503
        body = res.json()
        assert body["error"] == "GPU required"
        assert "hint" in body


def test_jobs_endpoints_404_on_unknown_ids(tmp_path):
    client = make_client(tmp_path)
    assert client.get("/api/jobs/bogus").status_code == 404
    assert client.post("/api/jobs/bogus/cancel").status_code == 404
