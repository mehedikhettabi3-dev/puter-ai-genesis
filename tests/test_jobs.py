import time

from jobs import TERMINAL, JobStore


def test_create_get_lifecycle_and_stub_field():
    store = JobStore()
    job = store.create("video", {"prompt": "neon medina"})
    assert set(job) >= {"id", "kind", "status", "params", "progress"}
    assert job["status"] == "queued"

    fetched = store.get(job["id"])
    assert fetched["params"] == {"prompt": "neon medina"}
    assert store.get("nonesuch") is None

    time.sleep(0.01)
    updated = store.update(job["id"], status="running", progress=0.4)
    assert updated["status"] == "running" and updated["progress"] == 0.4
    assert updated["updated"] >= fetched["created"]

    done = store.update(job["id"], status="completed", result={"stub": True})
    assert done["status"] in TERMINAL


def test_cancel_is_refused_on_terminal_jobs():
    store = JobStore()
    job = store.create("image", {"prompt": "x"})
    assert store.cancel(job["id"])["status"] == "cancelled"
    assert store.cancel(job["id"])["status"] == "cancelled"


def test_list_is_newest_first():
    store = JobStore()
    first = store.create("video", {"i": 1})
    time.sleep(0.01)
    second = store.create("video", {"i": 2})
    listed = store.list()
    assert [j["id"] for j in listed[:2]] == [second["id"], first["id"]]
