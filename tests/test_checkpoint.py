"""Tests for the crash-safe checkpoint store."""
import json

from grace_qaoa.utils.checkpoint import CheckpointStore, job_id


def test_append_and_resume(tmp_path):
    store = CheckpointStore(tmp_path / "run1")
    assert store.n_completed == 0

    jid = job_id("regular", 0, 0)
    assert not store.is_done(jid)
    store.append(jid, {"random": 0.5, "cobyla": 0.8, "grace": 0.7})
    assert store.is_done(jid)
    assert store.n_completed == 1

    # Simulate a restart: a fresh store over the same dir must see the done job.
    store2 = CheckpointStore(tmp_path / "run1")
    assert store2.is_done(jid)
    assert store2.n_completed == 1


def test_truncated_final_line_is_ignored(tmp_path):
    store = CheckpointStore(tmp_path / "run2")
    store.append(job_id("regular", 0, 0), {"grace": 0.7})
    # Append a corrupt half-written line, as a hard crash mid-write would.
    with open(store.path, "a", encoding="utf-8") as f:
        f.write('{"id": "regular|inst0|run1", "grace": 0.6')  # no newline, no close brace
    store3 = CheckpointStore(tmp_path / "run2")
    # The good record survives; the corrupt one is skipped (will be recomputed).
    assert store3.n_completed == 1
    assert store3.is_done(job_id("regular", 0, 0))


def test_load_all_records(tmp_path):
    store = CheckpointStore(tmp_path / "run3")
    store.append(job_id("regular", 0, 0), {"grace": 0.7})
    store.append(job_id("regular", 0, 1), {"grace": 0.8})
    rows = store.load_all_records()
    assert len(rows) == 2
    assert all("grace" in r for r in rows)
