"""Crash-safety tests for CheckpointStore.

These assert the behaviour that protects long office runs from power cuts:
  - a truncated final line (hard crash mid-write) is repaired on reopen;
  - a subsequent append cannot concatenate onto the broken line;
  - provenance is written once and preserved across resumes.

Place at tests/test_checkpoint_crashsafe.py.
"""
import json

from grace_qaoa.utils.checkpoint import CheckpointStore


def test_truncated_tail_is_repaired(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    ck = d / "checkpoint.jsonl"
    with open(ck, "w", encoding="utf-8") as f:
        f.write(json.dumps({"id": "a", "grace_ar": 0.9}) + "\n")
        f.write(json.dumps({"id": "b", "grace_ar": 0.8}) + "\n")
        f.write('{"id":"c","grace_ar":0.7')   # truncated: no brace, no newline

    store = CheckpointStore(d)
    assert store.n_completed == 2                 # 'c' dropped as truncated
    assert store.is_done("a") and store.is_done("b")
    assert not store.is_done("c")


def test_append_after_repair_is_clean(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    ck = d / "checkpoint.jsonl"
    with open(ck, "w", encoding="utf-8") as f:
        f.write(json.dumps({"id": "a"}) + "\n")
        f.write('{"id":"b"')                      # truncated

    store = CheckpointStore(d)
    store.append("b", {"grace_ar": 0.5})          # must not corrupt
    rows = store.load_all_records()
    assert len(rows) == 2
    assert {r["id"] for r in rows} == {"a", "b"}
    # every line must be valid JSON
    for line in open(ck, encoding="utf-8"):
        if line.strip():
            json.loads(line)


def test_append_many_is_atomic_group(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    store = CheckpointStore(d)
    store.append_many([("x", {"v": 1}), ("y", {"v": 2}), ("z", {"v": 3})])
    assert store.n_completed == 3
    rows = store.load_all_records()
    assert {r["id"] for r in rows} == {"x", "y", "z"}


def test_provenance_written_once_and_resume_logged(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    s1 = CheckpointStore(d)
    s1.write_provenance(cfg={"escape": "aco", "refiner": "none"})
    prov1 = json.loads((d / "config_used.json").read_text(encoding="utf-8"))
    assert prov1["cfg"]["escape"] == "aco"
    created = prov1["created"]

    # Reopen (resume): original provenance preserved, resume appended.
    s2 = CheckpointStore(d)
    s2.write_provenance(cfg={"escape": "aco", "refiner": "none"})
    prov2 = json.loads((d / "config_used.json").read_text(encoding="utf-8"))
    assert prov2["created"] == created            # not clobbered
    assert len(prov2.get("resumes", [])) == 1
