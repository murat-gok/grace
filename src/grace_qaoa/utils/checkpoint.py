"""Crash-safe checkpointing for long experiment runs (power-outage resilient).

Design:
  - Each job (family, instance, run) has a stable string id.
  - Every completed job is appended IMMEDIATELY to a JSONL file and flushed +
    fsync'd to disk, so a power loss at worst loses the job in flight, never the
    ones already written.
  - On restart, completed ids are read back and skipped, so the run resumes
    exactly where it stopped.

JSONL (one JSON object per line) is used on purpose: appending is atomic-ish and
a truncated final line (from a hard crash mid-write) only costs that one record,
which gets recomputed on resume.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


def job_id(family: str, instance: int, run: int) -> str:
    return f"{family}|inst{instance}|run{run}"


class CheckpointStore:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "checkpoint.jsonl"
        self._completed = self._load_completed_ids()

    def _load_completed_ids(self) -> set[str]:
        done: set[str] = set()
        if not self.path.exists():
            return done
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    # Truncated final line from a hard crash: ignore, will redo.
                    continue
                if "id" in rec:
                    done.add(rec["id"])
        return done

    def is_done(self, jid: str) -> bool:
        return jid in self._completed

    @property
    def n_completed(self) -> int:
        return len(self._completed)

    def append(self, jid: str, record: dict) -> None:
        """Append one completed job and force it to physical disk."""
        record = {"id": jid, **record}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
            os.fsync(f.fileno())   # the line that survives a power cut
        self._completed.add(jid)

    def load_all_records(self) -> list[dict]:
        """Read back every completed record (for final aggregation)."""
        rows = []
        if not self.path.exists():
            return rows
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows
