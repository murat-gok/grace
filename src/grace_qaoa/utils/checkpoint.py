"""Crash-safe checkpointing for long experiment runs (power-outage resilient).

Design:
  - Each job (family, instance, run) has a stable string id.
  - Every completed job is appended IMMEDIATELY to a JSONL file and flushed +
    fsync'd to disk, so a power loss at worst loses the job in flight, never the
    ones already written.
  - On restart, completed ids are read back and skipped, so the run resumes
    exactly where it stopped.
  - A truncated final line (from a hard crash mid-write) is detected on load and
    physically removed, so the next append cannot concatenate onto a half-written
    record and corrupt two lines instead of one.
  - Run provenance (argv, config, git commit, start time) is written once,
    atomically, so a resumed run can be matched to the exact code and config
    that produced it.

JSONL (one JSON object per line) is used on purpose: appending is atomic-ish and
a truncated final line only costs that one record, which gets recomputed on
resume.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def job_id(family: str, instance: int, run: int) -> str:
    return f"{family}|inst{instance}|run{run}"


def _atomic_write(path: Path, text: str) -> None:
    """Write text to `path` atomically: write a temp file, fsync, then replace.

    os.replace is atomic on the same filesystem, so a power cut leaves either the
    old file or the complete new one -- never a half-written file.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class CheckpointStore:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "checkpoint.jsonl"
        # Repair a truncated tail from a previous hard crash BEFORE reading ids,
        # so a half-written final line cannot corrupt the next append.
        self._repair_truncated_tail()
        self._completed = self._load_completed_ids()

    # ------------------------------------------------------------------ #
    # provenance
    # ------------------------------------------------------------------ #
    def write_provenance(self, cfg: dict | None = None,
                         extra: dict | None = None) -> None:
        """Record what produced this run. Written once (atomically); a resume
        keeps the original and appends a resume timestamp instead of clobbering.
        """
        prov_path = self.run_dir / "config_used.json"
        try:
            git = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip() or None
        except Exception:
            git = None
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        if prov_path.exists():
            # Resumed run: don't overwrite the original provenance; log a resume.
            try:
                meta = json.loads(prov_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
            meta.setdefault("resumes", []).append({"at": now, "git": git})
            _atomic_write(prov_path, json.dumps(meta, indent=2, default=str))
            return
        meta = {
            "created": now,
            "argv": sys.argv,
            "git": git,
            "cfg": cfg,
        }
        if extra:
            meta.update(extra)
        _atomic_write(prov_path, json.dumps(meta, indent=2, default=str))

    # ------------------------------------------------------------------ #
    # crash-recovery internals
    # ------------------------------------------------------------------ #
    def _repair_truncated_tail(self) -> None:
        """If the last line is not valid JSON (hard crash mid-write), drop it.

        We rewrite the file with only the intact leading lines, atomically, so
        the file is always left in a clean append-ready state.
        """
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if not lines:
            return
        # Find the largest prefix of lines that are all valid JSON.
        good = []
        truncated = False
        for line in lines:
            s = line.strip()
            if not s:
                continue
            try:
                json.loads(s)
                good.append(s)
            except json.JSONDecodeError:
                # First bad line: everything from here on is suspect. Stop.
                truncated = True
                break
        if truncated or len(good) != len([l for l in lines if l.strip()]):
            _atomic_write(self.path, "\n".join(good) + ("\n" if good else ""))

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

    def append_many(self, items: list[tuple[str, dict]]) -> None:
        """Append several completed jobs in one flush+fsync.

        Cheaper than one fsync per job when a batch finishes together, while
        still guaranteeing that after this returns, every listed job is on disk.
        A crash mid-call loses at most the un-flushed tail, which resumes cleanly.
        """
        if not items:
            return
        with open(self.path, "a", encoding="utf-8") as f:
            for jid, record in items:
                f.write(json.dumps({"id": jid, **record}) + "\n")
            f.flush()
            os.fsync(f.fileno())
        for jid, _ in items:
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
