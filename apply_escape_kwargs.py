"""Open the escape_kwargs channel so tuned operator coefficients reach the
escape operators. Pure Python -- runs in the same PowerShell .venv, no WSL.

Edits two files (backs each up to *.bak.<timestamp> first):
  src/grace_qaoa/metaheuristic/escape.py   run_escape_fair passes **op_kwargs
  src/grace_qaoa/controller/grace.py       controller stores + forwards them

Safe: exact-match anchors, idempotent (re-run does nothing), verifies every
edit and py_compiles both files. On any failure, restores from backup.

Usage (from the repo root, in the .venv PowerShell):
    python apply_escape_kwargs.py
"""
from __future__ import annotations

import io
import os
import py_compile
import re
import shutil
import sys
import time

REPO = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
ESC = os.path.join(REPO, "src", "grace_qaoa", "metaheuristic", "escape.py")
GRC = os.path.join(REPO, "src", "grace_qaoa", "controller", "grace.py")


def read(p):
    with io.open(p, encoding="utf-8") as f:
        return f.read()


def write(p, s):
    with io.open(p, "w", encoding="utf-8", newline="") as f:
        f.write(s)


def apply(label, text, old, new):
    if new in text and old not in text:
        print(f"  [skip] {label}: already applied")
        return text
    n = text.count(old)
    if n != 1:
        raise RuntimeError(
            f"{label}: expected exactly 1 match for anchor, found {n}. "
            f"File may differ from expected; no changes made.")
    print(f"  [ok]   {label}")
    return text.replace(old, new, 1)


def main():
    for f in (ESC, GRC):
        if not os.path.isfile(f):
            print(f"ERROR: not found: {f}")
            print("Run from the repo root (the folder containing src\\ and scripts\\).")
            sys.exit(1)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    esc_bak, grc_bak = f"{ESC}.bak.{stamp}", f"{GRC}.bak.{stamp}"
    shutil.copy2(ESC, esc_bak)
    shutil.copy2(GRC, grc_bak)
    print(f"Backups written:\n  {esc_bak}\n  {grc_bak}")

    try:
        # ---- escape.py ----
        esc = read(ESC)
        esc = apply(
            "escape.py signature", esc,
            "def run_escape_fair(name, cost_fn, x0, rng, max_evals, bounds=(0.0, np.pi)):",
            "def run_escape_fair(name, cost_fn, x0, rng, max_evals, bounds=(0.0, np.pi),\n"
            "                    **op_kwargs):",
        )
        esc = apply(
            "escape.py op() call", esc,
            "        out = op(capped, np.asarray(x0, dtype=float).copy(), bounds=bounds,\n"
            "                 rng=rng)",
            "        out = op(capped, np.asarray(x0, dtype=float).copy(), bounds=bounds,\n"
            "                 rng=rng, **op_kwargs)",
        )
        write(ESC, esc)

        # ---- grace.py ----
        grc = read(GRC)
        grc = apply(
            "grace.py __init__ signature", grc,
            "                 max_rounds: int = 6, seed: int = 0, escape_evals: int = 180):",
            "                 max_rounds: int = 6, seed: int = 0, escape_evals: int = 180,\n"
            "                 escape_kwargs: dict | None = None):",
        )
        b2_new = "        self.escape_kwargs = escape_kwargs or {}"
        if b2_new in grc:
            print("  [skip] grace.py __init__ body: already applied")
        else:
            grc = apply(
                "grace.py __init__ body", grc,
                "        self.escape_evals = escape_evals   # identical budget for every operator",
                "        self.escape_evals = escape_evals   # identical budget for every operator\n"
                "        self.escape_kwargs = escape_kwargs or {}   # tuned operator coefficients",
            )
        # B3: insert **self.escape_kwargs after the max_evals line (tolerant of indent)
        m = re.search(r"(?m)^(?P<ind>\s*)max_evals=self\.escape_evals,\s*$", grc)
        if m is None:
            raise RuntimeError(
                "grace.py run_escape_fair call: could not find "
                "'max_evals=self.escape_evals,' line.")
        kw_line = f"{m.group('ind')}**self.escape_kwargs,"
        if kw_line in grc:
            print("  [skip] grace.py run_escape_fair call: already applied")
        else:
            anchor = m.group(0)
            grc = grc.replace(anchor, anchor.rstrip("\n") + "\n" + kw_line, 1)
            print("  [ok]   grace.py run_escape_fair call")
        write(GRC, grc)

        # ---- verify ----
        print("Verifying...")
        py_compile.compile(ESC, doraise=True)
        py_compile.compile(GRC, doraise=True)
        print("  py_compile OK for both files")
        checks = [
            (ESC, "**op_kwargs):", "escape.py signature"),
            (ESC, "rng=rng, **op_kwargs", "escape.py op() call"),
            (GRC, "escape_kwargs: dict", "grace.py __init__ param"),
            (GRC, "self.escape_kwargs = escape_kwargs", "grace.py stores kwargs"),
            (GRC, "**self.escape_kwargs", "grace.py forwards kwargs"),
        ]
        for path, token, name in checks:
            if token in read(path):
                print(f"  [present] {name}")
            else:
                raise RuntimeError(f"verification failed: {name} missing")

    except Exception as e:
        print(f"\n!! FAILED: {e}\n   Restoring originals from backup.")
        shutil.copy2(esc_bak, ESC)
        shutil.copy2(grc_bak, GRC)
        sys.exit(1)

    print("\nSUCCESS: escape_kwargs channel is open.")
    print("Backups kept (delete when satisfied):")
    print(f"  {esc_bak}\n  {grc_bak}")


if __name__ == "__main__":
    main()
