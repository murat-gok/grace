"""Fix the Windows 'Bad file descriptor' crash in _atomic_torch_save.

Root cause: torch.save(obj, tmp_path) opens AND closes its own file handle when
given a path. Calling os.fsync on a separately-opened (or already-closed) handle
afterwards fails on Windows with OSError [Errno 9]. The fix writes torch's bytes
into a handle WE own, so the handle is still valid when we fsync it.

This script:
  - locates _atomic_torch_save in scripts/pretrain_gnn.py
  - backs the file up to *.bak.<timestamp>
  - replaces only the function body, exact-match (idempotent: re-run is a no-op)
  - verifies the new body is present and py_compiles the file
  - also sanity-checks checkpoint.py's _atomic_write (should already be correct)
  - restores from backup on any failure

Run from the repo root, in the .venv PowerShell:
    python fix_atomic_save.py
"""
from __future__ import annotations

import io
import os
import py_compile
import shutil
import sys
import time

REPO = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
PRE = os.path.join(REPO, "scripts", "pretrain_gnn.py")
CKP = os.path.join(REPO, "src", "grace_qaoa", "utils", "checkpoint.py")


def read(p):
    with io.open(p, encoding="utf-8") as f:
        return f.read()


def write(p, s):
    with io.open(p, "w", encoding="utf-8", newline="") as f:
        f.write(s)


# The corrected body we want present, regardless of which broken variant exists.
GOOD_BODY = (
    "    path = Path(path)\n"
    "    tmp = path.with_suffix(path.suffix + \".tmp\")\n"
    "    with open(tmp, \"wb\") as f:\n"
    "        torch.save(obj, f)\n"
    "        f.flush()\n"
    "        os.fsync(f.fileno())\n"
    "    os.replace(tmp, path)\n"
)

# Known broken variants seen in the wild (path-form torch.save + stray fsync).
BROKEN_VARIANTS = [
    # variant A: reopen in "rb" and fsync
    (
        "    path = Path(path)\n"
        "    tmp = path.with_suffix(path.suffix + \".tmp\")\n"
        "    torch.save(obj, tmp)\n"
        "    with open(tmp, \"rb\") as f:\n"
        "        os.fsync(f.fileno())\n"
        "    os.replace(tmp, path)\n"
    ),
    # variant B: same without Path(path) line
    (
        "    tmp = path.with_suffix(path.suffix + \".tmp\")\n"
        "    torch.save(obj, tmp)\n"
        "    with open(tmp, \"rb\") as f:\n"
        "        os.fsync(f.fileno())\n"
        "    os.replace(tmp, path)\n"
    ),
]


def main():
    if not os.path.isfile(PRE):
        print(f"ERROR: not found: {PRE}\nRun from the repo root.")
        sys.exit(1)

    src = read(PRE)

    # Already fixed?
    if GOOD_BODY in src:
        print("[skip] pretrain_gnn.py already has the corrected _atomic_torch_save.")
        _check_checkpoint()
        _verify_compile()
        print("\nNothing to do.")
        return

    # Find which broken variant is present.
    target = None
    for i, v in enumerate(BROKEN_VARIANTS):
        if v in src:
            target = v
            print(f"Found broken _atomic_torch_save (variant {chr(65+i)}).")
            break

    if target is None:
        print("Could not find a known broken _atomic_torch_save body.")
        print("The function may have been edited by hand. No changes made.")
        print("Paste the function here and I'll match it exactly:")
        # show the function so the user can copy it
        idx = src.find("def _atomic_torch_save")
        if idx != -1:
            end = src.find("\ndef ", idx + 1)
            print("---")
            print(src[idx: end if end != -1 else idx + 500])
            print("---")
        sys.exit(2)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    bak = f"{PRE}.bak.{stamp}"
    shutil.copy2(PRE, bak)
    print(f"Backup written: {bak}")

    try:
        new_src = src.replace(target, GOOD_BODY, 1)
        if new_src == src:
            raise RuntimeError("replacement produced no change")
        write(PRE, new_src)

        if GOOD_BODY not in read(PRE):
            raise RuntimeError("verification failed: corrected body not present")
        py_compile.compile(PRE, doraise=True)
        print("[ok] pretrain_gnn.py: _atomic_torch_save fixed and compiles.")
    except Exception as e:
        print(f"!! FAILED: {e}. Restoring from backup.")
        shutil.copy2(bak, PRE)
        sys.exit(1)

    _check_checkpoint()
    print("\nSUCCESS. Re-run the training:")
    print("  python scripts\\pretrain_gnn.py --config configs\\hard2gnn_p3.yaml --conv gcn")


def _check_checkpoint():
    """checkpoint.py::_atomic_write writes text into its own handle, which is
    correct. Warn only if it shows the same broken reopen-and-fsync pattern."""
    if not os.path.isfile(CKP):
        return
    c = read(CKP)
    if 'open(tmp, "rb")' in c and "os.fsync" in c:
        print("  WARNING: checkpoint.py may have the same reopen+fsync pattern; "
              "check _atomic_write.")
    else:
        print("  [ok] checkpoint.py::_atomic_write uses the correct pattern.")


def _verify_compile():
    py_compile.compile(PRE, doraise=True)
    print("  [ok] pretrain_gnn.py compiles.")


if __name__ == "__main__":
    main()
