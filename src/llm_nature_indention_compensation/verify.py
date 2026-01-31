from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .hashing import sha256_text, sha256_canonical_json


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=str)
    args = ap.parse_args()

    d = Path(args.run_dir)
    cfg = _read_json(d / "config.json")
    status = _read_json(d / "compile_status.json")
    bundle = _read_json(d / "bundle.json")
    gen = (d / "generated.py").read_text(encoding="utf-8")

    ok = True
    if bundle.get("config_digest") != sha256_canonical_json(cfg):
        print("FAIL: config_digest mismatch")
        ok = False
    if bundle.get("compile_status_digest") != sha256_canonical_json(status):
        print("FAIL: compile_status_digest mismatch")
        ok = False
    if bundle.get("generated_py_sha256") != sha256_text(gen):
        print("FAIL: generated_py_sha256 mismatch")
        ok = False

    # recompute bundle digest with stored fields
    recompute = dict(bundle)
    recompute.pop("bundle_digest", None)
    want = sha256_canonical_json(recompute)
    if bundle.get("bundle_digest") != want:
        print("FAIL: bundle_digest mismatch")
        ok = False

    pass_txt = (d / "PASS.txt").read_text(encoding="utf-8").strip()
    expected = "PASS_INDENT_COMPILE=1" if status.get("compile_ok") else "PASS_INDENT_COMPILE=0"
    if pass_txt != expected:
        print("FAIL: PASS.txt mismatch")
        ok = False

    print("VERIFY_OK=1" if ok else "VERIFY_OK=0")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
