from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .ui import HumanUI, RunHeader, RunSummary

from .hashing import sha256_text, sha256_canonical_json
from .token_gen import GenConfig, generate_with_indent_constraints


def _write_text(path: Path, s: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(s, encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--model", type=str, default="gpt2")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--indent-delta", type=int, default=4)
    ap.add_argument("--max-depth", type=int, default=20)
    ap.add_argument("--indent-score", type=str, default="first", choices=["first", "full"])
    ap.add_argument("--out", type=str, default="out/run")
    ap.add_argument("--ui", type=str, default="human", choices=["human", "json"])
    ap.add_argument("--prefetch", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModelForCausalLM.from_pretrained(args.model)
    mdl.eval()
    mdl.to(args.device)

    cfg = GenConfig(
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        rng_seed=args.seed,
        indent_delta=args.indent_delta,
        max_depth=args.max_depth,
        indent_score=args.indent_score,
    )

    code, compile_err = generate_with_indent_constraints(args.prompt, mdl, tok, cfg)

    _write_text(out / "generated.py", code)

    compile_status = {
        "compile_ok": compile_err is None,
        "compile_error": compile_err,
    }
    _write_json(out / "compile_status.json", compile_status)

    run_cfg = {
        "prompt": args.prompt,
        "model": args.model,
        "device": args.device,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "indent_delta": args.indent_delta,
        "max_depth": args.max_depth,
        "indent_score": args.indent_score,
    }
    _write_json(out / "config.json", run_cfg)

    # Witness bundle
    bundle = {
        "schema": "llm_nature_indention_compensation.bundle.v0.1",
        "config_digest": sha256_canonical_json(run_cfg),
        "generated_py_sha256": sha256_text(code),
        "compile_status_digest": sha256_canonical_json(compile_status),
    }
    bundle["bundle_digest"] = sha256_canonical_json(bundle)

    _write_json(out / "bundle.json", bundle)

    # PASS line for CI / grep-based verifiers
    pass_line = "PASS_INDENT_COMPILE=1" if compile_err is None else "PASS_INDENT_COMPILE=0"
    _write_text(out / "PASS.txt", pass_line + "\n")

    print(pass_line)
    print("bundle_digest:", bundle["bundle_digest"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
