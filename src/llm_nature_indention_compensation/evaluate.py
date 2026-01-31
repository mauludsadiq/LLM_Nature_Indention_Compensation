from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .token_gen import GenConfig, generate_with_indent_constraints


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def evaluate(prompts: List[str], model_name: str, device: str, cfg: GenConfig) -> Dict[str, Any]:
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForCausalLM.from_pretrained(model_name)
    mdl.eval()
    mdl.to(device)

    indent_errors = 0
    syntax_errors = 0
    total_tokens = 0
    total_ms = 0.0

    for p in prompts:
        t0 = time.time()
        code, err = generate_with_indent_constraints(p, mdl, tok, cfg)
        total_ms += (time.time() - t0) * 1000.0
        total_tokens += len(code.split())

        try:
            compile(code, "<string>", "exec")
        except IndentationError:
            indent_errors += 1
        except SyntaxError:
            syntax_errors += 1

    n = len(prompts)
    return {
        "n": n,
        "indent_error_rate": indent_errors / n,
        "syntax_error_rate": syntax_errors / n,
        "avg_tokens": total_tokens / n,
        "avg_latency_ms": total_ms / n,
        "model": model_name,
        "device": device,
        "cfg": {
            "max_new_tokens": cfg.max_new_tokens,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "seed": cfg.rng_seed,
            "indent_delta": cfg.indent_delta,
            "max_depth": cfg.max_depth,
            "indent_score": cfg.indent_score,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="gpt2")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="out/eval.json")
    args = ap.parse_args()

    prompts = [
        "Write a Python function f() with nested if/for/with and return 0.\n",
        "Write a Python function g(x) that loops and uses if/else and returns x.\n",
        "Generate Python code with 4 nested blocks and 3 print lines.\n",
    ]

    cfg = GenConfig(
        model_name=args.model,
        max_new_tokens=256,
        temperature=0.8,
        top_p=0.95,
        rng_seed=args.seed,
        indent_delta=4,
        max_depth=20,
        indent_score="first",
    )

    metrics = evaluate(prompts, args.model, args.device, cfg)
    _write_json(Path(args.out), metrics)
    print("WROTE_EVAL=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
