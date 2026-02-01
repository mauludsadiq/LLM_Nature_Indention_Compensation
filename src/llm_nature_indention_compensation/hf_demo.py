from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .hashing import sha256_bytes
from .hf_kernel import KernelConfig, UniversalIndentKernel
from .numerics import softmax_stable
from .repair import compile_and_repair


def _write(path: Path, s: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(s, encoding="utf-8")


def _sample_top_p(probs: np.ndarray, top_p: float, rng: np.random.Generator) -> int:
    idx = np.argsort(probs)[::-1]
    sp = probs[idx]
    cdf = np.cumsum(sp)
    cut = int(np.searchsorted(cdf, float(top_p), side="left")) + 1
    keep = idx[:cut]
    p = probs[keep]
    p = p / p.sum()
    return int(rng.choice(keep, p=p))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="gpt2")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--indent-delta", type=int, default=4)
    ap.add_argument("--max-depth", type=int, default=20)
    ap.add_argument("--out", type=str, default="out/hf_demo")
    ap.add_argument("--prompt", type=str, required=True)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(args.model).to(args.device).eval()

    prompt = args.prompt
    input_ids = tok.encode(prompt, return_tensors="pt").to(args.device)

    kernel = UniversalIndentKernel(
        tokenizer=tok,
        prompt_text=prompt,
        cfg=KernelConfig(
            indent_delta=int(args.indent_delta),
            max_depth=int(args.max_depth),
            rng_seed=int(args.seed),
        ),
    )

    decoded = ""
    for _ in range(int(args.max_new_tokens)):
        with torch.no_grad():
            out = mdl(input_ids=input_ids, use_cache=False)
        scores = out.logits[:, -1, :].detach()

        scores = kernel(input_ids, scores)

        logits = scores[0].float().cpu().numpy().astype(np.float64)
        logits = logits / max(1e-8, float(args.temperature))
        probs = softmax_stable(logits)

        tid = _sample_top_p(probs, float(args.top_p), rng)

        input_ids = torch.cat(
            [input_ids, torch.tensor([[int(tid)]], device=args.device, dtype=input_ids.dtype)],
            dim=1,
        )

        s = tok.decode([int(tid)], skip_special_tokens=False)
        decoded += s

    code = prompt + decoded
    fixed, err = compile_and_repair(code, max_iters=6)
    ok = err is None

    _write(out_dir / "generated.py", fixed)
    _write(out_dir / "compile_status.json", f'{{\n  "compile_ok": {str(ok).lower()},\n  "compile_error": {("null" if err is None else repr(err))}\n}}\n')

    digest = sha256_bytes(fixed.encode("utf-8"))
    _write(out_dir / "bundle_digest.txt", digest + "\n")

    print(f"HF_DEMO_OK=1 compile_ok={int(ok)} digest={digest}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
