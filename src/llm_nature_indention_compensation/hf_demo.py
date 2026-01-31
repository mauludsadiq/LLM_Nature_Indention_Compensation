from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

from .hashing import sha256_bytes
from .hf_kernel import KernelConfig, UniversalIndentKernel
from .repair import compile_and_repair


def _write(path: Path, s: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(s, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="gpt2")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--max-new-tokens", type=int, default=192)
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

    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModelForCausalLM.from_pretrained(args.model).to(args.device).eval()

    prompt = args.prompt
    input_ids = tok.encode(prompt, return_tensors="pt").to(args.device)

    kernel = UniversalIndentKernel(
        tokenizer=tok,
        prompt_text=prompt,
        cfg=KernelConfig(indent_delta=int(args.indent_delta), max_depth=int(args.max_depth), rng_seed=int(args.seed)),
    )

    procs = LogitsProcessorList([kernel])

    g = torch.Generator(device=str(args.device))
    g.manual_seed(int(args.seed))

    out = mdl.generate(
        input_ids=input_ids,
        do_sample=True,
        use_cache=True,
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        logits_processor=procs,
        generator=g,
        pad_token_id=tok.eos_token_id,
    )

    text = tok.decode(out[0].tolist(), skip_special_tokens=False)
    code = text

    repaired, trace, ok, err = compile_and_repair(code, max_iters=6)

    _write(out_dir / "generated.py", repaired)
    _write(out_dir / "compile_status.json", f'{{\n  "compile_ok": {str(ok).lower()},\n  "compile_error": {("null" if err is None else repr(err))}\n}}\n')
    if trace:
        _write(out_dir / "repair_trace.txt", trace)

    digest = sha256_bytes(repaired.encode("utf-8"))
    _write(out_dir / "bundle_digest.txt", digest + "\n")

    print(f"HF_DEMO_OK=1 compile_ok={int(ok)} digest={digest}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
