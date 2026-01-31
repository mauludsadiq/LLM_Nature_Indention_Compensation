# LLM Nature Indention Compensation (GPT‑2 / VSC)

A **VSC-style**, proof-producing mini-repo that improves Python code generation reliability by eliminating **IndentationError** as a dominant failure mode in autoregressive models.

It does this with a **hybrid controller**:

1. **Newline-boundary indentation constraints** (token-level): when the model emits a newline, we force the next line's indentation to be one of a small, valid set derived from an **indent stack**.
2. **Compiler-feedback repair loop**: after generation, run `compile()`; if any residual indentation errors remain (tabs/spaces, mismatch), apply conservative whitespace-only repairs keyed off the `IndentationError` category.

The demo target is **GPT‑2** (via Hugging Face), but the controller is model-agnostic and can be reused for any causal LM that exposes logits step-by-step.

---

## Repo layout

```
LLM_Nature_Indention_Compensation/
  pyproject.toml
  README.md
  LICENSE
  scripts/
    run_demo.sh
  src/llm_nature_indention_compensation/
    numerics.py                  # logsumexp + stable softmax
    hashing.py                   # sha256 + canonical JSON hashing
    indent_controller.py         # indent stack + newline constraints + token maps
    repair.py                    # IndentationError classifier + targeted repair
    token_gen.py                 # GPT-2 token loop with forced indent sequences
    run.py                       # CLI: generate + write VSC witnesses (bundle.json)
    verify.py                    # CLI: verify bundle hashes + PASS line
    evaluate.py                  # small evaluation harness (indent/syntax error rate)
  tests/
    test_numerics.py
    test_repair.py
    test_controller.py
  out/
    (generated artifacts go here)
```

---

## Core mechanism (precise)

### A) Indent stack

Maintain a stack of active indentation levels:

- Start: `[0]`
- **Indent**: if chosen indentation `> current`, push it
- **Dedent**: if chosen indentation `< current`, pop until matching a prior level

This enforces Python's rule:

> *“unindent does not match any outer indentation level”* is prevented by construction.

### B) Newline-boundary constraint

The dominant indentation failures happen exactly here:

**newline → leading whitespace → first token of next line**

So we intervene only at newline boundaries:

- If previous logical line ended with `:` → force exactly `+Δ` indentation
- Otherwise → allow only `current indent` or a dedent to any prior indent level

This yields a hard constraint:

	P(wrong_indent)=0

via masking (conceptually):

	`logits[forbidden] = -inf` before softmax.

In this repo, we implement the practical equivalent by **forcing the exact token sequence** that decodes to `" " * chosen_indent` immediately after newline.

### C) Compiler-guided repair

After generation, run:

- `compile(code, "<string>", "exec")`

If it fails with `IndentationError`, classify the error:

- `expected an indented block`
- `unexpected indent`
- `unindent does not match any outer indentation level`
- `inconsistent use of tabs and spaces`

Then apply a **minimal repair** that modifies only leading whitespace (no rewriting of code semantics).

---

## Setup (clone OR zip)

You can run this repo either way:

### Option 1: Clone

```
git clone <YOUR_URL_HERE> LLM_Nature_Indention_Compensation
cd LLM_Nature_Indention_Compensation
```

### Option 2: Zip

If you received `LLM_Nature_Indention_Compensation.zip`:

```
unzip LLM_Nature_Indention_Compensation.zip
cd LLM_Nature_Indention_Compensation
```

---

## Local environment setup (macOS/Linux)

```
python -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

Optional dev deps:

```
python -m pip install -e ".[dev]"
pytest
```

---

## Run: single witnessed generation

The `run` command generates Python code using GPT‑2 + indentation constraints, writes witnesses to `out/`, and prints a PASS line.

```
python -m llm_nature_indention_compensation.run       --prompt "Write a Python function stress_test() with nested if/for/with and return 0."       --model gpt2       --device cpu       --max-new-tokens 256       --temperature 0.8       --top-p 0.95       --seed 0       --indent-delta 4       --max-depth 20       --indent-score first       --out out/run_demo
```

Outputs:

- `out/run_demo/generated.py`        generated code
- `out/run_demo/compile_status.json` compile OK + error string if any
- `out/run_demo/config.json`         run configuration
- `out/run_demo/bundle.json`         hash-anchored witness bundle
- `out/run_demo/PASS.txt`            `PASS_INDENT_COMPILE=1|0`

---

## Verify: VSC-style replay check

Verifies that the emitted hashes match the stored artifacts:

```
python -m llm_nature_indention_compensation.verify out/run_demo
```

This prints:

- `VERIFY_OK=1` or `VERIFY_OK=0`

---

## Evaluation harness

A small harness that measures indentation and syntax error rates over a tiny prompt set:

```
python -m llm_nature_indention_compensation.evaluate       --model gpt2       --device cpu       --seed 0       --out out/eval.json
```

---

## Production-ready upgrade path (exact)

1. **Integrate as a decoding hook**
   - Transformers: custom generation loop OR logits processor + forced-token queue
   - vLLM: decode-step hook where newline triggers a forced indentation sequence

2. **Indent scoring**
   - `--indent-score first` is cheap: scores candidates by first indent token
   - `--indent-score full` scores full indent sequences by logprob (slower, more correct)

3. **Dedent keywords**
   - Add a 1–2 token lookahead after newline (beyond spaces) to detect `elif/else/except/finally`
   - If detected, override the indent choice to match the header level

4. **Continuation contexts**
   - Replace the heuristic `(paren_depth, in_triple_string)` with a tighter incremental tokenizer
   - Or accept heuristics (usually sufficient for most codegen prompts)

5. **Keep compile-and-repair**
   - It catches the tails: tabs/spaces mix, odd multi-line strings, rare edge cases

---

## Notes on determinism

- The decode loop uses a fixed RNG seed (`--seed`).
- All witness hashes are computed from **canonical JSON** (sorted keys, compact separators).
- `bundle.json` contains the digests and a `bundle_digest` that commits the bundle contents itself.

---

## One-liner demo

```
. .venv/bin/activate
sh scripts/run_demo.sh
```

Universal Indentation Kernel (Logit Middleware)
===============================================

This repo now supports two integration modes:

1) Token-loop generator (repo-native)
------------------------------------

Uses a custom token loop and an indent stack to intervene only at newline boundaries, plus a compiler-guided repair loop.

Known-good run:

  export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  rm -rf out/run_demo
  python -m llm_nature_indention_compensation.run --prompt $'def stress_test():\n    """Nested if/for/with. Must return 0."""\n    ' --model gpt2 --device cpu --max-new-tokens 192 --temperature 0.2 --top-p 0.95 --seed 0 --indent-delta 4 --max-depth 20 --indent-score full --out out/run_demo
  python -m llm_nature_indention_compensation.verify out/run_demo
  cat out/run_demo/PASS.txt
  cat out/run_demo/compile_status.json

2) Hugging Face LogitsProcessor (universal middleware)
------------------------------------------------------

UniversalIndentKernel is a model-agnostic logit interceptor that enforces indentation tokens at newline boundaries when representable by the tokenizer.

Run:

  rm -rf out/hf_demo
  python -m llm_nature_indention_compensation.hf_demo --model gpt2 --device cpu --max-new-tokens 160 --temperature 0.2 --top-p 0.95 --seed 0 --indent-delta 4 --max-depth 20 --out out/hf_demo --prompt $'def stress_test():\n    """Nested if/for/with. Must return 0."""\n    '

Human-centric output
====================

The default CLI now prints a progressive disclosure story in a TTY:

- progress bar
- compensation events (LLM drift vs kernel correction)
- success/failure summary

Use --ui json for machine logs.
