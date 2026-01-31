from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .indent_controller import (
    IndentController,
    IndentTokenMap,
    infer_indent_delta_from_text,
    sample_indent_width_from_scores,
)
from .numerics import softmax_stable
from .repair import compile_and_repair


@dataclass
class GenConfig:
    model_name: str = "gpt2"
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.95
    rng_seed: int = 0
    indent_delta: int = 4
    max_depth: int = 20
    max_repair_iters: int = 6
    indent_score: str = "first"  # first | full


def sample_top_p(probs: np.ndarray, top_p: float, rng: np.random.Generator) -> int:
    idx = np.argsort(probs)[::-1]
    sp = probs[idx]
    cdf = np.cumsum(sp)
    cut = int(np.searchsorted(cdf, float(top_p), side="left")) + 1
    keep = idx[:cut]
    p = probs[keep]
    p = p / p.sum()
    return int(rng.choice(keep, p=p))


def _is_newline_token(tokenizer, tid: int, newline_id: Optional[int]) -> bool:
    if newline_id is not None:
        return int(tid) == int(newline_id)
    return tokenizer.decode([int(tid)], skip_special_tokens=False) == "\n"


def _score_indent_first_token(
    logits: np.ndarray,
    indent_map: IndentTokenMap,
    allowed_widths: List[int],
    indent_widths: List[int],
) -> np.ndarray:
    # Scores shape (K,) over indent_widths list
    scores = np.full((len(indent_widths),), -np.inf, dtype=np.float64)
    for i, w in enumerate(indent_widths):
        if w not in allowed_widths:
            continue
        seq = indent_map.indent_to_ids[w]
        if len(seq) == 0:
            scores[i] = 0.0
        else:
            scores[i] = float(logits[int(seq[0])])
    return scores


def _score_indent_full_seq(
    model,
    input_ids_prefix,
    logits_after_newline: np.ndarray,
    indent_map: IndentTokenMap,
    allowed_widths: List[int],
    indent_widths: List[int],
    temperature: float,
    device,
) -> np.ndarray:
    """
    FULL scoring without KV-cache stepping.
    We score each candidate indent token sequence by running forward passes on:
        prefix + seq[:k]
    This avoids DynamicCache / past_key_values entirely (prevents Bus error: 10).
    """
    import torch

    scores = np.full((len(indent_widths),), -np.inf, dtype=np.float64)

    # First token logprobs from provided logits_after_newline
    probs0 = softmax_stable((logits_after_newline.astype(np.float64)) / max(1e-8, float(temperature)))
    logp0 = np.log(np.clip(probs0, 1e-300, 1.0))

    prefix = input_ids_prefix  # shape (1, T), ends with newline
    allowed = set(int(x) for x in allowed_widths)

    for i, w in enumerate(indent_widths):
        if int(w) not in allowed:
            continue
        seq = indent_map.indent_to_ids[int(w)]
        if len(seq) == 0:
            scores[i] = 0.0
            continue

        lp = float(logp0[int(seq[0])])

        # Score remaining tokens by forward passes on prefix + consumed tokens
        cur = prefix
        for tid in seq[:-1]:
            cur = torch.cat([cur, torch.tensor([[int(tid)]], device=device, dtype=cur.dtype)], dim=1)
            with torch.no_grad():
                out = model(input_ids=cur, use_cache=False)
            lg = out.logits[0, -1].detach().float().cpu().numpy()
            probs = softmax_stable((lg.astype(np.float64)) / max(1e-8, float(temperature)))
            nxt = int(seq[seq.index(tid) + 1])
            lp += float(np.log(np.clip(probs[nxt], 1e-300, 1.0)))

        scores[i] = lp

    return scores
def generate_with_indent_constraints(prompt: str, model, tokenizer, cfg: GenConfig, on_event=None) -> Tuple[str, Optional[str]]:
    """
    Generation WITHOUT KV-cache stepping.
    Reason: transformers==5.0.0 DynamicCache single-step with past_key_values can crash (Bus error: 10)
    on some macOS CPU builds. We keep correctness + determinism by using full-prefix forwards (use_cache=False).
    """
    import torch

    rng = np.random.default_rng(int(cfg.rng_seed))
    indent_map = IndentTokenMap.build(tokenizer, indent_delta=int(cfg.indent_delta), max_depth=int(cfg.max_depth))

    controller = IndentController.init(indent_delta=int(cfg.indent_delta))
    controller.indent_delta = infer_indent_delta_from_text(prompt, default=int(cfg.indent_delta))

    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    decoded_prompt = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=False)
    controller.observe_emitted_text(decoded_prompt)
    decoded = ""
    forced: List[int] = []

    def next_logits(cur_ids):
        with torch.no_grad():
            out = model(input_ids=cur_ids, use_cache=False)
        return out.logits[0, -1].detach().float().cpu().numpy()

    for _ in range(int(cfg.max_new_tokens)):
        if on_event is not None:
            on_event({'type': 'progress', 'frac': float(_ + 1) / float(max(1, int(cfg.max_new_tokens)))})

        # If we have forced indent tokens, append them verbatim (no sampling)
        if forced:
            tid = int(forced.pop(0))
            input_ids = torch.cat([input_ids, torch.tensor([[tid]], device=device, dtype=input_ids.dtype)], dim=1)
            s = tokenizer.decode([tid], skip_special_tokens=False)
            controller.observe_emitted_text(s)
            decoded += s
            continue

        logits = next_logits(input_ids).astype(np.float64) / max(1e-8, float(cfg.temperature))
        probs = softmax_stable(logits)
        tid = sample_top_p(probs, float(cfg.top_p), rng)

        input_ids = torch.cat([input_ids, torch.tensor([[int(tid)]], device=device, dtype=input_ids.dtype)], dim=1)

        s = tokenizer.decode([int(tid)], skip_special_tokens=False)
        controller.observe_emitted_text(s)
        decoded += s

        # If we just emitted newline, choose + force indentation tokens
        if _is_newline_token(tokenizer, int(tid), indent_map.newline_id):
            nxt_logits = next_logits(input_ids)

            indent_widths = indent_map.indent_options
            allowed = controller.allowed_indents(next_line_text_no_indent="")
            allowed = [w for w in allowed if w in indent_widths]
            if not allowed:
                allowed = [controller.indent_stack[-1]]

            if cfg.indent_score == "full":
                scores = _score_indent_full_seq(
                    model=model,
                    input_ids_prefix=input_ids,
                    logits_after_newline=nxt_logits,
                    indent_map=indent_map,
                    allowed_widths=allowed,
                    indent_widths=indent_widths,
                    temperature=float(cfg.temperature),
                    device=device,
                )
            else:
                scores = _score_indent_first_token(
                    logits=nxt_logits,
                    indent_map=indent_map,
                    allowed_widths=allowed,
                    indent_widths=indent_widths,
                )

            all_scores = _score_indent_first_token(
                logits=nxt_logits,
                indent_map=indent_map,
                allowed_widths=indent_widths,
                indent_widths=indent_widths,
            )
            llm_pref_idx = int(np.argmax(all_scores)) if all_scores.size else 0
            llm_pref_width = int(indent_widths[llm_pref_idx]) if indent_widths else 0
            chosen = sample_indent_width_from_scores(
                indent_widths=indent_widths,
                scores=scores,
                allowed_widths=allowed,
                temperature=1.0,
                rng=rng,
            )
            depth = max(0, len(controller.indent_stack) - 1)
            if (llm_pref_width != int(chosen)) and (on_event is not None):
                on_event({
                    'type': 'compensation',
                    'depth': int(depth),
                    'llm_spaces': int(llm_pref_width),
                    'chosen_spaces': int(chosen),
                    'reason': 'matching scope rules',
                })

            controller.update_stack(int(chosen))
            forced = list(indent_map.indent_to_ids[int(chosen)])

    fixed, err = compile_and_repair(decoded, max_iters=int(cfg.max_repair_iters))
    return fixed, err
