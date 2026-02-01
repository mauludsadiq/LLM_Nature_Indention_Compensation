from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from .indent_controller import IndentController, IndentTokenMap, infer_indent_delta_from_text, sample_indent_width_from_scores
from .numerics import softmax_stable


EventSink = Optional[Callable[[Dict], None]]


@dataclass
class KernelConfig:
    indent_delta: int = 4
    max_depth: int = 20
    mode: str = "first"
    rng_seed: int = 0


class KernelState:
    def __init__(self, tokenizer, prompt_text: str = "", cfg: Optional[KernelConfig] = None, on_event: EventSink = None):
        self.tokenizer = tokenizer
        self.cfg = cfg or KernelConfig()
        self.on_event = on_event
        self.rng = np.random.default_rng(int(self.cfg.rng_seed))

        self.indent_map = IndentTokenMap.build(
            tokenizer,
            indent_delta=int(self.cfg.indent_delta),
            max_depth=int(self.cfg.max_depth),
        )

        self.controller = IndentController.init(indent_delta=int(self.cfg.indent_delta))
        self.controller.indent_delta = infer_indent_delta_from_text(prompt_text, default=int(self.cfg.indent_delta))

        self.forced_ids: List[int] = []
        self._progress_last_bucket: Optional[int] = None

        if prompt_text:
            self.controller.observe_emitted_text(prompt_text)

    def emit(self, ev: Dict) -> None:
        if self.on_event is not None:
            self.on_event(ev)

    def progress(self, step: int, total: int) -> None:
        if self.on_event is None:
            return
        t = max(1, int(total))
        s = max(0, min(int(step), t))
        frac = float(s) / float(t)
        bucket = int(frac * 10.0)
        if self._progress_last_bucket is None or bucket != self._progress_last_bucket:
            self._progress_last_bucket = bucket
            self.emit({"type": "progress", "frac": frac})

    def observe_text(self, text_delta: str) -> None:
        if text_delta:
            self.controller.observe_emitted_text(text_delta)

    def is_newline_token_id(self, tid: int) -> bool:
        if self.indent_map.newline_id is not None:
            return int(tid) == int(self.indent_map.newline_id)
        s = self.tokenizer.decode([int(tid)], skip_special_tokens=False)
        return s == "\n"

    def indent_widths(self) -> List[int]:
        return list(self.indent_map.indent_options)

    def allowed_widths(self) -> List[int]:
        widths = self.indent_widths()
        allowed = self.controller.allowed_indents(next_line_text_no_indent="")
        allowed = [int(w) for w in allowed if int(w) in widths]
        if not allowed:
            if self.controller.indent_stack:
                return [int(self.controller.indent_stack[-1])]
            return []
        return allowed

    def llm_pref_width_from_logits_first_token(self, logits_after_newline: np.ndarray) -> int:
        widths = self.indent_widths()
        if not widths:
            return 0
        scores = np.full((len(widths),), -np.inf, dtype=np.float64)
        for i, w in enumerate(widths):
            seq = self.indent_map.indent_to_ids[int(w)]
            if len(seq) == 0:
                scores[i] = 0.0
            else:
                scores[i] = float(logits_after_newline[int(seq[0])])
        j = int(np.argmax(scores)) if scores.size else 0
        return int(widths[j])

    def choose_width_from_scores(self, widths: List[int], allowed: List[int], scores: np.ndarray) -> int:
        if not widths:
            return 0
        return int(
            sample_indent_width_from_scores(
                indent_widths=widths,
                scores=scores,
                rng=self.rng,
            )
        )

    def set_forced_from_width(self, width: int) -> None:
        seq = list(self.indent_map.indent_to_ids[int(width)])
        self.forced_ids = [int(x) for x in seq]

    def maybe_emit_compensation(self, depth: int, llm_width: int, chosen_width: int, reason: str) -> None:
        if self.on_event is None:
            return
        if int(llm_width) == int(chosen_width):
            return
        self.emit(
            {
                "type": "compensation",
                "depth": int(depth),
                "llm_spaces": int(llm_width),
                "chosen_spaces": int(chosen_width),
                "reason": str(reason),
            }
        )

    def plan_indent_from_logits_first_token(self, logits_after_newline: np.ndarray, temperature: float = 1.0) -> Optional[int]:
        allowed = self.allowed_widths()
        widths = self.indent_widths()
        if not allowed or not widths:
            return None

        logits = logits_after_newline.astype(np.float64)
        probs = softmax_stable(logits / max(1e-8, float(temperature)))

        best_w: Optional[int] = None
        best_p = -1.0
        for w in allowed:
            seq = self.indent_map.indent_to_ids[int(w)]
            if not seq:
                continue
            tid0 = int(seq[0])
            p = float(probs[tid0])
            if p > best_p:
                best_p = p
                best_w = int(w)

        if best_w is None:
            return None

        llm_pref = self.llm_pref_width_from_logits_first_token(logits_after_newline)
        depth = max(0, len(self.controller.indent_stack) - 1)
        self.maybe_emit_compensation(depth=depth, llm_width=llm_pref, chosen_width=best_w, reason="matching scope rules")

        seq = list(self.indent_map.indent_to_ids[int(best_w)])
        self.forced_ids = [int(x) for x in seq]
        if not self.forced_ids:
            return None
        return int(self.forced_ids.pop(0))
