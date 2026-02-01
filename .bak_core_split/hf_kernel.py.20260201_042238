from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import LogitsProcessor

from .indent_controller import IndentController, IndentTokenMap, infer_indent_delta_from_text
from .numerics import softmax_stable


@dataclass
class KernelConfig:
    indent_delta: int = 4
    max_depth: int = 20
    mode: str = "first"
    rng_seed: int = 0


class UniversalIndentKernel(LogitsProcessor):
    def __init__(self, tokenizer, prompt_text: str = "", cfg: Optional[KernelConfig] = None):
        super().__init__()
        self.tokenizer = tokenizer
        self.cfg = cfg or KernelConfig()
        self.rng = np.random.default_rng(int(self.cfg.rng_seed))

        self.indent_map = IndentTokenMap.build(
            tokenizer,
            indent_delta=int(self.cfg.indent_delta),
            max_depth=int(self.cfg.max_depth),
        )

        self.controller = IndentController.init(indent_delta=int(self.cfg.indent_delta))
        self.controller.indent_delta = infer_indent_delta_from_text(prompt_text, default=int(self.cfg.indent_delta))

        self._last_len: Optional[int] = None
        self._forced: List[int] = []

        if prompt_text:
            self.controller.observe_emitted_text(prompt_text)

    def _is_newline(self, tid: int) -> bool:
        if self.indent_map.newline_id is not None:
            return int(tid) == int(self.indent_map.newline_id)
        s = self.tokenizer.decode([int(tid)], skip_special_tokens=False)
        return s == "\n"

    def _update_controller_with_new_tokens(self, input_ids_1d: torch.Tensor) -> None:
        ids = input_ids_1d.detach().cpu().tolist()
        n = len(ids)
        if self._last_len is None:
            self._last_len = n
            return
        if n <= self._last_len:
            return
        new_ids = ids[self._last_len :]
        self._last_len = n
        if not new_ids:
            return
        s = self.tokenizer.decode([int(t) for t in new_ids], skip_special_tokens=False)
        self.controller.observe_emitted_text(s)

    def _mask_to_only(self, scores: torch.FloatTensor, allowed_ids: List[int]) -> torch.FloatTensor:
        if not allowed_ids:
            return scores
        mask = torch.full_like(scores, float("-inf"))
        mask[:, allowed_ids] = 0.0
        return scores + mask

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.ndim != 2:
            return scores
        if scores.ndim != 2:
            return scores
        if input_ids.shape[0] != 1:
            return scores

        self._update_controller_with_new_tokens(input_ids[0])

        if self._forced:
            want = int(self._forced.pop(0))
            return self._mask_to_only(scores, [want])

        last_tid = int(input_ids[0, -1].item())
        if not self._is_newline(last_tid):
            return scores

        allowed = self.controller.allowed_indents(next_line_text_no_indent="")
        indent_widths = self.indent_map.indent_options
        allowed = [w for w in allowed if w in indent_widths]
        if not allowed:
            return scores

        candidate_first_ids: Dict[int, int] = {}
        for w in allowed:
            seq = self.indent_map.indent_to_ids[int(w)]
            if not seq:
                continue
            candidate_first_ids[int(w)] = int(seq[0])

        if not candidate_first_ids:
            return scores

        logits = scores[0].detach().float().cpu().numpy().astype(np.float64)
        probs = softmax_stable(logits)
        best_w = None
        best_p = -1.0
        for w, tid in candidate_first_ids.items():
            p = float(probs[int(tid)])
            if p > best_p:
                best_p = p
                best_w = int(w)

        if best_w is None:
            return scores

        seq = self.indent_map.indent_to_ids[int(best_w)]
        if len(seq) >= 2:
            self._forced = [int(t) for t in seq[1:]]

        first = int(seq[0])
        return self._mask_to_only(scores, [first])
