from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from transformers import LogitsProcessor

from .kernel_core import KernelConfig, KernelState


@dataclass
class KernelAdapterConfig:
    indent_delta: int = 4
    max_depth: int = 20
    rng_seed: int = 0
    temperature: float = 1.0


class UniversalIndentKernel(LogitsProcessor):
    def __init__(self, tokenizer, prompt_text: str = "", cfg: Optional[KernelConfig] = None, on_event=None, temperature: float = 1.0):
        super().__init__()
        self.tokenizer = tokenizer
        self.state = KernelState(tokenizer=tokenizer, prompt_text=prompt_text, cfg=cfg, on_event=on_event)
        self.temperature = float(temperature)
        self._last_len: Optional[int] = None

    def _mask_to_only(self, scores: torch.FloatTensor, allowed_ids: List[int]) -> torch.FloatTensor:
        if not allowed_ids:
            return scores
        mask = torch.full_like(scores, float("-inf"))
        mask[:, allowed_ids] = 0.0
        return scores + mask

    def _observe_new_ids(self, input_ids_1d: torch.Tensor) -> None:
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
        self.state.observe_text(s)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.ndim != 2 or scores.ndim != 2:
            return scores
        if input_ids.shape[0] != 1:
            return scores

        self._observe_new_ids(input_ids[0])

        if self.state.forced_ids:
            want = int(self.state.forced_ids.pop(0))
            return self._mask_to_only(scores, [want])

        last_tid = int(input_ids[0, -1].item())
        if not self.state.is_newline_token_id(last_tid):
            return scores

        logits = scores[0].detach().float().cpu().numpy().astype(np.float64)
        tid0 = self.state.plan_indent_from_logits_first_token(logits_after_newline=logits, temperature=self.temperature)
        if tid0 is None:
            return scores
        return self._mask_to_only(scores, [int(tid0)])
