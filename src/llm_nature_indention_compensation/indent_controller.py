from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .numerics import softmax_stable


PY_DEDENT_KEYWORDS = ("elif", "else", "except", "finally")


def strip_comment_and_rstrip(line: str) -> str:
    s = line.rstrip()
    if "#" in s:
        s = s.split("#", 1)[0].rstrip()
    return s


def ends_with_colon(line: str) -> bool:
    return strip_comment_and_rstrip(line).endswith(":")


def leading_spaces(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def infer_indent_delta_from_text(text: str, default: int = 4) -> int:
    deltas: List[int] = []
    prev = 0
    for raw in text.splitlines():
        if not raw.strip():
            continue
        ind = leading_spaces(raw)
        if ind > prev:
            deltas.append(ind - prev)
        prev = ind

    if not deltas:
        return default

    counts: Dict[int, int] = {}
    for d in deltas:
        counts[d] = counts.get(d, 0) + 1
    # mode; tie -> smallest
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


@dataclass
class IndentController:
    """Maintains indentation stack and minimal lexer-ish state.

    - indent_stack is the set of active indentation levels (Python rule: dedent must match prior level)
    - prev_line_ended_with_colon forces +delta indentation on the next logical line
    - paren_depth and in_triple_string disable indentation constraints inside continuation contexts
    - cur_line_buf accumulates decoded characters since last newline to detect ':' and continuation context
    """

    indent_stack: List[int]
    indent_delta: int
    prev_line_ended_with_colon: bool
    paren_depth: int
    in_triple_string: bool
    cur_line_buf: str

    @staticmethod
    def init(indent_delta: int = 4) -> "IndentController":
        return IndentController(
            indent_stack=[0],
            indent_delta=indent_delta,
            prev_line_ended_with_colon=False,
            paren_depth=0,
            in_triple_string=False,
            cur_line_buf="",
        )

    def allowed_indents(self, next_line_text_no_indent: str) -> List[int]:
        # continuation contexts: don't constrain
        if self.in_triple_string or self.paren_depth > 0:
            return [self.indent_stack[-1]]

        head = next_line_text_no_indent.lstrip()
        head_kw = head.split(None, 1)[0] if head else ""
        if head_kw in PY_DEDENT_KEYWORDS:
            if len(self.indent_stack) >= 2:
                return [self.indent_stack[-2]]
            return [0]

        cur = self.indent_stack[-1]
        if self.prev_line_ended_with_colon:
            return [cur + self.indent_delta]
        return list(self.indent_stack)

    def update_stack(self, chosen_indent: int) -> None:
        cur = self.indent_stack[-1]
        if chosen_indent > cur:
            self.indent_stack.append(chosen_indent)
            return
        if chosen_indent < cur:
            while self.indent_stack and self.indent_stack[-1] > chosen_indent:
                self.indent_stack.pop()
            if not self.indent_stack or self.indent_stack[-1] != chosen_indent:
                self.indent_stack.append(chosen_indent)

    def observe_emitted_text(self, emitted: str) -> None:
        # naive triple-quote parity toggle
        if '"""' in emitted or "'''" in emitted:
            count = emitted.count('"""') + emitted.count("'''")
            if count % 2 == 1:
                self.in_triple_string = not self.in_triple_string

        # paren depth update when not in triple string
        if not self.in_triple_string:
            for ch in emitted:
                if ch in "([{":
                    self.paren_depth += 1
                elif ch in ")]}":
                    self.paren_depth = max(0, self.paren_depth - 1)

        # line buffer update
        if "\n" in emitted:
            parts = emitted.split("\n")
            for seg in parts[:-1]:
                self.cur_line_buf += seg
                self.prev_line_ended_with_colon = ends_with_colon(self.cur_line_buf)
                self.cur_line_buf = ""
            self.cur_line_buf += parts[-1]
        else:
            self.cur_line_buf += emitted


@dataclass
class IndentTokenMap:
    """Maps indentation widths -> token id sequences that decode to exactly that many spaces."""

    indent_to_ids: Dict[int, List[int]]
    indent_options: List[int]
    newline_id: Optional[int]

    @staticmethod
    def build(tokenizer, indent_delta: int = 4, max_depth: int = 20) -> "IndentTokenMap":
        indent_to_ids: Dict[int, List[int]] = {}
        indent_options = [i * indent_delta for i in range(max_depth + 1)]
        for n in indent_options:
            indent_to_ids[n] = tokenizer.encode(" " * n, add_special_tokens=False)

        newline_id: Optional[int] = None
        try:
            t = tokenizer.encode("\n", add_special_tokens=False)
            if len(t) == 1:
                newline_id = int(t[0])
        except Exception:
            newline_id = None

        return IndentTokenMap(indent_to_ids=indent_to_ids, indent_options=indent_options, newline_id=newline_id)


def apply_indent_candidate_mask(scores: np.ndarray, allowed_mask: np.ndarray) -> np.ndarray:
    masked = scores.astype(np.float64).copy()
    masked[allowed_mask == 0.0] = -np.inf
    if np.all(np.isneginf(masked)):
        return scores
    return masked


def sample_indent_width_from_scores(
    indent_widths: Sequence[int],
    scores: np.ndarray,
    allowed_widths: Sequence[int],
    temperature: float,
    rng: np.random.Generator,
) -> int:
    allowed = set(int(x) for x in allowed_widths)
    allow_mask = np.array([1.0 if int(w) in allowed else 0.0 for w in indent_widths], dtype=np.float64)
    masked = apply_indent_candidate_mask(scores, allow_mask)
    probs = softmax_stable(masked / max(1e-8, float(temperature)))
    idx = int(rng.choice(len(indent_widths), p=probs))
    return int(indent_widths[idx])
