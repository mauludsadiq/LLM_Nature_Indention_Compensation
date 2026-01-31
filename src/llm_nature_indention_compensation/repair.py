from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .indent_controller import ends_with_colon, infer_indent_delta_from_text, leading_spaces


@dataclass
class IndentErrorInfo:
    kind: str
    lineno: Optional[int]
    msg: str


def classify_indent_error(e: IndentationError) -> IndentErrorInfo:
    msg = getattr(e, "msg", str(e)) or str(e)
    lineno = getattr(e, "lineno", None)

    if "expected an indented block" in msg:
        return IndentErrorInfo("expected_indent", lineno, msg)
    if "unexpected indent" in msg:
        return IndentErrorInfo("unexpected_indent", lineno, msg)
    if "unindent does not match any outer indentation level" in msg:
        return IndentErrorInfo("unindent_mismatch", lineno, msg)
    if "inconsistent use of tabs and spaces" in msg:
        return IndentErrorInfo("tabs_spaces", lineno, msg)
    return IndentErrorInfo("unknown", lineno, msg)


def valid_indent_levels(prefix: str) -> List[int]:
    levels = {0}
    for ln in prefix.splitlines():
        if ln.strip():
            levels.add(leading_spaces(ln))
    return sorted(levels)


def repair_indentation(code: str, info: IndentErrorInfo) -> str:
    lines = code.splitlines()
    delta = infer_indent_delta_from_text(code, default=4)

    def set_indent(i0: int, new_indent: int) -> None:
        if 0 <= i0 < len(lines):
            lines[i0] = (" " * max(0, new_indent)) + lines[i0].lstrip(" ")

    if info.kind == "tabs_spaces":
        fixed: List[str] = []
        for ln in lines:
            m = re.match(r"^\t+", ln)
            if m:
                t = len(m.group(0))
                fixed.append((" " * (4 * t)) + ln[t:])
            else:
                fixed.append(ln)
        return "\n".join(fixed) + ("\n" if code.endswith("\n") else "")

    if info.lineno is None:
        lv = valid_indent_levels(code)
        fixed: List[str] = []
        for ln in lines:
            if not ln.strip():
                fixed.append(ln)
                continue
            ind = leading_spaces(ln)
            if ind in lv:
                fixed.append(ln)
                continue
            closest = min(lv, key=lambda x: abs(x - ind))
            fixed.append((" " * closest) + ln.lstrip(" "))
        return "\n".join(fixed) + ("\n" if code.endswith("\n") else "")

    i = int(info.lineno) - 1

    if info.kind == "expected_indent":
        hdr = i - 1
        while hdr >= 0 and not ends_with_colon(lines[hdr]):
            hdr -= 1
        if hdr >= 0:
            set_indent(i, leading_spaces(lines[hdr]) + delta)
        return "\n".join(lines) + ("\n" if code.endswith("\n") else "")

    if info.kind == "unexpected_indent":
        prev = i - 1
        while prev >= 0 and not lines[prev].strip():
            prev -= 1
        if prev >= 0:
            prev_ind = leading_spaces(lines[prev])
            if ends_with_colon(lines[prev]):
                set_indent(i, prev_ind + delta)
            else:
                set_indent(i, prev_ind)
        return "\n".join(lines) + ("\n" if code.endswith("\n") else "")

    if info.kind == "unindent_mismatch":
        lv = valid_indent_levels("\n".join(lines[:i]))
        if lv and 0 <= i < len(lines):
            curr = leading_spaces(lines[i])
            closest = min(lv, key=lambda x: abs(x - curr))
            set_indent(i, closest)
        return "\n".join(lines) + ("\n" if code.endswith("\n") else "")

    lv = valid_indent_levels("\n".join(lines[:i]))
    if lv and 0 <= i < len(lines):
        curr = leading_spaces(lines[i])
        closest = min(lv, key=lambda x: abs(x - curr))
        set_indent(i, closest)
    return "\n".join(lines) + ("\n" if code.endswith("\n") else "")


def compile_and_repair(code: str, max_iters: int = 6) -> Tuple[str, Optional[str]]:
    cur = code
    for _ in range(int(max_iters)):
        try:
            compile(cur, "<string>", "exec")
            return cur, None
        except IndentationError as e:
            cur = repair_indentation(cur, classify_indent_error(e))
        except SyntaxError as e:
            return cur, f"SyntaxError: {e}"
    try:
        compile(cur, "<string>", "exec")
        return cur, None
    except Exception as e:
        return cur, f"{type(e).__name__}: {e}"
