from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Optional


def _isatty() -> bool:
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


def _c(s: str, code: str) -> str:
    if not _isatty():
        return s
    return f"\033[{code}m{s}\033[0m"


def green(s: str) -> str:
    return _c(s, "32")


def yellow(s: str) -> str:
    return _c(s, "33")


def red(s: str) -> str:
    return _c(s, "31")


def dim(s: str) -> str:
    return _c(s, "2")


def bold(s: str) -> str:
    return _c(s, "1")


def progress_bar(frac: float, width: int = 18) -> str:
    f = max(0.0, min(1.0, float(frac)))
    n = int(round(f * width))
    return "[" + ("●" * n) + ("·" * (width - n)) + "]"


@dataclass
class RunHeader:
    model: str
    target_depth: int
    indent_delta: int
    strategy: str


@dataclass
class RunSummary:
    compile_ok: bool
    max_depth_reached: int
    structural_integrity_pct: float
    bundle_digest: str
    out_dir: str
    elapsed_s: float
    compile_error: Optional[str] = None


class HumanUI:
    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled)
        self.t0 = time.time()

    def header(self, h: RunHeader) -> None:
        if not self.enabled:
            return
        print(bold("🚀 LLM Indentation Kernel: Active"))
        print(f"Model: {h.model} | Target Depth: {h.target_depth} | Strategy: {h.strategy} ({h.indent_delta})")
        print()

    def generating(self, frac: float, msg: str = "Generating") -> None:
        if not self.enabled:
            return
        bar = progress_bar(frac)
        pct = int(round(100.0 * frac))
        print(f"{bar} {pct:3d}% {msg}...")

    def compensation(self, depth: int, llm_spaces: int, chosen_spaces: int, reason: str) -> None:
        if not self.enabled:
            return
        print(yellow("🔧 Compensation Event:"))
        print(f"   Depth: {depth}")
        print(f"   LLM drifted to {llm_spaces} spaces.")
        print(f"   Kernel corrected to {chosen_spaces} spaces ({reason}).")
        print()

    def done(self, s: RunSummary) -> None:
        if not self.enabled:
            return
        elapsed = s.elapsed_s
        if s.compile_ok:
            print(green("✅ Success: 'generated.py' compiled successfully."))
        else:
            print(red("❌ Failure: 'generated.py' did not compile."))
        print(f"   Max Depth Reached: {s.max_depth_reached} levels.")
        print(f"   Structural Integrity: {s.structural_integrity_pct:.0f}%")
        print(f"   Bundle Digest: {s.bundle_digest}")
        print(f"   Out: {s.out_dir}")
        print(dim(f"   Elapsed: {elapsed:.2f}s"))
        if (not s.compile_ok) and s.compile_error:
            print()
            print(red("Compile error:"))
            print(s.compile_error.strip())
