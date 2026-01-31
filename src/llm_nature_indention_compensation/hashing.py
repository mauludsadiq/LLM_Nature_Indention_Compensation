from __future__ import annotations

import hashlib
import json
from typing import Any


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_text(s: str) -> str:
    return sha256_bytes(s.encode("utf-8"))


def canonical_json_bytes(obj: Any) -> bytes:
    """Deterministic JSON encoding (sorted keys, no trailing whitespace)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_canonical_json(obj: Any) -> str:
    return sha256_bytes(canonical_json_bytes(obj))
