# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
diff.py — compare two normalized responses for parity.

Two correct LLM answers are rarely byte-identical, so the default parity model
is not exact string equality. It combines:

  1. Structural parity  — finish_reason matches, and tool calls match by name
                          and (JSON-normalized) arguments.
  2. Text similarity    — a difflib ratio over normalized text, compared to a
                          threshold (default 0.60).

Set `exact=True` to require byte-identical text instead of the similarity score
(structural checks still apply).

Only stdlib is used, so this runs offline with no network or ML dependencies.
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from typing import Any

DEFAULT_THRESHOLD = 0.60

_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lowercase, collapse whitespace, strip — for similarity and exact compare."""
    return _WS.sub(" ", (text or "").strip().lower())


def text_similarity(a: str, b: str) -> float:
    """difflib ratio in [0.0, 1.0] over normalized text."""
    na, nb = normalize_text(a), normalize_text(b)
    if not na and not nb:
        return 1.0
    return difflib.SequenceMatcher(None, na, nb, autojunk=False).ratio()


def _canonical_tool_calls(tool_calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Canonicalize tool calls to (name, sorted-json-args) tuples for comparison."""
    out: list[tuple[str, str]] = []
    for tc in tool_calls or []:
        name = tc.get("name") or ""
        args = tc.get("arguments")
        try:
            args_canon = json.dumps(args, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            args_canon = str(args)
        out.append((name, args_canon))
    return out


@dataclass
class DiffResult:
    case_id: str
    passed: bool
    similarity: float
    threshold: float
    exact: bool
    reasons: list[str] = field(default_factory=list)
    source_text: str = ""
    target_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "passed": self.passed,
            "similarity": round(self.similarity, 4),
            "threshold": self.threshold,
            "exact": self.exact,
            "reasons": self.reasons,
        }


def compare(
    case_id: str,
    source: dict[str, Any],
    target: dict[str, Any],
    threshold: float = DEFAULT_THRESHOLD,
    exact: bool = False,
) -> DiffResult:
    """Compare two normalized responses (source = recording, target = replay)."""
    reasons: list[str] = []

    # --- Structural: finish reason ---
    if source.get("finish_reason") != target.get("finish_reason"):
        reasons.append(
            f"finish_reason differs: {source.get('finish_reason')!r} vs "
            f"{target.get('finish_reason')!r}"
        )

    # --- Structural: tool calls ---
    src_tools = _canonical_tool_calls(source.get("tool_calls") or [])
    tgt_tools = _canonical_tool_calls(target.get("tool_calls") or [])
    if src_tools != tgt_tools:
        reasons.append(f"tool_calls differ: {src_tools} vs {tgt_tools}")

    # --- Text parity ---
    src_text = source.get("text") or ""
    tgt_text = target.get("text") or ""
    similarity = text_similarity(src_text, tgt_text)

    if exact:
        if normalize_text(src_text) != normalize_text(tgt_text):
            reasons.append("exact text mismatch (normalized)")
    else:
        if similarity < threshold:
            reasons.append(
                f"text similarity {similarity:.3f} below threshold {threshold:.3f}"
            )

    return DiffResult(
        case_id=case_id,
        passed=not reasons,
        similarity=similarity,
        threshold=threshold,
        exact=exact,
        reasons=reasons,
        source_text=src_text,
        target_text=tgt_text,
    )
