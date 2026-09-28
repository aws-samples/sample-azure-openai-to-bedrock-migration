# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
normalize.py — map provider-specific chat responses into one comparable shape.

Azure OpenAI and the Bedrock OpenAI-compatible endpoint both return the OpenAI
Chat Completions shape. The Bedrock Converse API returns a different shape. This
module flattens all of them into a single `NormalizedResponse` dict so the diff
layer compares like for like.

Normalized shape:
    {
        "text":          str,          # the assistant's reply text
        "finish_reason": str,          # normalized stop reason (see _FINISH_MAP)
        "tool_calls":    list[dict],    # [{"name": str, "arguments": <json-decoded or str>}]
        "role":          str,          # usually "assistant"
        "usage":         {"input_tokens": int|None,
                          "output_tokens": int|None,
                          "total_tokens": int|None},
    }

Only stdlib is used here so the normalizer can run anywhere, including in the
offline test.
"""

from __future__ import annotations

import json
from typing import Any

# Map each provider's stop/finish vocabulary onto a common set.
_FINISH_MAP = {
    # OpenAI / OpenAI-compatible endpoint
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "content_filter",
    # Bedrock Converse
    "end_turn": "stop",
    "max_tokens": "length",
    "tool_use": "tool_use",
    "stop_sequence": "stop",
    "guardrail_intervened": "content_filter",
    "content_filtered": "content_filter",
}


def normalize_finish_reason(raw: str | None) -> str:
    if raw is None:
        return "unknown"
    return _FINISH_MAP.get(raw, raw)


def _empty_usage() -> dict[str, int | None]:
    return {"input_tokens": None, "output_tokens": None, "total_tokens": None}


def _maybe_json(value: Any) -> Any:
    """Tool-call arguments come back as a JSON string on OpenAI; decode if possible."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return value
    return value


def normalize_openai(response: dict[str, Any]) -> dict[str, Any]:
    """Normalize an OpenAI Chat Completions response (Azure or Bedrock endpoint).

    Accepts a plain dict. If you hold an SDK object, call `.model_dump()` /
    `.to_dict()` first (providers.py does this).
    """
    choices = response.get("choices") or [{}]
    choice = choices[0] or {}
    message = choice.get("message") or {}

    tool_calls: list[dict[str, Any]] = []
    for tc in message.get("tool_calls") or []:
        fn = (tc or {}).get("function") or {}
        tool_calls.append(
            {"name": fn.get("name"), "arguments": _maybe_json(fn.get("arguments"))}
        )

    usage_raw = response.get("usage") or {}
    usage = {
        "input_tokens": usage_raw.get("prompt_tokens"),
        "output_tokens": usage_raw.get("completion_tokens"),
        "total_tokens": usage_raw.get("total_tokens"),
    }

    return {
        "text": message.get("content") or "",
        "finish_reason": normalize_finish_reason(choice.get("finish_reason")),
        "tool_calls": tool_calls,
        "role": message.get("role") or "assistant",
        "usage": usage or _empty_usage(),
    }


def normalize_converse(response: dict[str, Any]) -> dict[str, Any]:
    """Normalize a Bedrock Converse response dict (boto3 `converse` output)."""
    output = response.get("output") or {}
    message = output.get("message") or {}
    content_blocks = message.get("content") or []

    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in content_blocks:
        if not isinstance(block, dict):
            continue
        if "text" in block:
            text_parts.append(block["text"])
        elif "toolUse" in block:
            tu = block["toolUse"] or {}
            tool_calls.append({"name": tu.get("name"), "arguments": tu.get("input")})

    usage_raw = response.get("usage") or {}
    usage = {
        "input_tokens": usage_raw.get("inputTokens"),
        "output_tokens": usage_raw.get("outputTokens"),
        "total_tokens": usage_raw.get("totalTokens"),
    }

    return {
        "text": "".join(text_parts),
        "finish_reason": normalize_finish_reason(response.get("stopReason")),
        "tool_calls": tool_calls,
        "role": message.get("role") or "assistant",
        "usage": usage or _empty_usage(),
    }


def normalize(kind: str, response: dict[str, Any]) -> dict[str, Any]:
    """Dispatch on provider kind: 'openai' | 'converse'."""
    if kind == "openai":
        return normalize_openai(response)
    if kind == "converse":
        return normalize_converse(response)
    raise ValueError(f"unknown response kind: {kind!r} (expected 'openai' or 'converse')")
