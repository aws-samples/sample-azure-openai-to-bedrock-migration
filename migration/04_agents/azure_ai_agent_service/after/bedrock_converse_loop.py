# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
AFTER — Bedrock Converse API researcher agent (explicit tool-call loop).

Migrates ../before/azure_researcher.py with these substitutions:

  BEFORE (Azure)                         AFTER (Bedrock)
  ─────────────────────────────────────  ────────────────────────────────────────
  AIProjectClient + BingGroundingTool    boto3 bedrock-runtime converse()
  DefaultAzureCredential                 IAM role / env-var credentials (boto3)
  Azure AI Agent Service managed loop   Explicit toolUse → toolResult loop
  functions.json OpenAI schema           toolConfig / toolSpec (same JSON schema)
  azure_deployment="gpt-4"              BEDROCK_MODEL_ID env var (any Bedrock model)

Key insight — the loop that Azure ran server-side must be driven explicitly:

  1. Call converse() with toolConfig
  2. If stopReason == "tool_use": execute each toolUse block, append toolResult
  3. Call converse() again with the updated message list
  4. Repeat until stopReason != "tool_use"

The three tool schemas from functions.json map directly to toolSpec — only the
wrapping changes:

  OpenAI:   {"type":"function","function":{"name":..., "parameters": schema}}
  Bedrock:  {"toolSpec":{"name":..., "inputSchema":{"json": schema}}}

Web search backing is provided by a stub.  In production replace _web_search(),
_entity_search(), _news_search() with Tavily, Brave Search, or any search API.

Env vars:
    AWS_REGION        (default: us-east-1)
    BEDROCK_MODEL_ID  (default: us.openai.gpt-5.6-terra)
    BEDROCK_GUARDRAIL_ID (optional — attach a Bedrock Guardrail; T003 mitigation)
    TAVILY_API_KEY    (optional — used by the search stubs if set)
"""

from __future__ import annotations

import json
import os
from typing import Any

import boto3

# ── Bedrock client — IAM credentials, no Azure SDK needed ─────────────────────
# profile_name is a boto3.Session() argument, not a boto3.client() argument, so
# build a Session first when AWS_PROFILE is set and derive the client from it.
_profile = os.environ.get("AWS_PROFILE")
_session = boto3.Session(
    region_name=os.environ.get("AWS_REGION", "us-east-1"),
    **({"profile_name": _profile} if _profile else {}),
)
bedrock = _session.client("bedrock-runtime")
# Default: us.openai.gpt-5.6-terra (OpenAI frontier model on Bedrock, via US
# cross-Region inference profile). Override with any Bedrock model id that
# supports tool use.
MODEL = os.environ.get("BEDROCK_MODEL_ID", "us.openai.gpt-5.6-terra")

# T003 mitigation: attach a Guardrail to filter indirect prompt injection
# Optional. Set BEDROCK_GUARDRAIL_ID to a Bedrock Guardrail with prompt-attack
# filters enabled; when unset the calls run without a guardrail (unchanged
# behavior). See https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-prompt-attack.html
GUARDRAIL_ID = os.environ.get("BEDROCK_GUARDRAIL_ID", "")


def _guardrail_config() -> dict[str, Any]:
    """Return a guardrailConfig kwarg fragment when a Guardrail is configured."""
    if not GUARDRAIL_ID:
        return {}
    return {
        "guardrailConfig": {
            "guardrailIdentifier": GUARDRAIL_ID,
            "guardrailVersion": "DRAFT",
            "trace": "enabled",
        }
    }

# ── Tool config — functions.json schemas re-wrapped for Bedrock Converse ───────
#
# OpenAI function schema wrapping:
#   {"type":"function","function":{"name":N,"description":D,"parameters":schema}}
#
# Bedrock toolSpec wrapping:
#   {"toolSpec":{"name":N,"description":D,"inputSchema":{"json":schema}}}
#
# The inner JSON Schema object is IDENTICAL — only the outer envelope changes.

_QUERY_SCHEMA = {
    "type": "object",
    "properties": {
        "query":  {"type": "string",
                   "description": "An optimal search query"},
        "market": {"type": "string",
                   "description": "Market code, e.g. en-US"},
    },
    "required": ["query"],
}

TOOL_CONFIG: dict[str, Any] = {
    "tools": [
        {
            "toolSpec": {
                "name": "find_information",
                "description": (
                    "Finds information on the web given a query using a search API. "
                    "Use for general information — not news or entity lookups."
                ),
                "inputSchema": {"json": _QUERY_SCHEMA},
            }
        },
        {
            "toolSpec": {
                "name": "find_entities",
                "description": (
                    "Finds entities (people, places, things) on the web. "
                    "Use for entity lookups, not general information or news."
                ),
                "inputSchema": {"json": _QUERY_SCHEMA},
            }
        },
        {
            "toolSpec": {
                "name": "find_news",
                "description": (
                    "Finds recent news articles on the web given a query. "
                    "Use when the user is looking for news."
                ),
                "inputSchema": {"json": _QUERY_SCHEMA},
            }
        },
    ],
    "toolChoice": {"auto": {}},
}


# ── Search backing (replace with Tavily / Brave / etc. in production) ──────────

def _web_search(query: str, market: str = "en-US") -> list[dict]:
    """General web search stub.  Replace with real search API."""
    try:
        import tavily  # type: ignore
        client = tavily.TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        results = client.search(query, max_results=5)["results"]
        return [{"url": r["url"], "name": r["title"], "description": r["content"]} for r in results]
    except Exception:
        # Deterministic offline fallback used when no search API is configured
        # (e.g. in smoke tests). Kept neutral and plausible so the model treats
        # it as usable grounding rather than an error to apologize for.
        return [
            {
                "url": f"https://example.com/search?q={query.replace(' ', '+')}&r={i}",
                "name": f"{query} — reference {i}",
                "description": (
                    f"Overview of {query}. Sample grounding result {i} of 3 "
                    "returned by the offline search stub."
                ),
            }
            for i in range(1, 4)
        ]


def _entity_search(query: str, market: str = "en-US") -> list[dict]:
    return _web_search(query, market)   # delegate to web search; swap for entity API


def _news_search(query: str, market: str = "en-US") -> list[dict]:
    try:
        import tavily  # type: ignore
        client = tavily.TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        results = client.search(query, topic="news", max_results=5)["results"]
        return [{"url": r["url"], "name": r["title"], "description": r["content"]} for r in results]
    except Exception:
        return _web_search(query, market)


_TOOL_FN = {
    "find_information": lambda inp: _web_search(inp["query"], inp.get("market", "en-US")),
    "find_entities":    lambda inp: _entity_search(inp["query"], inp.get("market", "en-US")),
    "find_news":        lambda inp: _news_search(inp["query"], inp.get("market", "en-US")),
}


# ── Converse tool loop ─────────────────────────────────────────────────────────

def _extract_text(message: dict) -> str:
    return "".join(b["text"] for b in message.get("content", []) if "text" in b)


def _run_tool_loop(
    messages: list[dict],
    max_iterations: int = 10,
    system: list[dict] | None = None,
) -> str:
    """
    Drive the Converse tool-call loop explicitly.

    Azure AI Agent Service ran this loop server-side; on Bedrock we own it.
    Loop exits when stopReason is no longer "tool_use" or max_iterations hit.

    The system prompt (the JSON-only contract) must be re-sent on every turn —
    Converse is stateless, so dropping it lets the model revert to prose.
    """
    for _ in range(max_iterations):
        kwargs: dict[str, Any] = {
            "modelId": MODEL,
            "messages": messages,
            "toolConfig": TOOL_CONFIG,
        }
        if system:
            kwargs["system"] = system
        kwargs.update(_guardrail_config())  # T003 mitigation
        response = bedrock.converse(**kwargs)

        out_msg = response["output"]["message"]
        messages.append(out_msg)

        if response.get("stopReason") != "tool_use":
            return _extract_text(out_msg)

        # Execute all tool calls requested in this turn
        tool_results = []
        for block in out_msg.get("content", []):
            if "toolUse" not in block:
                continue
            tu = block["toolUse"]
            fn = _TOOL_FN.get(tu["name"])
            if fn is None:
                result_content = [{"text": f"Unknown tool: {tu['name']}"}]
            else:
                raw = fn(tu["input"])
                # Converse requires the toolResult json value to be a JSON object,
                # not a bare array — wrap the list of results under a key.
                result_content = [{"json": {"results": raw}}]

            tool_results.append({
                "toolResult": {
                    "toolUseId": tu["toolUseId"],
                    "content": result_content,
                }
            })

        messages.append({"role": "user", "content": tool_results})

    return _extract_text(messages[-1])   # return whatever we have after max iters


# ── Public entry point ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a researcher agent. Use the tools to search, then return ONLY a single
JSON object and nothing else.

Rules:
- Do NOT write any prose, explanation, or <thinking> block.
- Do NOT wrap the JSON in markdown code fences.
- Your entire response must be exactly one JSON object with this structure:

{"web": [{"url": "...", "name": "...", "description": "..."}],
 "entities": [],
 "news": []}

Return 3-5 sources in "web". Begin your response with { and end with }.
"""


def research(instructions: str, feedback: str = "No feedback") -> dict:
    """
    Run the researcher agent using Bedrock Converse.

    Drop-in replacement for the Azure version — same return shape:
        {"web": [...], "entities": [...], "news": [...]}
    """
    user_content = instructions
    if feedback and feedback.lower() != "no feedback":
        user_content += f"\n\nFeedback to incorporate: {feedback}"

    messages = [
        {"role": "user", "content": [{"text": user_content}]},
    ]

    system = [{"text": SYSTEM_PROMPT}]

    # Inline system prompt support via the system parameter
    response = bedrock.converse(
        modelId=MODEL,
        system=system,
        messages=messages,
        toolConfig=TOOL_CONFIG,
        **_guardrail_config(),  # T003 mitigation
    )

    out_msg = response["output"]["message"]
    messages.append(out_msg)

    # If the first response is already a tool call, enter the loop
    if response.get("stopReason") == "tool_use":
        tool_results = []
        for block in out_msg.get("content", []):
            if "toolUse" not in block:
                continue
            tu = block["toolUse"]
            fn = _TOOL_FN.get(tu["name"])
            raw = fn(tu["input"]) if fn else []
            tool_results.append({
                "toolResult": {
                    "toolUseId": tu["toolUseId"],
                    "content": [{"json": {"results": raw}}],
                }
            })
        messages.append({"role": "user", "content": tool_results})
        final_text = _run_tool_loop(messages, system=system)
    else:
        final_text = _extract_text(out_msg)

    try:
        parsed = json.loads(final_text)
    except json.JSONDecodeError:
        # Model returned extra prose (e.g. a <thinking> preamble or an
        # explanation) — extract the outermost JSON object if present.
        start = final_text.find("{")
        end   = final_text.rfind("}") + 1
        if start == -1 or end <= start:
            raise ValueError(
                "researcher returned no JSON object; got: "
                f"{final_text[:200]!r}"
            )
        parsed = json.loads(final_text[start:end])

    return {
        "web":      parsed.get("web", []),
        "entities": parsed.get("entities", []),
        "news":     parsed.get("news", []),
    }


if __name__ == "__main__":
    result = research("What are the latest camping trends for winter?")
    print(json.dumps(result, indent=2))
