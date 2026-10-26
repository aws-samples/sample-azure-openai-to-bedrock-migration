#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Researcher search tools as a STDIO MCP server — for Bedrock Managed Agents.

In Azure AI Agent Service the researcher's tools were Bing grounding plus the
`find_information` / `find_entities` / `find_news` function schemas in
functions.json. Bedrock Managed Agents (BMA, public preview) does not run your
tools for you: tools execute in the *execution environment you provide* — your
own host running `codex exec-server`, or an AgentCore Runtime. The documented
way to give a BMA session custom tools is an MCP server that runs there.

So this file is the same three tools, re-wrapped:

  functions.json (Azure)                    MCP tools/list (this file)
  ────────────────────────────────────────  ───────────────────────────────────
  {"type":"function","function":{           {"name": N, "description": D,
     "name": N, "description": D,              "inputSchema": <same schema>}
     "parameters": <schema>}}

The inner JSON Schema is identical — only the envelope changes (same theme as
the Converse and Responses API targets).

Protocol: newline-delimited JSON-RPC 2.0 over stdin/stdout. Diagnostics go to
stderr only; stdout must carry nothing but protocol messages.

Search backing is a stub, as in the other targets: set TAVILY_API_KEY and
`pip install tavily-python` for real results; otherwise a deterministic offline
fallback is returned. Forward the key to the server with the session's
`env_vars` (names only — do not put the value in the session config).

Runs with the Python standard library only. The exec server launches it, so it
must be reachable by an ABSOLUTE path inside the execution environment.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

PROTOCOL_VERSION = "2024-11-05"
MAX_QUERY_CHARS = 500

_QUERY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "maxLength": MAX_QUERY_CHARS,
            "description": "An optimal search query",
        },
        "market": {"type": "string", "description": "Market code, e.g. en-US"},
    },
    "required": ["query"],
    "additionalProperties": False,
}

TOOLS: list[dict[str, Any]] = [
    {
        "name": "find_information",
        "description": (
            "Finds information on the web given a query using a search API. "
            "Use for general information — not news or entity lookups."
        ),
        "inputSchema": _QUERY_SCHEMA,
    },
    {
        "name": "find_entities",
        "description": (
            "Finds entities (people, places, things) on the web. "
            "Use for entity lookups, not general information or news."
        ),
        "inputSchema": _QUERY_SCHEMA,
    },
    {
        "name": "find_news",
        "description": (
            "Finds recent news articles on the web given a query. "
            "Use when the user is looking for news."
        ),
        "inputSchema": _QUERY_SCHEMA,
    },
]


# ── Search backing (replace with Tavily / Brave / etc. in production) ─────────

def _stub_results(query: str) -> list[dict]:
    """Deterministic offline fallback. Neutral and plausible so the model
    treats it as usable grounding rather than an error to apologize for."""
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


def _tavily(query: str, **kwargs: Any) -> list[dict]:
    import tavily  # type: ignore  # optional dependency

    client = tavily.TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    results = client.search(query, max_results=5, **kwargs)["results"]
    return [
        {"url": r["url"], "name": r["title"], "description": r["content"]}
        for r in results
    ]


def _search(query: str, **kwargs: Any) -> list[dict]:
    if not os.environ.get("TAVILY_API_KEY"):
        return _stub_results(query)
    try:
        return _tavily(query, **kwargs)
    except Exception as exc:  # network, missing package, bad key, ...
        print(f"search backend failed, using stub: {type(exc).__name__}", file=sys.stderr)
        return _stub_results(query)


def run_tool(name: str, arguments: Any) -> list[dict]:
    """Execute one tool. Raises ValueError on invalid input.

    The model generates `arguments`, so validate them here — the MCP server is
    the trust boundary, not the prompt.
    """
    if not isinstance(arguments, dict):
        raise ValueError("arguments must be an object")
    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if len(query) > MAX_QUERY_CHARS:
        raise ValueError(f"query must be at most {MAX_QUERY_CHARS} characters")
    market = arguments.get("market", "en-US")
    if not isinstance(market, str):
        raise ValueError("market must be a string")

    if name == "find_news":
        return _search(query, topic="news")
    if name in ("find_information", "find_entities"):
        return _search(query)
    raise ValueError(f"unknown tool: {name}")


# ── JSON-RPC / MCP plumbing ───────────────────────────────────────────────────

def _error(request_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def handle(message: dict) -> dict | None:
    """Return the JSON-RPC response for one message, or None for notifications."""
    request_id = message.get("id")
    if request_id is None:  # notification (e.g. notifications/initialized)
        return None

    method = message.get("method")
    if method == "initialize":
        result: dict[str, Any] = {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "bma-researcher-search", "version": "1.0.0"},
        }
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        params = message.get("params") or {}
        try:
            results = run_tool(params.get("name", ""), params.get("arguments"))
        except ValueError as exc:
            return _error(request_id, -32602, str(exc))
        result = {"content": [{"type": "text", "text": json.dumps({"results": results})}]}
    elif method == "ping":
        result = {}
    else:
        return _error(request_id, -32601, "Method not found")
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def main() -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
            if not isinstance(message, dict):
                raise ValueError("not an object")
            response = handle(message)
        except (json.JSONDecodeError, ValueError):
            response = _error(None, -32700, "Invalid JSON-RPC request")
        if response is not None:
            print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
