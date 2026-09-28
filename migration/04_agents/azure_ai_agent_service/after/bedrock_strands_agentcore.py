# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
AFTER — Strands Agents + Amazon Bedrock AgentCore Runtime researcher agent.

This is the strategic target for Azure AI Agent Service migrations:

  Azure AI Agent Service (managed)     →   AgentCore Runtime (managed)
  BingGroundingTool (Azure-bundled)    →   @tool decorator (Strands)
  DefaultAzureCredential               →   IAM role
  Proprietary agent loop               →   Strands open-source loop

AgentCore Runtime provides the same managed primitives Azure AI Agent Service
bundled — Memory, Identity, Gateway (MCP), Code Interpreter — as composable
AWS services rather than a single opaque platform.

Deploy:
    pip install strands-agents strands-agents-tools bedrock-agentcore

Local test:
    python bedrock_strands_agentcore.py

Deploy to AgentCore Runtime:
    bedrock-agentcore deploy --entrypoint bedrock_strands_agentcore:app

Env vars:
    AWS_REGION        (default: us-east-1)
    BEDROCK_MODEL_ID  (default: us.openai.gpt-5.6-terra)
    TAVILY_API_KEY    (optional — used by search tools)
"""

from __future__ import annotations

import json
import os
from typing import Any

from strands import Agent, tool                                # pip install strands-agents
from bedrock_agentcore.runtime import BedrockAgentCoreApp     # pip install bedrock-agentcore

# ── Tool definitions ────────────────────────────────────────────────────────────
# Each @tool replaces one entry in functions.json + the Azure BingGroundingTool.
# Strands reads the docstring and type hints to build the tool schema automatically.

@tool
def find_information(query: str, market: str = "en-US") -> list[dict[str, str]]:
    """
    Find general information on the web given a query.
    Use for factual lookups — not news or entity searches.

    Args:
        query:  An optimal search query.
        market: Market/language code, e.g. en-US (default).
    """
    return _search(query, topic="general")


@tool
def find_entities(query: str, market: str = "en-US") -> list[dict[str, str]]:
    """
    Find entities (people, places, or things) on the web.
    Use for entity lookups, not general information or news.

    Args:
        query:  An optimal search query for people, places, or things.
        market: Market/language code, e.g. en-US (default).
    """
    return _search(query, topic="general")


@tool
def find_news(query: str, market: str = "en-US") -> list[dict[str, str]]:
    """
    Find recent news articles on the web given a query.
    Use when the user is looking for news.

    Args:
        query:  An optimal search query for news.
        market: Market/language code, e.g. en-US (default).
    """
    return _search(query, topic="news")


def _search(query: str, topic: str = "general") -> list[dict[str, str]]:
    """Search backing — replace with Tavily, Brave, or any search API."""
    try:
        import tavily  # type: ignore
        client = tavily.TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        kwargs: dict[str, Any] = {"max_results": 5}
        if topic == "news":
            kwargs["topic"] = "news"
        results = client.search(query, **kwargs)["results"]
        return [{"url": r["url"], "name": r["title"], "description": r["content"]}
                for r in results]
    except Exception:
        return [{"url": "https://example.com", "name": f"Result for: {query}",
                 "description": "Stub — configure TAVILY_API_KEY for live results."}]


# ── Strands Agent ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
Act as a researcher agent. Process the user query, search the web using your
tools, and return findings as a JSON object with this exact structure:

{"web": [{"url": "...", "name": "...", "description": "..."}],
 "entities": [],
 "news": []}

Return 4–5 sources. Only return the JSON object — no markdown fences, no prose.
"""

agent = Agent(
    model=os.environ.get("BEDROCK_MODEL_ID", "us.openai.gpt-5.6-terra"),
    system_prompt=SYSTEM_PROMPT,
    tools=[find_information, find_entities, find_news],
)


def research(instructions: str, feedback: str = "No feedback") -> dict:
    """
    Run the researcher agent via Strands.

    Same return shape as the Azure version:
        {"web": [...], "entities": [...], "news": [...]}
    """
    prompt = instructions
    if feedback and feedback.lower() != "no feedback":
        prompt += f"\n\nFeedback to incorporate: {feedback}"

    result = agent(prompt)
    text = result.message if hasattr(result, "message") else str(result)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end   = text.rfind("}") + 1
        parsed = json.loads(text[start:end])

    return {
        "web":      parsed.get("web", []),
        "entities": parsed.get("entities", []),
        "news":     parsed.get("news", []),
    }


# ── AgentCore Runtime entrypoint ───────────────────────────────────────────────
# When deployed to AgentCore Runtime, this is the container entrypoint.
# Local: python bedrock_strands_agentcore.py  (runs a local test)
# Deployed: bedrock-agentcore deploy --entrypoint bedrock_strands_agentcore:app

app = BedrockAgentCoreApp()

@app.entrypoint
def agent_invocation(payload: dict, context: Any) -> dict:
    instructions = payload.get("instructions", "")
    feedback     = payload.get("feedback", "No feedback")
    return research(instructions, feedback)


if __name__ == "__main__":
    # Local smoke test — runs without AgentCore Runtime
    result = research("What are the latest camping trends for winter?")
    print(json.dumps(result, indent=2))
    # app.run()   # uncomment to start AgentCore local server
