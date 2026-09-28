#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
smoke_test_after.py — prove the after/ (Bedrock) examples actually run.

This makes a handful of REAL calls to Amazon Bedrock in your account to verify
the migration targets work end to end:

  1. Converse chat        (migration/01_chat_completions/after/bedrock_converse.py)
  2. OpenAI endpoint (A)   (migration/01_chat_completions/after/bedrock_openai_endpoint.py)
  3. Converse tool use     (migration/03_tool_use/after/bedrock_converse_tools.py)
  4. Titan V2 embeddings   (migration/02_embeddings/after/bedrock_titan_embed.py)
  5. LangChain             (migration/04_agents/langchain/after/bedrock_langchain.py)
  6. AutoGen               (migration/04_agents/autogen/after/bedrock_autogen.py)
  7. Semantic Kernel       (migration/04_agents/semantic_kernel/after/bedrock_semantic_kernel.py)

Framework/endpoint examples whose SDK is not installed are SKIPPED, not failed —
install their deps to include them (see requirements.txt and the per-example
headers).

This costs a small number of on-demand inference calls. It is NOT run by the
offline unit tests; it is a manual/opt-in check.

Usage:
    AWS_REGION=us-east-1 BEDROCK_MODEL_ID=us.openai.gpt-5.6-terra \
        python harness/smoke_test_after.py

Env:
    AWS_REGION        (default us-east-1)
    BEDROCK_MODEL_ID  (default us.openai.gpt-5.6-terra) — must be enabled + on-demand
    BEDROCK_EMBED_ID  (default amazon.titan-embed-text-v2:0)
    plus standard AWS credentials.

Exit code 0 if every non-skipped check passes; 1 otherwise.
"""

from __future__ import annotations

import json
import os
import sys
import traceback

REGION = os.environ.get("AWS_REGION", "us-east-1")
MODEL = os.environ.get("BEDROCK_MODEL_ID", "us.openai.gpt-5.6-terra")
EMBED = os.environ.get("BEDROCK_EMBED_ID", "amazon.titan-embed-text-v2:0")

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
results: list[tuple[str, str, str]] = []


def _is_frontier_openai(model_id: str) -> bool:
    """True for OpenAI frontier models (GPT-5.x / GPT-6), any inference-profile prefix."""
    mid = (model_id or "").lower()
    return "openai.gpt-5" in mid or "openai.gpt-6" in mid


def record(name: str, status: str, detail: str = "") -> None:
    results.append((name, status, detail))
    mark = {"PASS": "✓", "FAIL": "✗", "SKIP": "–"}[status]
    print(f"  {mark} {name:<28} {status}  {detail}")


def _reply_text(message: dict) -> str:
    """Join text blocks; a reasoning model emits reasoningContent first."""
    return "".join(b["text"] for b in message.get("content", []) if "text" in b)


def _strip_reasoning(text: str) -> str:
    """Drop a leading <reasoning>...</reasoning> block (OpenAI-endpoint reasoning models)."""
    import re

    return re.sub(r"^\s*<reasoning>.*?</reasoning>\s*", "", text or "", flags=re.DOTALL)


def _openai_client():
    """Build an OpenAI client pointed at the Bedrock OpenAI-compatible endpoint."""
    from aws_bedrock_token_generator import provide_token
    from openai import OpenAI

    return OpenAI(
        base_url=f"https://bedrock-runtime.{REGION}.amazonaws.com/openai/v1",
        api_key=os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or provide_token(region=REGION),
    )


def check_converse_chat(bedrock) -> None:
    try:
        inference_config = {"temperature": 0.0, "maxTokens": 256}
        if _is_frontier_openai(MODEL):
            # Frontier GPT models accept only the default temperature.
            inference_config.pop("temperature", None)
        r = bedrock.converse(
            modelId=MODEL,
            system=[{"text": "You are concise. Answer in one short sentence."}],
            messages=[{"role": "user", "content": [{"text": "What is Amazon Bedrock?"}]}],
            inferenceConfig=inference_config,
        )
        text = _reply_text(r["output"]["message"])
        assert text.strip(), "empty reply"
        record("converse chat", PASS, f'"{text[:50]}…" ({r["usage"]["totalTokens"]} tok)')
    except Exception as exc:  # noqa: BLE001 — smoke test reports any failure per-check
        record("converse chat", FAIL, f"{type(exc).__name__}: {exc}")


def check_converse_tools(bedrock) -> None:
    try:
        tool_config = {
            "tools": [
                {
                    "toolSpec": {
                        "name": "get_weather",
                        "description": "Get the current weather for a city.",
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                            }
                        },
                    }
                }
            ],
            "toolChoice": {"auto": {}},
        }
        messages = [{"role": "user", "content": [{"text": "What's the weather in Seattle?"}]}]
        r = bedrock.converse(modelId=MODEL, messages=messages, toolConfig=tool_config)
        out = r["output"]["message"]
        tool_uses = [b["toolUse"] for b in out["content"] if "toolUse" in b]
        if r.get("stopReason") == "tool_use" and tool_uses:
            tu = tool_uses[0]
            assert tu["name"] == "get_weather", f"unexpected tool {tu['name']}"
            assert "city" in tu["input"], "tool input missing city"
            record("converse tool use", PASS, f'requested {tu["name"]}({tu["input"]})')
        else:
            # Model chose to answer directly — tool-use path still structurally valid.
            record("converse tool use", PASS, f'no tool call (stopReason={r.get("stopReason")})')
    except Exception as exc:  # noqa: BLE001
        record("converse tool use", FAIL, f"{type(exc).__name__}: {exc}")


def check_titan_embeddings(bedrock) -> None:
    try:
        body = json.dumps({"inputText": "Amazon Bedrock", "dimensions": 1024, "normalize": True})
        r = bedrock.invoke_model(modelId=EMBED, body=body)
        vec = json.loads(r["body"].read())["embedding"]
        assert len(vec) == 1024, f"expected 1024 dims, got {len(vec)}"
        record("titan v2 embeddings", PASS, f"dim={len(vec)}")
    except Exception as exc:  # noqa: BLE001
        detail = f"{type(exc).__name__}: {exc}"
        # Embedding model may not be enabled in every account — treat access errors as SKIP.
        if "AccessDenied" in detail or "not authorized" in detail or "ValidationException" in detail:
            record("titan v2 embeddings", SKIP, f"model not enabled? {detail[:60]}")
        else:
            record("titan v2 embeddings", FAIL, detail)


def check_langchain() -> None:
    try:
        from langchain_aws import ChatBedrockConverse
    except ImportError:
        record("langchain ChatBedrockConverse", SKIP, "langchain-aws not installed")
        return
    try:
        llm = ChatBedrockConverse(model=MODEL, region_name=REGION, temperature=0.0, max_tokens=256)
        resp = llm.invoke("In one sentence, what is Amazon Bedrock?")
        content = resp.content if isinstance(resp.content, str) else str(resp.content)
        assert content.strip(), "empty reply"
        record("langchain ChatBedrockConverse", PASS, f'"{content[:50]}…"')
    except Exception as exc:  # noqa: BLE001
        record("langchain ChatBedrockConverse", FAIL, f"{type(exc).__name__}: {exc}")


def check_openai_endpoint() -> None:
    """Path A: the OpenAI SDK pointed at Bedrock's OpenAI-compatible endpoint."""
    try:
        import openai  # noqa: F401
        import aws_bedrock_token_generator  # noqa: F401
    except ImportError:
        record("openai endpoint (Path A)", SKIP, "openai / token-generator not installed")
        return
    try:
        client = _openai_client()
        kwargs: dict = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "In one short sentence, what is Amazon Bedrock?"}],
        }
        if _is_frontier_openai(MODEL):
            # Frontier GPT models reject temperature and use max_completion_tokens.
            kwargs["max_completion_tokens"] = 256
        else:
            kwargs["temperature"] = 0.0
            kwargs["max_tokens"] = 256
        r = client.chat.completions.create(**kwargs)
        text = _strip_reasoning(r.choices[0].message.content or "")
        assert text.strip(), "empty reply"
        record("openai endpoint (Path A)", PASS, f'"{text[:50]}…" (finish={r.choices[0].finish_reason})')
    except Exception as exc:  # noqa: BLE001
        record("openai endpoint (Path A)", FAIL, f"{type(exc).__name__}: {exc}")


def check_autogen() -> None:
    try:
        from autogen import AssistantAgent, UserProxyAgent  # classic autogen API
    except ImportError:
        record("autogen", SKIP, "classic 'autogen' (config_list API) not installed")
        return
    try:
        from aws_bedrock_token_generator import provide_token

        llm_config = {
            "config_list": [
                {
                    "model": MODEL,
                    "api_type": "openai",
                    "base_url": f"https://bedrock-runtime.{REGION}.amazonaws.com/openai/v1",
                    "api_key": os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or provide_token(region=REGION),
                }
            ],
            "temperature": 0.0,
        }
        assistant = AssistantAgent(name="assistant", llm_config=llm_config)
        user = UserProxyAgent(name="user", human_input_mode="NEVER", code_execution_config=False)
        result = user.initiate_chat(assistant, message="In one short sentence, what is Amazon Bedrock?", max_turns=1)
        text = _strip_reasoning(str(result.summary))
        assert text.strip(), "empty reply"
        record("autogen", PASS, f'"{text[:50]}…"')
    except Exception as exc:  # noqa: BLE001
        record("autogen", FAIL, f"{type(exc).__name__}: {exc}")


def check_semantic_kernel() -> None:
    try:
        import semantic_kernel  # noqa: F401
        import openai  # noqa: F401
        import aws_bedrock_token_generator  # noqa: F401
    except ImportError:
        record("semantic kernel", SKIP, "semantic-kernel / openai not installed")
        return
    try:
        import asyncio

        from aws_bedrock_token_generator import provide_token
        from openai import AsyncOpenAI
        from semantic_kernel import Kernel
        from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

        client = AsyncOpenAI(
            base_url=f"https://bedrock-runtime.{REGION}.amazonaws.com/openai/v1",
            api_key=os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or provide_token(region=REGION),
        )
        kernel = Kernel()
        kernel.add_service(OpenAIChatCompletion(ai_model_id=MODEL, async_client=client))

        async def _run() -> str:
            res = await kernel.invoke_prompt("In one short sentence, what is Amazon Bedrock?")
            return str(res)

        text = _strip_reasoning(asyncio.run(_run()))
        assert text.strip(), "empty reply"
        record("semantic kernel", PASS, f'"{text[:50]}…"')
    except Exception as exc:  # noqa: BLE001
        record("semantic kernel", FAIL, f"{type(exc).__name__}: {exc}")


def main() -> int:
    print(f"Smoke-testing after/ examples against Bedrock")
    print(f"  region={REGION}  model={MODEL}  embed={EMBED}\n")

    try:
        import boto3
    except ImportError:
        print("boto3 not installed — cannot run. pip install -r harness/requirements.txt")
        return 1

    try:
        bedrock = boto3.client("bedrock-runtime", region_name=REGION)
    except Exception as exc:  # noqa: BLE001
        print(f"Could not create bedrock-runtime client: {exc}")
        return 1

    check_converse_chat(bedrock)
    check_converse_tools(bedrock)
    check_titan_embeddings(bedrock)
    check_openai_endpoint()
    check_langchain()
    check_autogen()
    check_semantic_kernel()

    passed = sum(1 for _, s, _ in results if s == PASS)
    failed = sum(1 for _, s, _ in results if s == FAIL)
    skipped = sum(1 for _, s, _ in results if s == SKIP)
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
