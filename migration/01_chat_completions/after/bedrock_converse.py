# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
AFTER — Path B: the Amazon Bedrock Converse API (boto3).

This is a larger diff than Path A, but it is the portable, idiomatic Bedrock
surface: one API shape (`converse` / `converse_stream`) that works across every
Bedrock model — Anthropic, Meta, Mistral, Amazon, and the OpenAI models — so
switching or A/B-testing models later becomes a config change, not a rewrite.

What changes vs. the OpenAI/Azure shape:

  - Auth is standard AWS: boto3 uses your credential chain (profile, env vars,
    IAM role). No API key, no bearer token.
  - The `system` message is a separate top-level `system=[...]` argument, not a
    role inside `messages`.
  - Each message's `content` is a list of content blocks: [{"text": "..."}].
  - Tuning params (temperature, max tokens) move into `inferenceConfig`.
  - The reply text is in the content blocks at response["output"]["message"]
    ["content"]; join every block that has a "text" key (a reasoning model may
    emit a reasoningContent block before the text block).

Install:
    pip install boto3

Env vars this expects (AWS):
    AWS_REGION            e.g. us-east-1  (or rely on your default AWS config)
    plus standard AWS credentials (profile, env vars, or IAM role)
"""

import os

import boto3

REGION = os.environ.get("AWS_REGION", "us-east-1")

# Standard AWS auth — boto3 resolves credentials from the usual chain.
client = boto3.client("bedrock-runtime", region_name=REGION)

# Any Bedrock model id / inference profile id works through Converse.
MODEL = os.environ.get("BEDROCK_MODEL_ID", "openai.gpt-oss-120b-1:0")

# System prompt is top-level in Converse, not a message role.
SYSTEM = [{"text": "You are a concise assistant. Answer in one sentence."}]

# Converse message content is a list of typed blocks.
MESSAGES = [
    {"role": "user", "content": [{"text": "What is Amazon Bedrock?"}]},
]

INFERENCE_CONFIG = {"temperature": 0.2, "maxTokens": 256}


def _reply_text(response: dict) -> str:
    """Join all text content blocks.

    Some Bedrock models (e.g. the OpenAI gpt-oss reasoning models) return a
    reasoningContent block *before* the text block, so don't assume content[0]
    is the answer — scan for text blocks.
    """
    blocks = response["output"]["message"]["content"]
    return "".join(b["text"] for b in blocks if "text" in b)


def chat_once() -> str:
    """Single, non-streaming completion via converse()."""
    response = client.converse(
        modelId=MODEL,
        system=SYSTEM,
        messages=MESSAGES,
        inferenceConfig=INFERENCE_CONFIG,
    )
    return _reply_text(response)


def chat_stream() -> str:
    """Streaming completion via converse_stream() — print tokens as they arrive."""
    response = client.converse_stream(
        modelId=MODEL,
        system=SYSTEM,
        messages=MESSAGES,
        inferenceConfig=INFERENCE_CONFIG,
    )
    chunks: list[str] = []
    for event in response["stream"]:
        # Text arrives in contentBlockDelta events.
        if "contentBlockDelta" in event:
            token = event["contentBlockDelta"]["delta"].get("text", "")
            if token:
                chunks.append(token)
                print(token, end="", flush=True)
    print()
    return "".join(chunks)


if __name__ == "__main__":
    print("== non-streaming ==")
    print(chat_once())
    print("\n== streaming ==")
    chat_stream()
