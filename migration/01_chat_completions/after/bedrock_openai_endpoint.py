# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
AFTER — Path A: keep the OpenAI SDK, point it at Amazon Bedrock.

This is the smallest possible diff from `before/azure_chat.py`. You keep the
`openai` library and the `chat.completions.create(...)` call shape. Three
things change:

  1. `AzureOpenAI(...)`            -> `OpenAI(base_url=..., api_key=...)`
  2. Azure endpoint + API version -> the Bedrock OpenAI-compatible endpoint
                                      https://bedrock-runtime.{region}.amazonaws.com/openai/v1
  3. Azure key / deployment name  -> a Bedrock API key (bearer token) + a Bedrock model id

Auth: instead of storing a static key, generate a short-term Bedrock API key
from your normal AWS credentials with `aws-bedrock-token-generator`. The token
is derived from whatever credential chain boto3/AWS already uses (env vars,
profile, IAM role) and is valid for up to 12 hours.

Install:
    pip install openai aws-bedrock-token-generator

Env vars this expects (AWS):
    AWS_REGION            e.g. us-east-1  (or rely on your default AWS config)
    plus standard AWS credentials (profile, env vars, or IAM role)

Model id note: Bedrock uses model ids / inference profile ids, not Azure
deployment names. See ../../docs/model-mapping.md for the GPT -> Bedrock table.
"""

import os

from aws_bedrock_token_generator import provide_token
from openai import OpenAI

REGION = os.environ.get("AWS_REGION", "us-east-1")

# provide_token() mints a short-term Bedrock API key from your current AWS
# credentials, so no static key is stored in the app or environment.
client = OpenAI(
    base_url=f"https://bedrock-runtime.{REGION}.amazonaws.com/openai/v1",
    api_key=provide_token(region=REGION),
)

# Bedrock model id / inference profile id — replaces the Azure deployment name.
# Examples: "openai.gpt-oss-120b-1:0", "us.openai.gpt-5.6-terra", "global.openai.gpt-5.6-terra".
MODEL = os.environ.get("BEDROCK_MODEL_ID", "openai.gpt-oss-120b-1:0")

MESSAGES = [
    {"role": "system", "content": "You are a concise assistant. Answer in one sentence."},
    {"role": "user", "content": "What is Amazon Bedrock?"},
]


def _strip_reasoning(text: str) -> str:
    """Drop any <reasoning>...</reasoning> block(s).

    On the OpenAI-compatible endpoint, reasoning models (e.g. gpt-oss) inline
    their chain-of-thought as <reasoning>...</reasoning> inside the message
    content — there's no separate field, and there may be more than one block.
    Return just the answer.
    """
    import re

    return re.sub(r"<reasoning>.*?</reasoning>", "", text or "", flags=re.DOTALL).strip()


def chat_once() -> str:
    """Single, non-streaming completion — identical call shape to the Azure version."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=MESSAGES,
        temperature=0.2,
        max_tokens=256,
    )
    return _strip_reasoning(response.choices[0].message.content)


def chat_stream() -> str:
    """Streaming completion.

    The call is unchanged from the Azure version. The only wrinkle is display:
    reasoning models stream a <reasoning>...</reasoning> preamble, so here we
    buffer the stream and print the answer once reasoning is stripped. (To show
    reasoning live, print each token as it arrives instead of buffering.)
    """
    stream = client.chat.completions.create(
        model=MODEL,
        messages=MESSAGES,
        temperature=0.2,
        max_tokens=256,
        stream=True,
    )
    chunks: list[str] = []
    for event in stream:
        if event.choices and event.choices[0].delta.content:
            chunks.append(event.choices[0].delta.content)
    answer = _strip_reasoning("".join(chunks))
    print(answer)
    return answer


if __name__ == "__main__":
    print("== non-streaming ==")
    print(chat_once())
    print("\n== streaming ==")
    chat_stream()
