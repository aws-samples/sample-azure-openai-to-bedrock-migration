# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
AFTER — Semantic Kernel on Amazon Bedrock via the OpenAI-compatible endpoint.

Semantic Kernel's OpenAIChatCompletion service accepts a custom async_client, so
you can point SK at Bedrock's OpenAI-compatible endpoint without a provider-
specific connector. This keeps the example portable and dependency-light: your
Kernel, plugins, and planners are unchanged — only the chat service swaps.

  AzureChatCompletion(deployment_name=..., endpoint=..., api_key=...)
    ->  OpenAIChatCompletion(ai_model_id=<bedrock model id>,
                             async_client=OpenAI-compatible client at Bedrock)

Alternative: Semantic Kernel also ships an Amazon Bedrock connector
(BedrockChatCompletion) that uses AWS credentials directly; prefer it if you
want native SigV4 auth. Check the version of semantic-kernel you have installed
for its exact constructor, since the connector surface evolves.

Install:
    pip install semantic-kernel openai aws-bedrock-token-generator

Env vars:
    AWS_REGION (default us-east-1), BEDROCK_MODEL_ID (default openai.gpt-oss-120b-1:0),
    plus standard AWS credentials.
"""

import asyncio
import os

from aws_bedrock_token_generator import provide_token
from openai import AsyncOpenAI
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

REGION = os.environ.get("AWS_REGION", "us-east-1")
MODEL = os.environ.get("BEDROCK_MODEL_ID", "openai.gpt-oss-120b-1:0")

# A standard OpenAI async client pointed at Bedrock's OpenAI-compatible endpoint.
bedrock_client = AsyncOpenAI(
    base_url=f"https://bedrock-runtime.{REGION}.amazonaws.com/openai/v1",
    api_key=provide_token(region=REGION),
)

kernel = Kernel()
kernel.add_service(
    OpenAIChatCompletion(ai_model_id=MODEL, async_client=bedrock_client)
)


async def run(prompt: str) -> str:
    result = await kernel.invoke_prompt(prompt)
    return str(result)


if __name__ == "__main__":
    print(asyncio.run(run("In one sentence, what is Amazon Bedrock?")))
