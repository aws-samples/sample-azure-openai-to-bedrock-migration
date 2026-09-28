# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
BEFORE — Semantic Kernel with the Azure OpenAI connector.

Semantic Kernel wires a chat service into a Kernel. On Azure that's
AzureChatCompletion. The Bedrock move swaps the connector; your Kernel,
plugins, and planners stay the same. See ../after/bedrock_semantic_kernel.py.

Env vars:
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT
"""

import asyncio
import os

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

kernel = Kernel()
kernel.add_service(
    AzureChatCompletion(
        deployment_name=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "my-gpt-4o"),
        endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
    )
)


async def run(prompt: str) -> str:
    result = await kernel.invoke_prompt(prompt)
    return str(result)


if __name__ == "__main__":
    print(asyncio.run(run("In one sentence, what is Amazon Bedrock?")))
