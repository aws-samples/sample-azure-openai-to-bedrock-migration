# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
AFTER — LangChain agent backed by Amazon Bedrock (ChatBedrockConverse).

The connector swap: langchain_openai.AzureChatOpenAI -> langchain_aws
.ChatBedrockConverse. Your surrounding LangChain code (chains, agents, tools,
.invoke()) is unchanged — you only change how the LLM object is built.

ChatBedrockConverse uses the Converse API under the hood, so it works across all
Bedrock models. Auth is standard AWS credentials; there's no endpoint, key, or
api_version to set.

Install:
    pip install langchain-aws

Env vars:
    AWS_REGION (default us-east-1), BEDROCK_MODEL_ID (default openai.gpt-oss-120b-1:0),
    plus standard AWS credentials.
"""

import os

from langchain_aws import ChatBedrockConverse

llm = ChatBedrockConverse(
    model=os.environ.get("BEDROCK_MODEL_ID", "openai.gpt-oss-120b-1:0"),
    region_name=os.environ.get("AWS_REGION", "us-east-1"),
    temperature=0.2,
)


def run(prompt: str) -> str:
    response = llm.invoke(prompt)
    return _text(response.content)


def _text(content) -> str:
    """Extract plain text from a LangChain message.

    With a reasoning model (e.g. OpenAI gpt-oss), ChatBedrockConverse returns
    `content` as a list of blocks (reasoning + text) rather than a plain string.
    Handle both shapes.
    """
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif isinstance(block, str):
            parts.append(block)
    return "".join(parts)


if __name__ == "__main__":
    print(run("In one sentence, what is Amazon Bedrock?"))
