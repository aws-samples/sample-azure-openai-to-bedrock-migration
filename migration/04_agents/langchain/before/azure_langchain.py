# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
BEFORE — LangChain agent backed by Azure OpenAI (AzureChatOpenAI).

A minimal LangChain chat model + a tool, the way it's typically wired against
Azure. The Bedrock move is a connector swap — see ../bedrock_langchain.py.

Env vars:
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT
    AZURE_OPENAI_API_VERSION (default 2024-10-21)
"""

import os

from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "my-gpt-4o"),
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    temperature=0.2,
)


def run(prompt: str) -> str:
    response = llm.invoke(prompt)
    return response.content


if __name__ == "__main__":
    print(run("In one sentence, what is Amazon Bedrock?"))
