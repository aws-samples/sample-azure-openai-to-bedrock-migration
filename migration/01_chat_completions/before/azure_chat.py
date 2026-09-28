# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
BEFORE — Azure OpenAI chat completion.

This is the starting point: a typical Azure OpenAI chat client. Running
`az2br scan` over this file flags the Azure-specific auth, client init,
deployment name, and api_version so you know exactly what changes.

Env vars this expects (Azure):
    AZURE_OPENAI_ENDPOINT   e.g. https://my-resource.openai.azure.com
    AZURE_OPENAI_API_KEY    the Azure resource key
    AZURE_OPENAI_DEPLOYMENT the *deployment name* you created in Azure (not the model id)
"""

import os

from openai import AzureOpenAI

# Azure couples the client to a resource endpoint + key + API version.
client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version="2024-10-21",
)

# In Azure, `model` is the *deployment name* you chose, not the underlying model id.
DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "my-gpt-4o")

MESSAGES = [
    {"role": "system", "content": "You are a concise assistant. Answer in one sentence."},
    {"role": "user", "content": "What is Amazon Bedrock?"},
]


def chat_once() -> str:
    """Single, non-streaming completion."""
    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=MESSAGES,
        temperature=0.2,
        max_tokens=256,
    )
    return response.choices[0].message.content


def chat_stream() -> str:
    """Streaming completion — print tokens as they arrive, return the full text."""
    stream = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=MESSAGES,
        temperature=0.2,
        max_tokens=256,
        stream=True,
    )
    chunks: list[str] = []
    for event in stream:
        if event.choices and event.choices[0].delta.content:
            token = event.choices[0].delta.content
            chunks.append(token)
            print(token, end="", flush=True)
    print()
    return "".join(chunks)


if __name__ == "__main__":
    print("== non-streaming ==")
    print(chat_once())
    print("\n== streaming ==")
    chat_stream()
