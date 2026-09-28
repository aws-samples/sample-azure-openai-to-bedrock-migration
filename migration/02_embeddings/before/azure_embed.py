# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
BEFORE — Azure OpenAI embeddings (text-embedding-ada-002, 1536 dimensions).

Typical pattern: embed a batch of documents and store the vectors. The stored
dimension (1536) is baked into your vector store's index — which is exactly what
makes the Bedrock move a re-embed, not a swap. See ../after/ and
../dimension-mapping.md.

Env vars:
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY
    AZURE_OPENAI_EMBED_DEPLOYMENT   deployment name for text-embedding-ada-002
"""

import os

from openai import AzureOpenAI

client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version="2024-10-21",
)

EMBED_DEPLOYMENT = os.environ.get("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-ada-002")

DOCUMENTS = [
    "Amazon Bedrock is a managed service for foundation models.",
    "Titan Text Embeddings V2 outputs 1024-dimensional vectors by default.",
    "Azure OpenAI ada-002 outputs 1536-dimensional vectors.",
]


def embed(texts: list[str]) -> list[list[float]]:
    """Return one embedding vector per input text."""
    response = client.embeddings.create(model=EMBED_DEPLOYMENT, input=texts)
    # Preserve input order; the API returns items with an `index`.
    items = sorted(response.data, key=lambda d: d.index)
    return [item.embedding for item in items]


if __name__ == "__main__":
    vectors = embed(DOCUMENTS)
    print(f"embedded {len(vectors)} docs, dim={len(vectors[0])}")  # dim=1536
