# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
AFTER — Amazon Bedrock Titan Text Embeddings V2 (boto3, InvokeModel).

Migrates the Azure ada-002 batch embedder to Titan V2. Two things change:

  1. Auth + client: AzureOpenAI(...) -> boto3 bedrock-runtime.
  2. Dimension: ada-002 is 1536; Titan V2 defaults to 1024 (also 512 / 256).
     Because the dimension changes, you MUST re-embed everything already in your
     vector store and rebuild the index at the new dimension — you cannot mix
     dimensions in one index. See ../dimension-mapping.md.

Titan V2 embeds one input text per InvokeModel call, so batch by iterating.
`dimensions` and `normalize` are Titan V2 body parameters. (Titan does NOT
accept maxTokenCount / topP.)

Env vars:
    AWS_REGION (default: us-east-1), plus standard AWS credentials.
"""

import json
import os

import boto3

REGION = os.environ.get("AWS_REGION", "us-east-1")
MODEL_ID = "amazon.titan-embed-text-v2:0"
# 1024 (default), 512, or 256. Pick once, then keep it fixed for the whole index.
OUTPUT_DIM = int(os.environ.get("TITAN_EMBED_DIM", "1024"))

client = boto3.client("bedrock-runtime", region_name=REGION)

DOCUMENTS = [
    "Amazon Bedrock is a managed service for foundation models.",
    "Titan Text Embeddings V2 outputs 1024-dimensional vectors by default.",
    "Azure OpenAI ada-002 outputs 1536-dimensional vectors.",
]


def embed_one(text: str) -> list[float]:
    body = json.dumps({"inputText": text, "dimensions": OUTPUT_DIM, "normalize": True})
    response = client.invoke_model(modelId=MODEL_ID, body=body)
    payload = json.loads(response["body"].read())
    return payload["embedding"]


def embed(texts: list[str]) -> list[list[float]]:
    """One vector per input text (Titan V2 takes a single inputText per call)."""
    return [embed_one(t) for t in texts]


if __name__ == "__main__":
    vectors = embed(DOCUMENTS)
    print(f"embedded {len(vectors)} docs, dim={len(vectors[0])}")  # dim=OUTPUT_DIM
