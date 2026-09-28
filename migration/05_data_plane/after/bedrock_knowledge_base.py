# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
AFTER — Amazon Bedrock Knowledge Bases retrieval (managed RAG).

Replaces the Azure AI Search query with a Bedrock Knowledge Base `retrieve`
call. The Knowledge Base owns the embedding + vector store (e.g. OpenSearch
Serverless) and returns the top chunks for a text query — you don't manage the
index or embed the query yourself.

Two calls are available:
  - retrieve             -> just the retrieved chunks (shown here; drop-in for
                            the Azure search() function).
  - retrieve_and_generate -> retrieval + a grounded answer in one call.

Set up the Knowledge Base once (S3 data source + embedding model + vector
store); see ../ai-search-to-opensearch.md. Then query it by ID.

Env vars:
    AWS_REGION (default us-east-1), BEDROCK_KB_ID (your knowledge base id),
    plus standard AWS credentials.
"""

import os

import boto3

client = boto3.client("bedrock-agent-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))
KB_ID = os.environ.get("BEDROCK_KB_ID", "REPLACE_WITH_KB_ID")


def search(query: str, k: int = 3) -> list[dict]:
    """Return the top-k retrieved chunks — same signature as the Azure version."""
    response = client.retrieve(
        knowledgeBaseId=KB_ID,
        retrievalQuery={"text": query},
        retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": k}},
    )
    hits = []
    for r in response.get("retrievalResults", []):
        hits.append(
            {
                "id": (r.get("location") or {}).get("s3Location", {}).get("uri", ""),
                "content": (r.get("content") or {}).get("text", ""),
                "score": r.get("score"),
            }
        )
    return hits


if __name__ == "__main__":
    for hit in search("What is Amazon Bedrock?"):
        print(hit["score"], hit["content"][:80])
