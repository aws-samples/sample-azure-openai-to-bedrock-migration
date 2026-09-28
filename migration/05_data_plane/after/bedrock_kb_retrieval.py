# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
AFTER — Amazon Bedrock Knowledge Bases + Titan Text Embeddings v2.

Migrates ../before/azure_hybrid_search.py with these substitutions:

  BEFORE (Azure)                             AFTER (Bedrock)
  ─────────────────────────────────────────  ────────────────────────────────────
  AzureOpenAI + text-embedding-ada-002       boto3 bedrock-runtime + Titan v2
  DefaultAzureCredential / token provider    IAM role (boto3 session)
  SearchClient (azure.search.documents)      bedrock-agent-runtime retrieve()
  VectorizedQuery + semantic hybrid          Native KB hybrid search (auto)
  Azure AI Search index + contentVector      Bedrock Knowledge Base (any backend)

⚠ Dimension mismatch:
  ada-002 = 1536d | Titan v2 = 1024d (default) or 256d / 512d
  You MUST re-embed your corpus when switching embedding models — Titan v2 has
  no 1536d setting, so re-indexing is mandatory.
  See migration/02_embeddings/dimension-mapping.md.

Two retrieval strategies are provided:
  A. retrieve_from_kb()  — Bedrock Knowledge Bases (managed, recommended)
  B. retrieve_with_opensearch()  — manual Titan v2 + OpenSearch Serverless
     (use when you need full control over the index or hybrid configuration)

Env vars:
    AWS_REGION          (default: us-east-1)
    BEDROCK_MODEL_ID    (default: openai.gpt-oss-120b-1:0 — for query generation)
    KNOWLEDGE_BASE_ID   (required for Strategy A)
    OPENSEARCH_ENDPOINT (required for Strategy B)
    OPENSEARCH_INDEX    (required for Strategy B, default: contoso-products)
"""

from __future__ import annotations

import json
import os
from typing import Any

import boto3

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

bedrock_runtime = boto3.client("bedrock-runtime",       region_name=AWS_REGION)
bedrock_agent   = boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)

TITAN_EMBED_MODEL = "amazon.titan-embed-text-v2:0"   # 1024d (default), 256d, 512d available


# ── Stage 1: LLM query generation (replaces product.prompty + prompty.execute) ─

QUERY_GEN_PROMPT = """\
You are an AI assistant. Given context, generate up to 5 specialized search
queries for a product index. Return ONLY a JSON array of strings, e.g.:
["outdoor tents", "camping shelter", "backpacking tent"]

Context: {context}"""


def generate_queries(context: str) -> list[str]:
    """Generate specialized product search queries using Bedrock Converse."""
    model_id = os.environ.get("BEDROCK_MODEL_ID", "openai.gpt-oss-120b-1:0")
    response = bedrock_runtime.converse(
        modelId=model_id,
        messages=[{
            "role": "user",
            "content": [{"text": QUERY_GEN_PROMPT.format(context=context)}],
        }],
    )
    text = "".join(
        b["text"] for b in response["output"]["message"]["content"] if "text" in b
    )
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end   = text.rfind("]") + 1
        return json.loads(text[start:end])


# ── Stage 2A: Bedrock Knowledge Bases retrieval (recommended) ──────────────────

def retrieve_from_kb(
    queries: list[str],
    knowledge_base_id: str | None = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Retrieve products from a Bedrock Knowledge Base.

    Bedrock KB handles embedding, indexing, and hybrid search internally.
    You don't call the embedding model directly — KB does it on query.

    Migration note:
      Azure AI Search hybrid = vector + BM25 + semantic re-ranking (explicit)
      Bedrock KB hybrid      = configured at KB creation time (transparent)
    """
    kb_id = knowledge_base_id or os.environ["KNOWLEDGE_BASE_ID"]
    products: list[dict] = []
    seen_ids: set[str]   = set()

    for query in queries:
        response = bedrock_agent.retrieve(
            knowledgeBaseId=kb_id,
            retrievalQuery={"text": query},
            retrievalConfiguration={
                "vectorSearchConfiguration": {
                    "numberOfResults": top_k,
                    "overrideSearchType": "HYBRID",   # HYBRID | SEMANTIC
                }
            },
        )
        for result in response.get("retrievalResults", []):
            metadata = result.get("metadata", {})
            doc_id   = metadata.get("id", result["location"].get("s3Location", {}).get("uri", ""))
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                products.append({
                    "id":      doc_id,
                    "title":   metadata.get("title", ""),
                    "content": result["content"]["text"],
                    "url":     metadata.get("url", ""),
                    "score":   result.get("score", 0.0),
                })

    return products


# ── Stage 2B: Manual Titan v2 + OpenSearch Serverless (optional) ───────────────

def generate_embeddings_titan(queries: list[str], dimensions: int = 1024) -> list[dict[str, Any]]:
    """
    Embed queries using Amazon Titan Text Embeddings v2.

    dimensions: 256 | 512 | 1024 (default)
    Note: ada-002 uses 1536d — Titan v2 has no 1536d setting, so re-embed and re-index.
    """
    results = []
    for query in queries:
        response = bedrock_runtime.invoke_model(
            modelId=TITAN_EMBED_MODEL,
            body=json.dumps({
                "inputText": query,
                "dimensions": dimensions,
                "normalize": True,
            }),
            contentType="application/json",
            accept="application/json",
        )
        body = json.loads(response["body"].read())
        results.append({"item": query, "embedding": body["embedding"]})
    return results


def retrieve_with_opensearch(
    items: list[dict[str, Any]],
    index_name: str | None = None,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """
    k-NN retrieval against Amazon OpenSearch Serverless.

    Use when you need direct control over the index schema or hybrid config.
    Requires: opensearch-py  (pip install opensearch-py)
    """
    from opensearchpy import OpenSearch, RequestsHttpConnection  # type: ignore
    from requests_aws4auth import AWS4Auth                        # type: ignore

    endpoint = os.environ["OPENSEARCH_ENDPOINT"]
    idx      = index_name or os.environ.get("OPENSEARCH_INDEX", "contoso-products")

    session    = boto3.Session()
    creds      = session.get_credentials()
    awsauth    = AWS4Auth(
        creds.access_key, creds.secret_key, AWS_REGION, "aoss",
        session_token=creds.token,
    )
    os_client = OpenSearch(
        hosts=[{"host": endpoint, "port": 443}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
    )

    products: list[dict] = []
    seen_ids: set[str]   = set()

    for item in items:
        query_body = {
            "size": top_k,
            "query": {
                "knn": {
                    "content_vector": {
                        "vector": item["embedding"],
                        "k": top_k,
                    }
                }
            },
            "_source": ["id", "title", "content", "url"],
        }
        response = os_client.search(index=idx, body=query_body)
        for hit in response["hits"]["hits"]:
            src = hit["_source"]
            if src["id"] not in seen_ids:
                seen_ids.add(src["id"])
                products.append({
                    "id":      src["id"],
                    "title":   src.get("title", ""),
                    "content": src.get("content", ""),
                    "url":     src.get("url", ""),
                    "score":   hit["_score"],
                })

    return products


# ── Public entry point ─────────────────────────────────────────────────────────

def find_products(
    context: str,
    strategy: str = "kb",  # "kb" | "opensearch"
) -> list[dict[str, Any]]:
    """
    Full pipeline: generate queries → retrieve products.

    strategy="kb"         → Bedrock Knowledge Bases (recommended)
    strategy="opensearch" → manual Titan v2 + OpenSearch Serverless
    """
    queries = generate_queries(context)

    if strategy == "kb":
        return retrieve_from_kb(queries)
    elif strategy == "opensearch":
        items = generate_embeddings_titan(queries)
        return retrieve_with_opensearch(items)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}. Choose 'kb' or 'opensearch'.")


if __name__ == "__main__":
    products = find_products(
        "Can you use a selection of tents and backpacks as context?",
        strategy="kb",
    )
    print(json.dumps(products, indent=2))
