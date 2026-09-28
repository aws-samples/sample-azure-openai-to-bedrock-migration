# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
BEFORE — Azure AI Search + Azure OpenAI embeddings (hybrid retrieval).

Sourced from Azure-Samples/contoso-creative-writer product.py (MIT).
Illustrates the patterns az2br flags:

  [auth]          AzureOpenAI, DefaultAzureCredential, get_bearer_token_provider
  [embeddings]    text-embedding-ada-002  (1536 dimensions)
  [vector_search] SearchClient, VectorizedQuery, semantic hybrid retrieval

The pipeline is two stages:
  1. LLM generates specialized search queries from context (product.prompty)
  2. Each query is embedded → Azure AI Search vector + semantic hybrid

Env vars required:
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_NAME,
    AZURE_SEARCH_ENDPOINT
"""

from __future__ import annotations

import os
from typing import Any

from azure.identity import DefaultAzureCredential, get_bearer_token_provider  # [HIGH auth]
from azure.search.documents import SearchClient                                  # [HIGH vector_search]
from azure.search.documents.models import (                                      # [HIGH vector_search]
    QueryAnswerType,
    QueryCaptionType,
    QueryType,
    VectorizedQuery,
)
from openai import AzureOpenAI                                                   # [HIGH auth]

EMBEDDING_MODEL    = "text-embedding-ada-002"   # [MEDIUM embeddings] — 1536 dimensions
SEARCH_INDEX_NAME  = "contoso-products"


def generate_embeddings(queries: list[str]) -> list[dict[str, Any]]:
    """
    Embed a list of queries with Azure OpenAI text-embedding-ada-002 (1536d).

    Returns: [{"item": query_str, "embedding": [float, ...]}, ...]
    """
    token_provider = get_bearer_token_provider(                    # [HIGH auth]
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default",
    )
    client = AzureOpenAI(                                          # [HIGH auth]
        azure_endpoint=f"https://{os.environ['AZURE_OPENAI_NAME']}.cognitiveservices.azure.com/",
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_ad_token_provider=token_provider,
    )
    response = client.embeddings.create(input=queries, model=EMBEDDING_MODEL)
    return [
        {"item": queries[i], "embedding": e.embedding}
        for i, e in enumerate(response.data)
    ]


def retrieve_products(
    items: list[dict[str, Any]],
    index_name: str = SEARCH_INDEX_NAME,
) -> list[dict[str, Any]]:
    """
    Hybrid vector + semantic search over an Azure AI Search index.

    For each embedded query: k-NN vector search + semantic re-ranking.
    Deduplicates results by id.
    """
    search_client = SearchClient(                                  # [HIGH vector_search]
        endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        index_name=index_name,
        credential=DefaultAzureCredential(),                       # [HIGH auth]
    )

    products: list[dict] = []
    seen_ids: set[str]   = set()

    for item in items:
        vector_query = VectorizedQuery(
            vector=item["embedding"],
            k_nearest_neighbors=3,
            fields="contentVector",
        )
        results = search_client.search(
            search_text=item["item"],
            vector_queries=[vector_query],
            query_type=QueryType.SEMANTIC,
            semantic_configuration_name="default",
            query_caption=QueryCaptionType.EXTRACTIVE,
            query_answer=QueryAnswerType.EXTRACTIVE,
            top=2,
        )
        for doc in results:
            if doc["id"] not in seen_ids:
                seen_ids.add(doc["id"])
                products.append({
                    "id":      doc["id"],
                    "title":   doc["title"],
                    "content": doc["content"],
                    "url":     doc["url"],
                })

    return products


def find_products(context: str) -> list[dict[str, Any]]:
    """
    Full pipeline: generate queries → embed → hybrid search → return products.

    Stage 1 (LLM query generation) is handled by product.prompty + prompty.execute().
    This module handles stages 2 and 3.
    """
    import json
    import prompty          # type: ignore
    import prompty.azure    # type: ignore

    queries_json = prompty.execute("product.prompty", inputs={"context": context})
    queries      = json.loads(queries_json)
    items        = generate_embeddings(queries)
    return retrieve_products(items)


if __name__ == "__main__":
    import json
    products = find_products("Can you use a selection of tents and backpacks as context?")
    print(json.dumps(products, indent=2))
