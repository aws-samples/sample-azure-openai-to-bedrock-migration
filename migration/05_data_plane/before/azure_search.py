# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
BEFORE — Azure AI Search vector query.

A typical retrieval call: embed the query, search an Azure AI Search index, and
return the top hits to feed a RAG prompt. The Bedrock move is either a managed
Knowledge Base (retrieve API) or your own OpenSearch index. See ../after/ and
../ai-search-to-opensearch.md.

Env vars:
    AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_API_KEY, AZURE_SEARCH_INDEX
"""

import os

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

client = SearchClient(
    endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
    index_name=os.environ.get("AZURE_SEARCH_INDEX", "docs-index"),
    credential=AzureKeyCredential(os.environ["AZURE_SEARCH_API_KEY"]),
)


def search(query: str, k: int = 3) -> list[dict]:
    """Return the top-k documents for a text query."""
    results = client.search(search_text=query, top=k)
    return [{"id": r["id"], "content": r["content"], "score": r["@search.score"]} for r in results]


if __name__ == "__main__":
    for hit in search("What is Amazon Bedrock?"):
        print(hit["score"], hit["content"][:80])
