# 05 — Data Plane: Azure AI Search → Amazon OpenSearch / Bedrock Knowledge Bases

Your vector index and retrieval layer move to AWS. There are two targets
depending on how much you want managed for you.

- **Before:** [`before/azure_search.py`](before/azure_search.py) — Azure AI Search query
- **After:** [`after/bedrock_knowledge_base.py`](after/bedrock_knowledge_base.py) — Bedrock Knowledge Base `retrieve`
- **Index & setup details:** [`ai-search-to-opensearch.md`](ai-search-to-opensearch.md)

---

## Two targets

| | **Bedrock Knowledge Bases** (managed RAG) | **Amazon OpenSearch** (DIY) |
|---|---|---|
| You manage | S3 data + config | chunking, embedding, index, queries |
| Embedding | pick `embeddingModelArn`; KB embeds for you | you embed and write vectors |
| Retrieval | `retrieve` / `retrieve_and_generate` | your own `knn` query |
| Best when | you want managed chunk→embed→retrieve | you need full control of the index |

**Recommendation:** use **Knowledge Bases** unless you have a reason to own the
index — it removes the chunking/embedding/retrieval plumbing and integrates
directly with Bedrock models. Under the hood, a Knowledge Base can create and
manage an **OpenSearch Serverless** vector store for you.

---

## Code change (managed path)

```python
# before — Azure AI Search
from azure.search.documents import SearchClient
client = SearchClient(endpoint=..., index_name="docs-index", credential=AzureKeyCredential(key))
results = client.search(search_text=query, top=3)

# after — Bedrock Knowledge Base
import boto3
client = boto3.client("bedrock-agent-runtime", region_name="us-east-1")
resp = client.retrieve(
    knowledgeBaseId=KB_ID,
    retrievalQuery={"text": query},
    retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": 3}},
)
```

- Use **`retrieve`** for just the chunks (drop-in for an existing search call).
- Use **`retrieve_and_generate`** to get retrieval + a grounded answer in one
  call — replaces a search-then-prompt round trip.
- The KB embeds the query for you, so there's no separate embedding step.

---

## The migration, end to end

1. **Land your data in S3** (the KB data source). Re-export documents from
   whatever Azure AI Search ingested.
2. **Create the Knowledge Base:** choose an embedding model
   (`amazon.titan-embed-text-v2:0`) and a vector store (let Bedrock create
   OpenSearch Serverless, or bring your own). See
   [`ai-search-to-opensearch.md`](ai-search-to-opensearch.md).
3. **Ingest** — Bedrock chunks + embeds + indexes automatically.
4. **Swap the query** to `retrieve` / `retrieve_and_generate` by KB ID.
5. **Mind the dimensions** — the index dimension must match the embedding model
   (Titan V2 = 1024). This is the same re-embed concern as
   [../02_embeddings/dimension-mapping.md](../02_embeddings/dimension-mapping.md).

---

## Verify with `az2br`

```bash
az2br scan migration/05_data_plane/before
```

Flags the `SearchClient` / `azure.search.documents` usage, pointing here.
