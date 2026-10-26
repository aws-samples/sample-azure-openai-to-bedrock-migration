# 05 — Data Plane: Azure AI Search → Bedrock Knowledge Bases / Amazon OpenSearch

Your vector index and retrieval layer move to AWS. The target depends on how much
you want managed for you.

- **Before:** [`before/azure_search.py`](before/azure_search.py) — Azure AI Search query
- **After:** [`after/bedrock_knowledge_base.py`](after/bedrock_knowledge_base.py) — Bedrock Knowledge Base `retrieve` (Managed or customer-managed)
- **Index & setup details:** [`ai-search-to-opensearch.md`](ai-search-to-opensearch.md)

---

## Three targets

Amazon Bedrock Knowledge Bases comes in two types, plus the DIY OpenSearch route.
AWS recommends the **Managed** type for new work.

| | **Bedrock Managed Knowledge Base** (recommended) | **Customer-managed Knowledge Base** | **Amazon OpenSearch** (DIY) |
|---|---|---|---|
| You manage | S3/SaaS data sources + config | S3 data + vector store + embedding model | chunking, embedding, index, queries |
| Embedding | service-managed by default; optional custom (1024-dim float32) | you pick `embeddingModelArn` | you embed and write vectors |
| Vector store | managed, auto-scaling | you choose and provision | you run it |
| Retrieval | `retrieve` (hybrid search + managed reranker), `retrieve_and_generate`, **agentic retrieval** | `retrieve` / `retrieve_and_generate` | your own `knn` query |
| Connectors | S3, SharePoint, Confluence, Google Drive, OneDrive, Salesforce, ServiceNow, Zendesk, Box, Web Crawler, custom | S3 and custom (new Confluence/SharePoint/Salesforce/Web Crawler connectors can't be created since Sept 30, 2026) | your ingestion pipeline |
| Best when | you want managed parse→chunk→embed→retrieve | you need a specific vector DB or embedding setup | you need full control of the index |

**Recommendation:** use a **Managed Knowledge Base** unless you have a reason to
own the vector store, embedding model, or index. Choose **customer-managed** for
a specific vector store or embedding configuration, and **OpenSearch** for full
control. A customer-managed Knowledge Base can still create and manage an
**OpenSearch Serverless** vector store for you.

> Managed Knowledge Bases are available in a subset of Regions (at launch:
> us-east-1, us-west-2, eu-west-1, eu-central-1, eu-west-2, ap-southeast-2,
> ap-northeast-1, and GovCloud us-gov-west-1). Check the Bedrock User Guide for the
> current list. The embedding model type can't be changed after creation.

---

## Code change

```python
# before — Azure AI Search
from azure.search.documents import SearchClient
client = SearchClient(endpoint=..., index_name="docs-index", credential=AzureKeyCredential(key))
results = client.search(search_text=query, top=3)

# after — Bedrock Managed Knowledge Base
import boto3
client = boto3.client("bedrock-agent-runtime", region_name="us-east-1")
resp = client.retrieve(
    knowledgeBaseId=KB_ID,
    retrievalQuery={"text": query},
    retrievalConfiguration={"managedSearchConfiguration": {"numberOfResults": 3}},
)
```

- The config key depends on the Knowledge Base type: `managedSearchConfiguration`
  for a Managed KB, `vectorSearchConfiguration` for a customer-managed KB. Using
  the wrong one is a request error. `after/bedrock_knowledge_base.py` handles both
  via `BEDROCK_KB_TYPE`.
- Use **`retrieve`** for just the chunks (drop-in for an existing search call).
- Use **`retrieve_and_generate`** to get retrieval + a grounded answer in one
  call — replaces a search-then-prompt round trip.
- Use **agentic retrieval** (`AgenticRetrieveStream`, Managed KBs only) for
  multi-hop questions: a model decomposes the query, searches one or more
  knowledge bases iteratively, and checks that the results are sufficient.
- The KB embeds the query for you, so there's no separate embedding step.

---

## The migration, end to end

1. **Land your data in S3** (or connect a SaaS source on a Managed KB).
   Re-export documents from whatever Azure AI Search ingested.
2. **Create the Knowledge Base.** For a Managed KB, call `create_knowledge_base`
   with `type: "MANAGED"` and `embeddingModelType: "MANAGED"` — no vector store
   or model ARN. For a customer-managed KB, choose an embedding model
   (`amazon.titan-embed-text-v2:0`) and a vector store. See
   [`ai-search-to-opensearch.md`](ai-search-to-opensearch.md).
3. **Ingest** — Bedrock parses, chunks, embeds, and indexes automatically.
4. **Swap the query** to `retrieve` / `retrieve_and_generate` by KB ID.
5. **Mind the dimensions** — on a customer-managed KB the index dimension must
   match the embedding model (Titan V2 = 1024); on a Managed KB the managed model
   handles this and a custom model must be 1024-dim float32. This is the same
   re-embed concern as
   [../02_embeddings/dimension-mapping.md](../02_embeddings/dimension-mapping.md).

---

## Verify with `az2br`

```bash
az2br scan migration/05_data_plane/before
```

Flags the `SearchClient` / `azure.search.documents` usage, pointing here.
