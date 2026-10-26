# Azure AI Search → Bedrock Knowledge Bases / Amazon OpenSearch

Detail companion to the [section README](README.md): how the index and setup
map, and the dimension rule that catches people.

---

## Concept mapping

| Azure AI Search | Bedrock Managed KB | Customer-managed KB | Amazon OpenSearch (DIY) |
|---|---|---|---|
| Search service | Managed Knowledge Base | Knowledge Base + your vector store | OpenSearch Serverless collection / domain |
| Index | managed (no store to provision) | vector store you choose | vector index |
| Indexer / data source | data source (S3, SaaS connectors) | data source (S3, custom) | your ingestion pipeline |
| Skillset (chunk/embed) | managed parsing, chunking, embedding | managed chunking; you pick the embedding model | you build it |
| Vector field | managed | managed | `knn_vector` field |
| Query (`search`) | `retrieve` / `retrieve_and_generate` / agentic retrieval | `retrieve` / `retrieve_and_generate` | `knn` query |
| API key | AWS IAM | AWS IAM | AWS IAM |

---

## Managed Knowledge Base setup (recommended)

A Managed KB has no `storageConfiguration` and, by default, no embedding model to
choose:

```python
import boto3

agent = boto3.client("bedrock-agent", region_name="us-east-1")

kb = agent.create_knowledge_base(
    name="docs-kb",
    roleArn="arn:aws:iam::123456789012:role/BedrockKBRole",
    knowledgeBaseConfiguration={
        "type": "MANAGED",
        "managedKnowledgeBaseConfiguration": {"embeddingModelType": "MANAGED"},
    },
)["knowledgeBase"]["knowledgeBaseId"]
```

Then create a data source and start ingestion (`create_data_source`,
`start_ingestion_job`). S3 and the SaaS connectors use the
`MANAGED_KNOWLEDGE_BASE_CONNECTOR` data source type; see the
[Bedrock User Guide](https://docs.aws.amazon.com/bedrock/latest/userguide/kb-managed-create.html)
for the exact connector configuration.

To use your own embedding model instead, set `embeddingModelType` to `CUSTOM` and
pass `embeddingModelArn` plus `embeddingModelConfiguration` with `dimensions: 1024`
and float32. Supported: Titan Text Embeddings V2, Cohere Embed English/Multilingual
v3 and v4, and Nova Multimodal Embeddings, plus TwelveLabs Marengo Embed 3.0 as
the multimodal (image, audio, video) option. Two caveats: you can't change the embedding type after
creation, and a custom embedding model disables the managed reranker.

Query with `retrieve` and `managedSearchConfiguration` (see the
[section README](README.md)).

---

## Vector store options for a customer-managed Knowledge Base

A customer-managed Knowledge Base needs a vector store. Bedrock can **create and
manage an OpenSearch Serverless** store for you, or you can bring your own:

- **Amazon OpenSearch Serverless** (default; Bedrock can auto-create it)
- **Amazon Aurora PostgreSQL** (pgvector)
- **Pinecone**
- **Redis Enterprise Cloud**
- **MongoDB Atlas**

(Other stores, such as Amazon S3 Vectors, are also supported; check the Bedrock
User Guide for the current list.) Data source: **Amazon S3** (drop your documents
in a bucket; the KB ingests them).

---

## Setup outline (customer-managed KB)

1. **S3 bucket** with your source documents.
2. **Embedding model** — set `embeddingModelArn`, e.g.
   `arn:aws:bedrock:{region}::foundation-model/amazon.titan-embed-text-v2:0`.
3. **Vector store** — let Bedrock create OpenSearch Serverless, or point at your
   own collection ARN + index.
4. **Data source** — `S3` pointing at the bucket.
5. **Ingest** — start an ingestion job; Bedrock chunks, embeds, and indexes.
6. **Query** — `bedrock-agent-runtime.retrieve(...)` by KB ID with
   `vectorSearchConfiguration` (see
   [`after/bedrock_knowledge_base.py`](after/bedrock_knowledge_base.py)).

---

## The dimension rule (don't skip)

For a customer-managed KB or your own OpenSearch index, the vector index
dimension **must match** the embedding model (a Managed KB handles this for you):

| Embedding model | Index dimension |
|---|---|
| Titan Text Embeddings V2 (`amazon.titan-embed-text-v2:0`) | 1024 (default), 512, or 256 |
| Titan Text Embeddings G1 (`amazon.titan-embed-text-v1`) | 1536 |

If you're coming from ada-002 (1536), the vectors do **not** carry over — the
Knowledge Base re-embeds your S3 data with its model at that model's dimension.
This is the same re-embed concern covered in
[../02_embeddings/dimension-mapping.md](../02_embeddings/dimension-mapping.md).
When bringing your own OpenSearch index, create the `knn_vector` field with the
matching dimension and the `faiss` engine.

---

## When to go DIY OpenSearch instead

Choose a self-managed OpenSearch index when you need:

- custom chunking / hybrid (lexical + vector) search logic,
- fields, filters, or scoring the managed KB doesn't expose,
- an index shared with non-Bedrock workloads.

Otherwise, a Managed Knowledge Base is less code and less to operate.
