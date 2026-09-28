# Azure AI Search → Amazon OpenSearch / Bedrock Knowledge Bases

Detail companion to the [section README](README.md): how the index and setup
map, and the dimension rule that catches people.

---

## Concept mapping

| Azure AI Search | Bedrock Knowledge Bases | Amazon OpenSearch (DIY) |
|---|---|---|
| Search service | Knowledge Base | OpenSearch Serverless collection / domain |
| Index | KB (managed vector store) | vector index |
| Indexer / data source | KB data source (S3) | your ingestion pipeline |
| Skillset (chunk/embed) | managed by the KB | you build it |
| Vector field | managed | `knn_vector` field |
| Query (`search`) | `retrieve` / `retrieve_and_generate` | `knn` query |
| API key | AWS IAM | AWS IAM |

---

## Vector store options for a Knowledge Base

A Knowledge Base needs a vector store. Bedrock can **create and manage an
OpenSearch Serverless** store for you, or you can bring your own:

- **Amazon OpenSearch Serverless** (default; Bedrock can auto-create it)
- **Amazon Aurora PostgreSQL** (pgvector)
- **Pinecone**
- **Redis Enterprise Cloud**
- **MongoDB Atlas**

Data source: **Amazon S3** (drop your documents in a bucket; the KB ingests
them).

---

## Setup outline (managed KB)

1. **S3 bucket** with your source documents.
2. **Embedding model** — set `embeddingModelArn`, e.g.
   `arn:aws:bedrock:{region}::foundation-model/amazon.titan-embed-text-v2:0`.
3. **Vector store** — let Bedrock create OpenSearch Serverless, or point at your
   own collection ARN + index.
4. **Data source** — `S3` pointing at the bucket.
5. **Ingest** — start an ingestion job; Bedrock chunks, embeds, and indexes.
6. **Query** — `bedrock-agent-runtime.retrieve(...)` by KB ID (see
   [`after/bedrock_knowledge_base.py`](after/bedrock_knowledge_base.py)).

---

## The dimension rule (don't skip)

The vector index dimension **must match** the embedding model:

| Embedding model | Index dimension |
|---|---|
| Titan Text Embeddings V2 (`amazon.titan-embed-text-v2:0`) | 1024 (default), 512, or 256 |
| Titan Text Embeddings G1 (`amazon.titan-embed-text-v1`) | 1536 |

If you're coming from ada-002 (1536), the vectors do **not** carry over — the
Knowledge Base re-embeds your S3 data with the chosen model at its dimension.
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

Otherwise, the managed Knowledge Base is less code and less to operate.
