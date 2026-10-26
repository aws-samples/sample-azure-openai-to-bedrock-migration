# Dimension Mapping — the embedding gotcha

The single biggest embedding-migration risk is **dimension mismatch**. A vector
index is built for one fixed dimension; you cannot store 1024-dim and 1536-dim
vectors in the same index, and you cannot compare vectors produced by different
models. So switching embedding models is a **re-embed + re-index**, not a swap.

## Dimensions at a glance

| Model | Dimensions | Context |
|---|---|---|
| Azure `text-embedding-ada-002` | 1536 | 8,191 tokens |
| Azure `text-embedding-3-small` | 1536 | 8,191 tokens |
| Azure `text-embedding-3-large` | 3072 | 8,191 tokens |
| Amazon Titan Text Embeddings **V2** (`amazon.titan-embed-text-v2:0`) | **1024** (default), 512, 256 | 8,192 tokens / 50,000 chars |
| Amazon Titan Text Embeddings **G1** (`amazon.titan-embed-text-v1`) | 1536 | 8,192 tokens |
| Cohere Embed v4 (`cohere.embed-v4`) | configurable 256–1536 | ~128K tokens |
| Amazon Nova Multimodal Embeddings (`amazon.nova-2-multimodal-embeddings-v1:0`) | 3072, 1024, 384, 256 | 8K tokens |

Key point: **a matching dimension is never a drop-in.** Titan G1 and Cohere
Embed v4 can output 1536 (ada-002 / 3-small), and Nova Multimodal Embeddings
can output 3072 (3-large). That lets you keep the index *size*, but the
vectors come from a different model, so you still re-embed everything. Pick
the model you actually want, not the one with the matching number.

> **Bedrock Managed Knowledge Base:** the managed embedding model needs no
> dimension planning. A custom embedding model on a managed KB must be
> 1024-dimension float32 and can't be changed after the KB is created. The
> dimension rules in this guide apply in full to customer-managed KBs and to
> your own OpenSearch index.

## Migration playbook

1. **Pick the target model + dimension** and fix it for the whole index. Titan
   V2 at 1024 is a good default; 512 or 256 trade recall for storage/speed.
2. **Rebuild the index at the new dimension.** Create a fresh vector index (new
   OpenSearch index, new Knowledge Base, etc.) sized to the target dimension —
   don't try to migrate vectors in place.
3. **Re-embed the whole corpus** with the target model and write to the new
   index. Embeddings from different models are not comparable, so partial
   migration produces wrong search results.
4. **Re-embed queries with the same model** at request time. The query vector
   must come from the same model/dimension as the index.
5. **Cut over** reads to the new index, then retire the old one.

## Shrinking Titan V2 output (optional)

Titan V2 lets you request 1024, 512, or 256 dimensions via the `dimensions`
body parameter. Smaller vectors mean cheaper storage and faster search at some
recall cost. Choose once — the index is built for that size:

```python
body = json.dumps({"inputText": text, "dimensions": 512, "normalize": True})
```

## Why not "just pad/truncate to 1536"?

Don't. Padding ada-002 vectors to match, or truncating Titan vectors, produces
vectors that don't live in the same space — similarity scores become
meaningless. Re-embedding is the only correct path.

See the runnable example in [`before/`](before/) and [`after/`](after/), and
the model table in [../../docs/model-mapping.md](../../docs/model-mapping.md).
