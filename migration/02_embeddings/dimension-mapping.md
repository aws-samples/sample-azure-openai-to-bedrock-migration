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

Key point: **none of the common Bedrock choices match ada-002's 1536 as a
drop-in you'd want.** Titan G1 is 1536 but is the older model; Titan V2 (1024)
is the current recommendation. Either way you re-embed, so migrate to the model
you actually want, not the one with the matching number.

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
