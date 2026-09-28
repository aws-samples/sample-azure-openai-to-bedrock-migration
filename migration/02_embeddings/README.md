# 02 — Embeddings: Azure OpenAI → Amazon Bedrock

Move your embedding workload from Azure OpenAI (`text-embedding-ada-002` and
friends) to Amazon Bedrock. The code change is small; the **operational** change
— re-embedding your corpus because the vector dimension changes — is the part
that needs planning.

- **Before:** [`before/azure_embed.py`](before/azure_embed.py) — ada-002, 1536-dim
- **After:** [`after/bedrock_titan_embed.py`](after/bedrock_titan_embed.py) — Titan V2, 1024-dim
- **The gotcha:** [`dimension-mapping.md`](dimension-mapping.md)

---

## The one thing to get right: dimensions

| | Azure ada-002 | Bedrock Titan V2 |
|---|---|---|
| Model | `text-embedding-ada-002` | `amazon.titan-embed-text-v2:0` |
| Dimensions | 1536 | 1024 (default), 512, 256 |
| Max input | 8,191 tokens | 8,192 tokens / 50,000 chars |

Because the dimension changes, you **cannot** swap models against an existing
index. You must **re-embed the whole corpus and rebuild the index** at the new
dimension. Full playbook in [`dimension-mapping.md`](dimension-mapping.md).

---

## Code change

**Before (Azure):**

```python
from openai import AzureOpenAI
client = AzureOpenAI(azure_endpoint=..., api_key=..., api_version="2024-10-21")
resp = client.embeddings.create(model="text-embedding-ada-002", input=texts)
vectors = [d.embedding for d in resp.data]   # 1536-dim
```

**After (Bedrock Titan V2):**

```python
import boto3, json
client = boto3.client("bedrock-runtime", region_name="us-east-1")

def embed_one(text):
    body = json.dumps({"inputText": text, "dimensions": 1024, "normalize": True})
    resp = client.invoke_model(modelId="amazon.titan-embed-text-v2:0", body=body)
    return json.loads(resp["body"].read())["embedding"]   # 1024-dim
```

Two differences beyond auth:

- **Batching:** Titan V2 embeds one `inputText` per `invoke_model` call, so loop
  over your batch (the example does this). ada-002 accepts a list in one call.
- **Parameters:** Titan V2 supports `dimensions` (1024/512/256) and `normalize`.
  It does **not** accept `maxTokenCount` or `topP`.

### Prefer to keep the OpenAI SDK?

The Bedrock OpenAI-compatible endpoint also serves embedding models, so a
Path-A-style `client.embeddings.create(...)` works too. Use boto3 (above) when
you want the native `dimensions`/`normalize` controls and standard AWS auth.

---

## Titan V2 dimension options

| Target | Dims | When |
|---|---|---|
| Titan V2 (`amazon.titan-embed-text-v2:0`) | 1024/512/256 | Default; configurable size, long input. |
| Titan G1 (`amazon.titan-embed-text-v1`) | 1536 | Only if you specifically need 1536. |

All require re-embedding vs. ada-002. See
[../../docs/model-mapping.md](../../docs/model-mapping.md).

---

## Verify with `az2br`

```bash
az2br scan migration/02_embeddings/before
```

Flags the `embeddings.create` call and the `text-embedding-ada-002` reference,
each pointing back here.
