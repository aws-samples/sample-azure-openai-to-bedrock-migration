# Model Mapping — Azure OpenAI → Amazon Bedrock

Azure uses **deployment names** (a label you assign to a model in your Azure
resource). Bedrock uses **model IDs** and **inference profile IDs**. When you
migrate, you replace the deployment name you pass as `model` (OpenAI-compatible
endpoint) or `modelId` (Converse) with a Bedrock ID from the tables below.

> Model availability is **per Region** and some models require **requesting
> access** first. Confirm the ID is enabled in your Region before relying on it.
> IDs and the model lineup change over time — treat this table as a starting
> point and check the Amazon Bedrock console for the current list.

---

## Chat / completion models

You have two families to choose from on Bedrock:

1. **Keep GPT.** AWS hosts OpenAI models natively on Bedrock. This is the
   lowest-friction move — same model family, new home.
2. **Switch family** (Anthropic Claude, Amazon Nova, Meta Llama, Mistral, etc.)
   if you want to evaluate alternatives. The Converse API makes this a config
   change. Family-switch IDs are out of scope here; see the Bedrock console.

### OpenAI models on Bedrock (keep GPT)

| On Azure (a deployment of) | Bedrock model ID | Notes |
|---|---|---|
| `gpt-oss` 120B (open-weight) | `openai.gpt-oss-120b-1:0` | 128K context. Invoke / Converse / Chat Completions. |
| `gpt-oss` 20B (open-weight) | `openai.gpt-oss-20b-1:0` | 128K context, 16K max output. Lower latency/cost. |
| Latest GPT (frontier) | `us.openai.gpt-5.6-sol` | Inference-profile-only — use the `us.` (or `global.`) prefix; see cross-Region note below. |
| Balanced GPT | `us.openai.gpt-5.6-terra` | Everyday production workloads. Inference-profile id. |
| Fast / low-cost GPT | `us.openai.gpt-5.6-luna` | High-volume, latency-sensitive. Inference-profile id. |
| Prior-gen GPT | `us.openai.gpt-5.5`, `us.openai.gpt-5.4` | Also available (inference-profile ids). |

**Cross-Region inference profiles.** The GPT-5.x models are typically invoked
through an inference profile ID rather than the bare model ID, which routes for
throughput and availability:

- **US Geo:** prefix with `us.` → `us.openai.gpt-5.6-terra`
- **Global:** prefix with `global.` → `global.openai.gpt-5.6-terra`

Use the profile ID as the `model` / `modelId` value exactly like a model ID.

---

## Embedding models

Azure's `text-embedding-*` deployments map to a Bedrock embedding model. **The
output dimension is the thing that matters** — if it changes, you must re-embed
your existing vectors (a stored index can't mix dimensions). See
[../migration/02_embeddings/dimension-mapping.md](../migration/02_embeddings/dimension-mapping.md).

| On Azure | Dimensions | Bedrock option | Bedrock dims | Re-embed needed? |
|---|---|---|---|---|
| `text-embedding-ada-002` | 1536 | Amazon Titan Text Embeddings V2 (`amazon.titan-embed-text-v2:0`) | 1024 (default), 512, 256 | **Yes** — dims differ |
| `text-embedding-ada-002` | 1536 | Amazon Titan Text Embeddings G1 (`amazon.titan-embed-text-v1`) | 1536 | Yes (different model) |
| `text-embedding-3-small` | 1536 | Amazon Titan Text Embeddings V2 | 1024 | Yes |
| `text-embedding-3-large` | 3072 | Amazon Titan Text Embeddings V2 | 1024 | Yes |

Notes:
- **Titan V2** takes up to 8,192 tokens / 50,000 characters and supports
  configurable output dimensions (1024 / 512 / 256). It does **not** accept
  `maxTokenCount` or `topP` inference parameters.
- There is no drop-in 1536-dim Titan V2 setting; matching ada-002's 1536
  exactly means Titan **G1** is still a different model, so plan to re-index.

---

## How the ID is used

**Path A — OpenAI-compatible endpoint** (`model=`):

```python
client.chat.completions.create(model="openai.gpt-oss-120b-1:0", messages=[...])
```

**Path B — Converse** (`modelId=`):

```python
client.converse(modelId="us.openai.gpt-5.6-terra", messages=[...])
```

**Embeddings** (Invoke, or via a Knowledge Base's `embeddingModelArn`):

```python
client.invoke_model(modelId="amazon.titan-embed-text-v2:0", body=...)
```

See also: [api-mapping.md](api-mapping.md) for the API surface, and
[tool-mapping.md](tool-mapping.md) for server tools.
