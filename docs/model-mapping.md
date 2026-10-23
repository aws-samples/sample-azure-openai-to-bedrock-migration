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

IDs below are the `bedrock-runtime` IDs (Path A `/openai/v1` and Path B
Converse). Context / max output are from the Bedrock model cards.

| On Azure (a deployment of) | Bedrock model ID | Notes |
|---|---|---|
| Most capable GPT | `us.openai.gpt-6-astra` | GPT-6 Astra (Sep 2026). 1.05M context / 128K output. Hardest reasoning, coding, computer use, research. |
| Frontier, cost-balanced | `us.openai.gpt-6.1-sol` | GPT-6.1 Sol (Sep 2026). Approaches Astra on several evals at Sol pricing. 1M context / 131,072 output. |
| Coding / agentic GPT | `us.openai.gpt-6-sol` | GPT-6 Sol (Sep 2026). 1.05M / 128K. Features, debugging, refactoring, multi-step tool work. |
| Fast / low-cost GPT | `us.openai.gpt-6-luna` | GPT-6 Luna (Sep 2026). 1.05M / 128K. Summarization, extraction, classification, routing. |
| Prior-gen frontier | `us.openai.gpt-5.6-sol` | GPT-5.6 Sol (Jul 2026). 1.05M / 128K. |
| Prior-gen balanced | `us.openai.gpt-5.6-terra` | GPT-5.5-level quality at half the cost. Default model in this repo's harness. |
| Prior-gen fast | `us.openai.gpt-5.6-luna` | High-volume, latency-sensitive. |
| Older GPT | `us.openai.gpt-5.5`, `us.openai.gpt-5.4` | 1.05M context shared between input and output / 128K output. |
| `gpt-oss` 120B (open-weight) | `openai.gpt-oss-120b-1:0` | 128K context / 16K output. Invoke / Converse / Chat Completions. In-Region ID (no prefix). |
| `gpt-oss` 20B (open-weight) | `openai.gpt-oss-20b-1:0` | 128K / 16K. Lower latency/cost. In-Region ID. |
| Content-moderation classifier | `openai.gpt-oss-safeguard-120b`, `openai.gpt-oss-safeguard-20b` | Open-weight safety models for policy-based moderation. 128K / 16K. In-Region ID. |

Two cybersecurity-specialized variants (Daybreak Red: GPT-5.6 Cyber, Daybreak
Blue: GPT-5.6 Sol) also exist but require Trusted Access for Cyber enrollment;
they aren't general migration targets.

**Cross-Region inference profiles.** On `bedrock-runtime`, the GPT-5.x and
GPT-6.x models are **inference-profile-only** — the bare `openai.gpt-...` ID
isn't accepted for in-Region calls. Prefix it:

- **US Geo:** `us.` → `us.openai.gpt-6-sol`
- **Global:** `global.` → `global.openai.gpt-6-sol`
- **India Geo** (GPT-5.6 Terra only, per its model card): `in.openai.gpt-5.6-terra`

Use the profile ID as the `model` / `modelId` value exactly like a model ID.
Global CRIS is priced at OpenAI first-party rates; US CRIS (and in-Region
mantle) carries a ~10% premium per the GPT-6.1 Sol card, so prefer `global.`
unless you need US data residency.

**On `bedrock-mantle` (Path C, Responses API) the IDs differ.** Mantle is
in-Region only, so use the **bare** ID with no `us.`/`global.` prefix —
`openai.gpt-6-sol`, `openai.gpt-5.6-terra` — on the `/openai/v1` base path.
The open-weight models use different IDs and the plain `/v1` path on mantle:
`openai.gpt-oss-120b`, `openai.gpt-oss-20b` (no `-1:0` suffix). The mantle
Region list is narrower than CRIS (GPT-6.x: `us-east-1`; Astra also
`us-west-2`), so check the model card.

### Reasoning models: replacing temperature/top_p

The frontier GPT models (`gpt-5.x`, `gpt-6.x`) are **reasoning models**: they
accept only the *default* `temperature` and reject `top_p` outright — passing
a non-default `temperature` or any `top_p` value is an error, not a silent
no-op. If you're modernizing from an older Azure GPT deployment that tuned
these knobs, that tuning has to move to two different parameters:

| Replaces | Param | Values | Effect |
|---|---|---|---|
| `temperature` / `top_p` | `reasoning_effort` | `none`, `low`, `medium`, `high`, `xhigh`, `max` | How much the model reasons before answering. Lower = faster/cheaper, higher = more deliberate on hard, multi-step tasks. Default: `medium`. |
| — (new control) | `verbosity` | `low`, `medium`, `high` | How expansive the final reply is, independent of reasoning effort. |

The `reasoning_effort` enum and `medium` default are confirmed on the GPT-6
Sol and GPT-6 Luna model cards. The open-weight `gpt-oss` and `gpt-oss-safeguard`
models also reason by default, but their effort scale is only `low` /
`medium` / `high`.

Both are **top-level fields** on Chat Completions and the Responses API —
same shape on Azure OpenAI's newer models and on Bedrock's OpenAI-compatible
endpoint (Path A):

```python
client.chat.completions.create(
    model="us.openai.gpt-5.6-terra",
    messages=[{"role": "user", "content": "Summarize this incident report."}],
    reasoning_effort="low",   # not temperature
    verbosity="low",
)
```

On **Converse (Path B)**, neither field has a slot in `inferenceConfig` — they
aren't part of the base inference config Converse normalizes across every
provider — so they travel through `additionalModelRequestFields`. Note the
**shape differs from Path A**: Converse rejects the flat `reasoning_effort` /
`verbosity` names (`ValidationException: unknown_parameter`) and instead takes
the nested structured form — `reasoning.effort` and `text.verbosity`:

```python
client.converse(
    modelId="us.openai.gpt-5.6-terra",
    messages=[{"role": "user", "content": [{"text": "Summarize this incident report."}]}],
    inferenceConfig={"maxTokens": 512},
    additionalModelRequestFields={
        "reasoning": {"effort": "low"},   # NOT {"reasoning_effort": "low"}
        "text": {"verbosity": "low"},     # NOT {"verbosity": "low"}
    },
)
```

**Other GPT-6.x Converse caveats** (from the GPT-6 Astra and GPT-6.1 Sol model
cards):

- Converse accepts only `text` and `image` content blocks in user messages.
  A `document` block, or an `image` inside a `toolResult`, is rejected with
  `ValidationException`. Send documents through the Responses API as an
  `input_file` part instead.
- Converse gets **implicit** prompt caching only; the native `cachePoint`
  block isn't supported for explicit caching. Use Responses, Chat Completions,
  or InvokeModel for explicit cache breakpoints.
- JSON Schema output on Converse needs `outputConfig.textFormat` **and**
  `additionalModelRequestFields={"text": {"format": {"strict": true}}}`.
- Reasoning summaries on Converse go in
  `additionalModelRequestFields.reasoning.summary`, next to `reasoning.effort`.

The harness implements this mapping in
[`../harness/providers.py`](../harness/providers.py) —
`_drop_unsupported_sampling` drops `temperature`/`top_p` for frontier models
with a warning, and `BedrockConverse._to_additional_fields` reshapes
`reasoning_effort`/`verbosity` into the nested `reasoning.effort` /
`text.verbosity` form inside `additionalModelRequestFields`. See
[`../harness/test_providers.py`](../harness/test_providers.py) for unit tests
and `smoke_test_after.py`'s `check_reasoning_effort_verbosity` for a live
check against both API paths.

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
| `text-embedding-ada-002` / `-3-small` | 1536 | Cohere Embed v4 (`cohere.embed-v4`) | configurable 256–1536 | Yes (different model), but the index dimension can stay 1536 |
| `text-embedding-3-small` | 1536 | Amazon Titan Text Embeddings V2 | 1024 | Yes |
| `text-embedding-3-large` | 3072 | Amazon Nova Multimodal Embeddings (`amazon.nova-2-multimodal-embeddings-v1:0`) | 3072, 1024, 384, 256 | Yes (different model), but the index dimension can stay 3072 |
| `text-embedding-3-large` | 3072 | Amazon Titan Text Embeddings V2 | 1024 | Yes |

Notes:
- **Titan V2** takes up to 8,192 tokens / 50,000 characters and supports
  configurable output dimensions (1024 / 512 / 256). It does **not** accept
  `maxTokenCount` or `topP` inference parameters.
- **Cohere Embed v4** is multimodal (text + images, interleaved), takes up to
  ~128K tokens per document, and outputs float / int8 / uint8 / binary /
  ubinary embeddings.
- **Nova Multimodal Embeddings** puts text, image, document image, video, and
  audio in one vector space. Max input is 8K tokens or 30s of video/audio. It
  has both sync and async (`StartAsyncInvoke`) APIs, and per its model card it
  is in-Region only in `us-east-1` (and GovCloud) today.
- A matching dimension saves you from resizing the index, **not from
  re-embedding**. Vectors from different models aren't comparable, so every
  row above means a full re-embed.

---

## How the ID is used

**Path A — OpenAI-compatible endpoint** (`model=`):

```python
client.chat.completions.create(model="openai.gpt-oss-120b-1:0", messages=[...])
```

**Path B — Converse** (`modelId=`):

```python
client.converse(modelId="us.openai.gpt-6-sol", messages=[...])
```

**Path C — Responses API on `bedrock-mantle`** (bare ID, no CRIS prefix):

```python
client.responses.create(model="openai.gpt-6-sol", input="...")
```

**Embeddings** (Invoke, or via a Knowledge Base's `embeddingModelArn`):

```python
client.invoke_model(modelId="amazon.titan-embed-text-v2:0", body=...)
```

See also: [api-mapping.md](api-mapping.md) for the API surface, and
[tool-mapping.md](tool-mapping.md) for server tools.
