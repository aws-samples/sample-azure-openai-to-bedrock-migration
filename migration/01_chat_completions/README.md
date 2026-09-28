# 01 — Chat Completions: Azure OpenAI → Amazon Bedrock

This is the most common migration and the best place to start. Your Azure
OpenAI chat calls move to Amazon Bedrock with **either a near-zero code change
or a clean, portable rewrite** — you choose based on how much you want to
invest now versus later.

- **Before:** [`before/azure_chat.py`](before/azure_chat.py)
- **After, Path A:** [`after/bedrock_openai_endpoint.py`](after/bedrock_openai_endpoint.py)
- **After, Path B:** [`after/bedrock_converse.py`](after/bedrock_converse.py)

---

## Two paths

| | **Path A — OpenAI-compatible endpoint** | **Path B — Converse API** |
|---|---|---|
| SDK | Keep the `openai` library | `boto3` (`bedrock-runtime`) |
| Diff size | Smallest — swap client init, keep `chat.completions.create(...)` | Larger — different call shape |
| Auth | Bedrock API key (bearer token) | Standard AWS credentials (SigV4) |
| Portability | OpenAI-shaped; tied to that call surface | One shape across **all** Bedrock models |
| Best when | Fast cutover, minimal risk, existing OpenAI code | You want to swap/A-B models later without rewrites |

**Recommendation:** start with **Path A** to get onto Bedrock with the smallest
diff, then adopt **Path B** where you want provider-independence (tool use,
multimodal, and model swaps all share one API on Converse).

> Naming note: Path A uses the Amazon Bedrock **OpenAI-compatible endpoint**
> (also called the **Chat Completions API** on Bedrock). It is served on the
> `/openai/v1` path of the `bedrock-runtime` endpoint.

---

## Path A — keep the OpenAI SDK, point it at Bedrock

Three things change from the Azure version:

1. `AzureOpenAI(...)` → `OpenAI(base_url=..., api_key=...)`
2. Azure endpoint + `api_version` → `https://bedrock-runtime.{region}.amazonaws.com/openai/v1`
3. Azure key + deployment name → a **Bedrock API key** + a **Bedrock model id**

```python
from aws_bedrock_token_generator import provide_token
from openai import OpenAI

region = "us-east-1"
client = OpenAI(
    base_url=f"https://bedrock-runtime.{region}.amazonaws.com/openai/v1",
    api_key=provide_token(region=region),  # short-term key from your AWS creds
)

response = client.chat.completions.create(
    model="openai.gpt-oss-120b-1:0",       # a Bedrock model id, not a deployment name
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.choices[0].message.content)
```

The `.chat.completions.create(...)` call — messages, `temperature`,
`max_tokens`, `stream=True` — is unchanged. That's the whole point of Path A.

```bash
pip install openai aws-bedrock-token-generator
```

### Authentication

Don't store a static key. `provide_token(region=...)` from
[`aws-bedrock-token-generator`](https://pypi.org/project/aws-bedrock-token-generator/)
derives a short-term Bedrock API key (valid up to 12 hours) from whatever AWS
credential chain you already use — env vars, a named profile, or an IAM role.
For a quick local experiment you *can* set a long-lived key in
`AWS_BEARER_TOKEN_BEDROCK`, but prefer the generated token for anything real.

---

## Path B — the Converse API (portable)

Converse is one API (`converse` / `converse_stream`) that works across every
Bedrock model, so later model swaps or A/B tests are a config change. The
call shape differs from OpenAI in four ways:

| OpenAI / Azure | Converse |
|---|---|
| `api_key` / bearer token | standard AWS credentials (SigV4) |
| `system` as a `messages` role | top-level `system=[{"text": ...}]` |
| `content` is a string | `content` is a list of blocks: `[{"text": ...}]` |
| `temperature`, `max_tokens` top-level | inside `inferenceConfig` (`maxTokens`) |
| `resp.choices[0].message.content` | `resp["output"]["message"]["content"][0]["text"]` |

```python
import boto3

client = boto3.client("bedrock-runtime", region_name="us-east-1")

response = client.converse(
    modelId="openai.gpt-oss-120b-1:0",
    system=[{"text": "You are a concise assistant."}],
    messages=[{"role": "user", "content": [{"text": "What is Amazon Bedrock?"}]}],
    inferenceConfig={"temperature": 0.2, "maxTokens": 256},
)
# Join text blocks — a reasoning model may emit a reasoningContent block first.
blocks = response["output"]["message"]["content"]
print("".join(b["text"] for b in blocks if "text" in b))
```

```bash
pip install boto3
```

---

## Model mapping

Azure uses **deployment names** (a label you assign); Bedrock uses **model ids**
or **inference profile ids**. You pass the Bedrock id as the `model` (Path A) or
`modelId` (Path B) value.

| On Azure (deployment of) | On Bedrock (example id) |
|---|---|
| GPT open-weight models | `openai.gpt-oss-120b-1:0`, `openai.gpt-oss-20b-1:0` |
| Latest GPT (US cross-Region) | `us.openai.gpt-5.6-terra` |
| Latest GPT (global cross-Region) | `global.openai.gpt-5.6-terra` |

Regional (`openai.*`) vs. cross-Region inference profile ids (`us.openai.*`,
`global.openai.*`) affect where inference runs and your availability posture.
The full GPT → Bedrock table lives in
[`../../docs/model-mapping.md`](../../docs/model-mapping.md) *(roadmap)*.

---

## Gotchas

- **Deployment name ≠ model id.** The single most common break: passing your
  Azure deployment name as `model`. Use a Bedrock model id (see the table).
- **Drop `api_version`.** It's Azure-specific and not used by Bedrock. `az2br`
  flags it as LOW severity.
- **Model availability is regional.** Not every model id exists in every Region;
  a missing model surfaces as an access/validation error. Confirm the id is
  enabled in your Region (and request model access if needed).
- **Region must match the token.** On Path A, the `region` in `base_url` and in
  `provide_token(region=...)` must be the same, or the token won't validate.
- **Streaming event shape differs on Path B.** Converse streams
  `contentBlockDelta` events, not OpenAI-style `choices[].delta`. The example
  handles this.
- **Reasoning models return multiple content blocks.** On Converse, models like
  the OpenAI `gpt-oss` family emit a `reasoningContent` block *before* the
  `text` block, so `content[0]` may not be the answer. Join every block that
  has a `"text"` key instead of indexing `[0]`. The `after/` example does this.

---

## Verify with `az2br`

Scanning the `before/` example flags exactly what this guide changes:

```bash
az2br scan migration/01_chat_completions/before
```

You'll see HIGH findings for the `AzureOpenAI` import/init and Azure auth, plus
MEDIUM/LOW for the deployment name and `api_version`. Each finding points back
to this guide.
