# 01 — Chat Completions: Azure OpenAI → Amazon Bedrock

This is the most common migration and the best place to start. Your Azure
OpenAI chat calls move to Amazon Bedrock with **either a near-zero code change
or a clean, portable rewrite** — you choose based on how much you want to
invest now versus later.

- **Before:** [`before/azure_chat.py`](before/azure_chat.py)
- **After, Path A:** [`after/bedrock_openai_endpoint.py`](after/bedrock_openai_endpoint.py)
- **After, Path B:** [`after/bedrock_converse.py`](after/bedrock_converse.py)
- **After, Path C:** [`after/bedrock_responses_api.py`](after/bedrock_responses_api.py)

---

## Three paths

There's no single "right" path — it depends on **what API you're on today** and
**whether you plan to stay on OpenAI models**. Any of the three is valid.

| Path | Use when | API surface |
|---|---|---|
| **A — OpenAI-compatible endpoint** | On Chat Completions today, want smallest diff | Chat Completions (OpenAI SDK) |
| **B — Converse** | Want a model-agnostic interface, may evaluate non-OpenAI models later | Bedrock Converse |
| **C — Responses API (mantle)** | On Responses/Assistants today, want to keep that surface, or need server-side tools (Web Search, code interpreter) | Responses (OpenAI SDK) |

Two questions decide it:

- **What API are you on today?** On Chat Completions → Path A is the smallest
  move. On the Responses API or Azure Assistants → Path C keeps your call shape.
- **Do you plan to stay on OpenAI models?** If you may evaluate non-OpenAI
  models (Claude, Nova, Llama) later, Path B gives you one API across the whole
  Bedrock catalog so a model swap is config, not a rewrite.

**Recommendation:** match the path to where you are — start with **Path A** for
the smallest Chat Completions cutover, choose **Path C** if you're already on
the Responses API (or need mantle-only server-side tools), and adopt **Path B**
where you want provider-independence across the full model catalog.

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

## Path C — the Responses API on `bedrock-mantle`

If you're already calling `client.responses.create(...)` (or you're on Azure
OpenAI Assistants, which maps conceptually to the Responses API), you can keep
that surface. Point the OpenAI SDK at the Bedrock **mantle** endpoint. This is
also the path to reach **server-side tools** — Web Search, code interpreter —
which are mantle-only (the Responses API on `bedrock-runtime` does not expose
them).

```python
from aws_bedrock_token_generator import provide_token
from openai import OpenAI

region = "us-east-1"
client = OpenAI(
    base_url=f"https://bedrock-mantle.{region}.api.aws/openai/v1",
    api_key=provide_token(region=region),
)

response = client.responses.create(
    model="openai.gpt-5.6-terra",           # bare model id on mantle
    input="What is Amazon Bedrock?",
)
print(response.output_text)
```

Add the built-in Web Search tool with one tool object — near drop-in from Azure:

```python
response = client.responses.create(
    model="openai.gpt-5.6-terra",
    input="What's the latest AWS Lambda cold start guidance?",
    tools=[{
        "type": "web_search",
        "search_context_size": "low",
        "external_web_access": False,  # keep retrieval inside the AWS boundary
    }],
)
```

Full runnable example: [`after/bedrock_responses_api.py`](after/bedrock_responses_api.py).

```bash
pip install openai aws-bedrock-token-generator
```

> Mantle authorizes inference with the `bedrock-mantle:CreateInference` IAM
> action **plus** `bedrock-mantle:CallWithBearerToken` when you authenticate
> with a Bedrock API key (the OpenAI SDK path here — granting only
> `CreateInference` fails 403). Web Search adds `bedrock-websearch:InvokeSearch`
> + `bedrock-websearch:InvokeFetch`. Leaving `external_web_access` at its
> default of `True` without `bedrock-websearch:ExternalWebAccess` makes live
> Fetch fail silently (the call still returns 200 using Search-only results —
> the failed Fetch shows up only as a `web_search_call` with `status: "failed"`
> in the response body) — set it to `False` to stay inside the AWS boundary.
> Also note: mantle takes the **bare** model id (`openai.gpt-5.6-terra`); the
> `us.` inference-profile id is a `bedrock-runtime` concept and 404s on mantle.
> See [`../03_tool_use/server-tools.md`](../03_tool_use/server-tools.md).

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
