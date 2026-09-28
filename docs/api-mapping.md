# API Mapping — Azure OpenAI → Amazon Bedrock

Azure OpenAI exposes the OpenAI API surface behind Azure-managed endpoints.
Amazon Bedrock gives you **two ways** to reach the same capabilities:

- **Path A — OpenAI-compatible endpoint** — keep the OpenAI SDK; smallest diff.
- **Path B — AWS SDK (Converse / Invoke)** — idiomatic, portable across all
  Bedrock models.

This page maps each Azure OpenAI call to its Bedrock equivalent on both paths.

---

## Endpoints & auth

| | Azure OpenAI | Bedrock Path A (OpenAI-compatible) | Bedrock Path B (AWS SDK) |
|---|---|---|---|
| Endpoint | `https://<resource>.openai.azure.com` | `https://bedrock-runtime.{region}.amazonaws.com/openai/v1` | `bedrock-runtime` (via boto3) |
| Auth | API key or Azure AD/Entra | Bedrock API key (bearer token) | AWS SigV4 (IAM) |
| Key handling | static key / token credential | short-term token via `aws-bedrock-token-generator` | AWS credential chain |
| Model selector | deployment name | Bedrock model ID | Bedrock model ID (`modelId`) |
| `api_version` | required | not used | not used |

---

## Operation mapping

| Azure OpenAI operation | Path A (OpenAI SDK on Bedrock) | Path B (AWS SDK) |
|---|---|---|
| `client.chat.completions.create(...)` | identical call, Bedrock `base_url` + model ID | `bedrock_runtime.converse(...)` |
| streaming (`stream=True`) | identical | `bedrock_runtime.converse_stream(...)` |
| `client.embeddings.create(...)` | supported on the endpoint for embedding models | `bedrock_runtime.invoke_model(...)` |
| `client.responses.create(...)` (Responses API) | supported for GPT-5.x on the endpoint | *(no Converse equivalent; use Path A)* |
| function calling (`tools=`, `tool_choice=`) | identical shape | `toolConfig=` + `toolChoice=` (see [tool-mapping.md](tool-mapping.md)) |
| server tools (`web_search`, etc.) | see [tool-mapping.md](tool-mapping.md) | see [tool-mapping.md](tool-mapping.md) |

OpenAI models on Bedrock support **Invoke, Converse, Chat Completions, and
Responses** APIs. Non-OpenAI models (Claude, Nova, Llama, …) support **Invoke
and Converse**; use Converse for a single portable shape.

---

## Request/response field mapping (chat)

The OpenAI shape (Azure and Path A) and the Converse shape (Path B) differ:

| Concept | OpenAI / Path A | Converse / Path B |
|---|---|---|
| system prompt | `{"role": "system", "content": "..."}` in `messages` | top-level `system=[{"text": "..."}]` |
| message content | string | list of blocks: `[{"text": "..."}]` |
| temperature / max tokens | top-level `temperature`, `max_tokens` | `inferenceConfig={"temperature":…, "maxTokens":…}` |
| top_p / stop | `top_p`, `stop` | `inferenceConfig={"topP":…, "stopSequences":[…]}` |
| reply text | `resp.choices[0].message.content` | `resp["output"]["message"]["content"][0]["text"]` |
| finish reason | `choices[0].finish_reason` (`stop`, `length`, `tool_calls`) | `resp["stopReason"]` (`end_turn`, `max_tokens`, `tool_use`) |
| token usage | `usage.prompt_tokens` / `completion_tokens` / `total_tokens` | `usage.inputTokens` / `outputTokens` / `totalTokens` |
| streaming delta | `event.choices[0].delta.content` | `event["contentBlockDelta"]["delta"]["text"]` |

The regression harness in [`../harness/`](../harness/) normalizes both shapes
into one dict so you can diff Azure vs. Bedrock output for parity.

---

## Which path?

- **Start with Path A** to get onto Bedrock with the smallest change.
- **Adopt Path B (Converse)** where you want provider independence — one API
  across every Bedrock model, unified tool use, and multimodal — so future
  model swaps and A/B tests are config, not code.

See [model-mapping.md](model-mapping.md) for model IDs and
[../migration/01_chat_completions/](../migration/01_chat_completions/) for
runnable before/after examples.
