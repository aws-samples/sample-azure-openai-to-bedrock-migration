# Regression Test Harness — Azure ↔ Bedrock chat parity

A small **record / replay / diff** harness to validate that your chat workloads
behave equivalently on Amazon Bedrock before you cut over from Azure OpenAI.

1. **record** — call your **source** (Azure OpenAI) for a set of test cases and
   save the normalized responses.
2. **replay** — call a **Bedrock target** (the OpenAI-compatible endpoint *or*
   the Converse API) for the same cases.
3. **diff** — compare the two sets for parity and exit non-zero if any case
   regresses (CI-friendly).

Dependencies: **Python stdlib + `openai` + `boto3` only.** The `diff` step is
pure stdlib and runs offline — you can re-check parity, and run the test suite,
without any credentials or network.

```
harness/
├── cases.jsonl        # test cases: {id, messages, params} per line
├── harness.py         # CLI: record / replay / diff
├── providers.py       # source (Azure/OpenAI) + Bedrock targets (endpoint, Converse)
├── normalize.py       # map both provider shapes into one comparable dict
├── diff.py            # structural parity + difflib similarity
├── test_diff.py       # offline unit tests (no network)
└── recordings/        # saved responses, per label (git-ignored)
```

## Install

```bash
pip install -r harness/requirements.txt   # openai, boto3, aws-bedrock-token-generator, pytest
```

## Quick start

```bash
# 1. Record from Azure (reads AZURE_OPENAI_* env vars)
python harness/harness.py record --label azure

# 2. Replay on Bedrock's OpenAI-compatible endpoint (Path A)
python harness/harness.py replay --provider bedrock-endpoint --label bedrock

#    …or on the Converse API (Path B)
python harness/harness.py replay --provider bedrock-converse --label bedrock-converse

# 3. Diff for parity (writes a JSON report; exits 1 if any case fails)
python harness/harness.py diff --source azure --target bedrock --report parity.json
```

## The parity model

Two correct LLM answers are rarely byte-identical, so the **default** is not
exact string equality. A case passes when **all** of the following hold:

| Check | What it compares |
|-------|------------------|
| **finish reason** | Normalized stop reason matches (`end_turn`→`stop`, `max_tokens`→`length`, …). |
| **tool calls** | Same tool names and same JSON-normalized arguments, in order. |
| **text similarity** | `difflib` ratio over normalized (lowercased, whitespace-collapsed) text ≥ threshold. |

- **Default threshold:** `0.60`. Raise it for deterministic tasks
  (`--threshold 0.85`), lower it for open-ended generation.
- **Exact mode:** `--exact` requires byte-identical text (after normalization)
  instead of the similarity score. Best paired with `temperature: 0.0` cases.

```bash
# stricter
python harness/harness.py diff --source azure --target bedrock --threshold 0.85
# strictest
python harness/harness.py diff --source azure --target bedrock --exact
```

### Normalized response shape

Both provider response formats are flattened to one dict before comparison:

```json
{
  "text": "…",
  "finish_reason": "stop",
  "tool_calls": [{"name": "get_weather", "arguments": {"city": "Seattle"}}],
  "role": "assistant",
  "usage": {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30}
}
```

The OpenAI-compatible endpoint returns the OpenAI Chat Completions shape
(`choices[0].message.content`, `finish_reason`, `usage.prompt_tokens`), while
Converse returns `output.message.content[].text`, `stopReason`, and
`usage.inputTokens`. `normalize.py` maps both onto the shape above so the diff
compares like for like.

## Test cases

`cases.jsonl` — one JSON object per line:

```json
{"id": "math", "messages": [{"role": "user", "content": "What is 17 * 23? Reply with only the number."}], "params": {"temperature": 0.0, "max_tokens": 16}}
```

- `id` — unique; used as the recording filename.
- `messages` — OpenAI-style messages. For Converse replay, `system` messages are
  split out and each message's content is wrapped in a text block automatically.
- `params` — OpenAI-style params. For Converse, `max_tokens`→`maxTokens`,
  `top_p`→`topP`, `stop`→`stopSequences` are mapped for you.

Point at your own file with `--cases path/to/cases.jsonl`.

## Configuration (environment variables)

**Source (record):** Azure by default —
`AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT`,
`AZURE_OPENAI_API_VERSION` (default `2024-10-21`). To record from a generic
OpenAI-compatible source instead, set `SOURCE_BASE_URL`, `SOURCE_API_KEY`,
`SOURCE_MODEL`.

**Bedrock targets (replay):** `AWS_REGION` (default `us-east-1`),
`BEDROCK_MODEL_ID` (default `us.openai.gpt-5.6-terra`), plus standard AWS
credentials. For the endpoint target, a short-term Bedrock API key is generated
from your AWS credentials via `aws-bedrock-token-generator`; set
`AWS_BEARER_TOKEN_BEDROCK` to supply your own instead.

No secrets are stored in code — everything comes from the environment.

## Run the tests

```bash
python harness/test_diff.py    # 10 offline tests, no network
```

## Proving the examples work

The `before/` (Azure) and `after/` (Bedrock) examples need different providers,
so they're validated differently:

| Level | What it proves | How |
|-------|----------------|-----|
| Static | code is well-formed and representative | `py_compile` + `az2br scan` flags each `before/` |
| Live (after) | the Bedrock targets actually run | `smoke_test_after.py` (this repo) — real Bedrock calls |
| Live (before) | the Azure examples actually run | run them against a live Azure OpenAI resource |
| Parity | before ≈ after | `record` (Azure) → `replay` (Bedrock) → `diff` |

### Smoke-test the `after/` examples against your account

```bash
AWS_REGION=us-east-1 BEDROCK_MODEL_ID=us.openai.gpt-5.6-terra \
    python harness/smoke_test_after.py
```

Makes real Bedrock calls to verify: Converse chat, Converse tool use, Titan V2
embeddings, the OpenAI-compatible endpoint (Path A), LangChain, AutoGen, and
Semantic Kernel. Framework/endpoint examples whose SDK isn't installed are
skipped, not failed. Exits non-zero if any check fails.

To include the framework checks, install their SDKs:

```bash
pip install langchain-aws semantic-kernel   # + pyautogen for the AutoGen check
```

### CI

`.github/workflows/ci.yml` runs two tiers:

- **offline** (always): builds `az2br`, scans every `before/` example (expects
  HIGH findings), `py_compile`s all Python, and runs the offline unit tests.
- **smoke** (gated): runs this live smoke test, but only when an OIDC role is
  configured. Set the repo variable `AWS_SMOKE_ROLE_ARN` to an IAM role that
  trusts the GitHub OIDC provider and can invoke Bedrock; optionally set
  `AWS_SMOKE_REGION` and `BEDROCK_MODEL_ID`. On forks/PRs without the variable,
  the smoke job is skipped so CI stays green without credentials.

> Note: on reasoning models (e.g. `openai.gpt-oss-*`), Converse returns a
> `reasoningContent` block before the `text` block, and `ChatBedrockConverse`
> returns `content` as a list. The `after/` examples extract text across blocks
> rather than assuming `content[0]` — see the code.
