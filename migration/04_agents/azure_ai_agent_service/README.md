# Azure AI Agent Service → Amazon Bedrock

## What az2br flags

Running `az2br scan` on a real Azure AI Agent Service app
(`Azure-Samples/contoso-creative-writer`) produces:

| File | Severity | Category | Finding |
|---|---|---|---|
| `researcher.py:10` | HIGH | agents | `AIProjectClient` import |
| `researcher.py:12` | HIGH | agents | `AIProjectClient.from_connection_string` |
| `researcher.py:38` | HIGH | agents | `BingGroundingTool` |
| `researcher.py:11,39` | HIGH | auth | `DefaultAzureCredential` |
| `functions.json:3,26,49` | HIGH | tools | OpenAI function calling schema |
| `researcher.py:63` | MEDIUM | completions | `azure_deployment="gpt-4"` |

## The core migration challenge

Azure AI Agent Service runs the tool-call loop **server-side**.
You call `create_and_process_run()` and get back a final message —
you never see the intermediate `tool_call → tool_result` turns.

On Bedrock you **own the loop**: call `converse()`, check `stopReason`,
execute any `toolUse` blocks, append `toolResult`, call `converse()` again.

```
Azure                              Bedrock
─────────────────────────────────  ──────────────────────────────────────
create_agent(tools=bing.defs)      toolConfig = {tools: [toolSpec, ...]}
create_and_process_run()  ◄loop►   while stopReason == "tool_use":
                                       execute tools
                                       converse() again
list_messages()  ← final answer    response["output"]["message"]
```

## Tool schema conversion

`functions.json` (OpenAI) → `toolConfig` (Bedrock Converse)

```python
# OpenAI (functions.json)
{
  "type": "function",
  "function": {
    "name": "find_information",
    "description": "...",
    "parameters": { "type": "object", "properties": { ... } }
  }
}

# Bedrock toolConfig
{
  "toolSpec": {
    "name": "find_information",
    "description": "...",
    "inputSchema": { "json": { "type": "object", "properties": { ... } } }
  }
}
```

The inner JSON Schema is **identical** — only the outer wrapping changes.

## Four migration targets

| Target | File | When to use |
|---|---|---|
| Bedrock Converse loop | `after/bedrock_converse_loop.py` | No new framework — pure `boto3`. Smallest surface area. |
| Responses API loop | `after/bedrock_responses_api_loop.py` | You already use the OpenAI **Responses API** (or Azure OpenAI Assistants) and want to keep that surface for the agent loop. Runs on the mantle endpoint; can use the built-in server-side Web Search tool. |
| Strands + AgentCore Runtime | `after/bedrock_strands_agentcore.py` | Strategic: managed runtime with Memory, Identity, Gateway. Mirrors what Azure AI Agent Service provided. |
| Managed Agents (**preview**) | `after/bedrock_managed_agents_researcher.py` | You want the loop run *for you* as a service, like Azure AI Agent Service did, and you are standardized on OpenAI models. Tools run in an environment you provide. See [Bedrock Managed Agents](#bedrock-managed-agents-preview) below. |

All four migrate the same researcher scenario and return the same shape
(`{"web": [...], "entities": [...], "news": [...]}`), so you can diff targets
side by side.

### Converse loop vs. Responses API loop

Both own the tool-call loop (Azure ran it server-side); they differ only in the
API surface:

| | Converse (`bedrock_converse_loop.py`) | Responses API (`bedrock_responses_api_loop.py`) |
|---|---|---|
| Endpoint | `bedrock-runtime` (boto3) | `bedrock-mantle` (OpenAI SDK) |
| Model id | `us.openai.gpt-5.6-terra` (inference profile) | `openai.gpt-5.6-terra` (bare id) |
| "A tool is requested" | `stopReason == "tool_use"` | `output` item of type `function_call` |
| Tool schema | `toolSpec` + `inputSchema={"json": …}` | `{"type":"function","name":…,"parameters":…}` |
| Return a result | `{"toolResult":{"toolUseId":…}}` | `{"type":"function_call_output","call_id":…}` |
| Loop state | resend `messages` each turn | resend `input` list each turn (or `previous_response_id`) |
| Server-side Web Search | no | yes (mantle built-in `web_search` tool) |

## Bedrock Managed Agents (preview)

Of the four targets this is the only one besides Strands + AgentCore where you do
not own the tool-call loop. It is also the most different from Azure in one way
that matters: **the service runs the model loop, but your tools run in an
execution environment you provide** (your own host with `codex exec-server`, or
an AgentCore Runtime). Azure ran Bing grounding for you; here you serve the search
tools yourself, over MCP.

> **Public preview.** APIs and IAM action names can change; do not build a
> production dependency on it yet. Verified live (us-east-1, Codex 0.164) while
> writing this example. Current surface:
> [Bedrock User Guide](https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-managed-agents-openai.html).

### Concept map

| Azure AI Agent Service / Assistants | Bedrock Managed Agents |
|---|---|
| `create_agent(model, instructions, tools)` | the `agent` block of `POST /openai/v1/agents/sessions` (no standalone agent resource) |
| `create_thread()` | the session (created by the same call) |
| `create_message(thread_id, …)` | `POST …/sessions/{id}/events` with `agent.session.input.message` |
| `create_and_process_run()` | a turn: asynchronous, no run object; wait for the new `final_answer` item |
| `list_messages()` | `GET …/sessions/{id}/items` (cursor-paginated; items carry a `turn_id`) |
| cancel a run | `agent.session.input.cancel` event |
| `delete_agent()` / `delete_thread()` | `DELETE …/sessions/{id}` |
| `BingGroundingTool`, `functions.json` | a stdio MCP server in the execution environment: [`after/bma_research_mcp.py`](after/bma_research_mcp.py) |
| `code_interpreter` | shell commands in the execution environment (you own the sandbox) |
| `.prompty` / instructions | `agent.instructions`, plus filesystem skills (`SKILL.md`) |
| `DefaultAzureCredential` | AWS SigV4, signing service `bedrock-mantle`; no API key |
| `tenacity` retry on `rate_limit_exceeded` | backoff with jitter on HTTP 429 |

The tool's JSON Schema is the one from `functions.json` unchanged; only the
envelope changes (`parameters` becomes MCP `inputSchema`), the same as the other
targets.

### Run it

```bash
# 1. One-time: the two least-privilege IAM roles (no compute, no cost)
cd migration/04_agents/azure_ai_agent_service/bma_iam_cdk
pip install -r requirements.txt
cdk bootstrap                                        # once per account/Region
cdk deploy                                           # cdk destroy to remove

# 2. Run the example as BmaClientRole, not as an admin. Needs Codex CLI >= 0.154.0.
pip install boto3 requests
AWS_PROFILE=<profile that assumes BmaClientRole> \
  python migration/04_agents/azure_ai_agent_service/after/bedrock_managed_agents_researcher.py
```

The script creates the session, starts `codex exec-server` against it, runs two
turns on the same session (the second narrows the first, as a thread would), then
deletes the session and stops the exec server. Set `BMA_ATTACH=external` to start
`codex exec-server` yourself. Offline tests:
`python harness/test_bma_researcher.py`; the live check is Test 4 in
`harness/smoke_test_researcher.py` (`BMA_SMOKE=1`). Run it as `BmaClientRole` and
Tests 1-3 (direct Bedrock calls) will fail with `AccessDenied` by design, since
that role cannot invoke models itself; use a profile with Bedrock access for those.

### IAM: three identities, kept separate

[`bma_iam_cdk/app.py`](bma_iam_cdk/app.py) creates the two roles you own. It is
narrower than the example AWS ships with the preview, which grants
`bedrock-mantle:CreateInference` on every model plus Web Search.

| Identity | Can do | Cannot do (verified) |
|---|---|---|
| `BmaClientRole` (your code and the exec server) | create/get/list/delete sessions, send events, read items, register and connect the environment; `iam:PassRole` for the session role only | pass any other role (`403`), call anything outside BMA (`ListRoles` denied) |
| `BedrockManagedAgentsPreviewInferenceServiceRole` (assumed by BMA) | `bedrock-mantle:CreateInference` for the models in `-c models=…` (default `openai.gpt-5.6-luna`); trust is scoped with `aws:SourceAccount`/`aws:SourceArn` | invoke any other model |

Auth here is SigV4, so `bedrock-mantle:CallWithBearerToken` (needed by the
Responses API loop's OpenAI-SDK path) is **not** required, and no
`x-amzn-account-id` header is needed.

### Things that will bite you (all verified live)

- **A failed turn is only visible in the event stream.** If the session role may
  not invoke the model, the session still reports `status: idle` with
  `error: null`, and the items hold only your message. A client that only polls
  hangs until its timeout. The stream carries an `error` event and
  `agent.session.turn.failed` with the real reason (for example the missing
  `bedrock-mantle:CreateInference`). The example opens the stream before it
  submits and raises that message, in about 12 seconds instead of a timeout.
- **`idle` does not mean the turn finished.** Right after you submit, the session
  can still read `idle`. The example waits for a new assistant message with
  `phase: "final_answer"` from a turn it has not seen.
- **Bare model ids only.** `openai.gpt-5.6-luna`, not `us.openai.gpt-5.6-luna`;
  cross-Region inference profiles are not supported by the preview.
- **Paths are as the execution environment sees them.** The MCP `command`, `args`
  and `cwd` must be absolute paths on the host running the exec server.
- **Pass secrets by name.** Use `env_vars: ["TAVILY_API_KEY"]`; values placed in
  `env` may be stored with the session and printed in logs (per the BMA docs).
- **When not to use it.** For a plain function-calling research task the
  [Responses API loop](after/bedrock_responses_api_loop.py) is simpler and faster
  to start. Managed Agents pays off for longer, stateful work that needs files,
  shell and skills.
- **Not covered here:** the AgentCore Runtime environment (its example stack
  creates a NAT gateway and storage that keep billing while idle), subagents and
  code mode (unsupported in the preview), and long-term memory (none built in).

## Auth migration

| Azure | Bedrock |
|---|---|
| `DefaultAzureCredential()` | IAM role (EC2/ECS/Lambda instance profile) |
| `get_bearer_token_provider(...)` | `boto3.Session()` — credentials from env / instance profile |
| `azure_endpoint=...cognitiveservices.azure.com` | Regional Bedrock endpoint (boto3 auto-resolves) |
| `AZURE_OPENAI_API_KEY` env var | AWS credentials (env, `~/.aws/credentials`, or instance profile) |

## Search tool replacement

`BingGroundingTool` backed by Bing Search API.
Bedrock equivalents (choose one):

- **Bedrock Web Search** — built-in server-side tool on the Responses API
  (`bedrock-mantle`, OpenAI GPT-5.4 and later). Near drop-in and stays inside the
  AWS boundary; see [`../../03_tool_use/server-tools.md`](../../03_tool_use/server-tools.md).
  Optionally wired into `after/bedrock_responses_api_loop.py`.
- **AgentCore Gateway Web Search** — managed MCP connector that works with any
  model and framework (Strands, LangChain, LangGraph, CrewAI), with domain and
  published-date filters. The natural fit for the Converse and Strands targets.
- **Tavily** — `pip install tavily-python` — clean async API, news + general
- **Brave Search API** — privacy-focused, good coverage
- **Bedrock AgentCore Gateway** (custom targets) — MCP-based tool proxy for your own search API
- **Bedrock Managed Knowledge Base** — managed RAG (if the corpus is internal docs); see [05_data_plane](../../05_data_plane/README.md)
- **Amazon OpenSearch** (Serverless or managed) — when you want to own the index and retrieval logic

## Verify

```bash
az2br scan migration/04_agents/azure_ai_agent_service/before   # should flag HIGH
az2br scan migration/04_agents/azure_ai_agent_service/after    # should be clean
```
