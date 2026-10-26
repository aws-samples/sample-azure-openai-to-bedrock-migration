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

## Three migration targets

| Target | File | When to use |
|---|---|---|
| Bedrock Converse loop | `after/bedrock_converse_loop.py` | No new framework — pure `boto3`. Smallest surface area. |
| Responses API loop | `after/bedrock_responses_api_loop.py` | You already use the OpenAI **Responses API** (or Azure OpenAI Assistants) and want to keep that surface for the agent loop. Runs on the mantle endpoint; can use the built-in server-side Web Search tool. |
| Strands + AgentCore Runtime | `after/bedrock_strands_agentcore.py` | Strategic: managed runtime with Memory, Identity, Gateway. Mirrors what Azure AI Agent Service provided. |

All three migrate the same researcher scenario and return the same shape
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
