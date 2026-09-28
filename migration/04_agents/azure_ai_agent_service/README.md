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

## Two migration targets

| Target | File | When to use |
|---|---|---|
| Bedrock Converse loop | `after/bedrock_converse_loop.py` | No new framework — pure `boto3`. Smallest surface area. |
| Strands + AgentCore Runtime | `after/bedrock_strands_agentcore.py` | Strategic: managed runtime with Memory, Identity, Gateway. Mirrors what Azure AI Agent Service provided. |

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

- **Tavily** — `pip install tavily-python` — clean async API, news + general
- **Brave Search API** — privacy-focused, good coverage
- **Bedrock AgentCore Gateway** — MCP-based tool proxy (no custom search code)
- **Amazon Kendra** — enterprise search (if corpus is internal docs)

## Verify

```bash
az2br scan migration/04_agents/azure_ai_agent_service/before   # should flag HIGH
az2br scan migration/04_agents/azure_ai_agent_service/after    # should be clean
```
