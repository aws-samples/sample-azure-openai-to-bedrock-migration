# 03 — Tool Use: Azure OpenAI → Amazon Bedrock

Azure OpenAI has two tool kinds, and they migrate differently:

1. **Client-side function calling** — you own the function; portable. → Bedrock
   `toolConfig` on Converse (this guide).
2. **Server-side tools** (`web_search`, `code_interpreter`, `file_search`) —
   Azure runs them. → different Bedrock services per tool (see
   [`server-tools.md`](server-tools.md)).

- **Before:** [`before/azure_tools.py`](before/azure_tools.py) — function calling + a `web_search` server tool
- **After:** [`after/bedrock_converse_tools.py`](after/bedrock_converse_tools.py) — the Converse tool-use loop
- **Server tools:** [`server-tools.md`](server-tools.md)

---

## Function calling → `toolConfig`

The concept is identical; the wire shape and the loop differ.

| Azure OpenAI | Bedrock Converse |
|---|---|
| `tools=[{"type":"function","function":{"name","description","parameters"}}]` | `toolConfig={"tools":[{"toolSpec":{"name","description","inputSchema":{"json": …}}}]}` |
| `tool_choice="auto"` | `toolChoice={"auto":{}}` |
| `tool_choice="required"` | `toolChoice={"any":{}}` |
| `tool_choice={"type":"function","function":{"name":"X"}}` | `toolChoice={"tool":{"name":"X"}}` *(Claude 3 / Nova only)* |

**The loop changes shape:**

| Step | Azure | Converse |
|---|---|---|
| model requests a tool | `message.tool_calls[]` | a `toolUse` content block; `stopReason == "tool_use"` |
| you return the result | `{"role":"tool","tool_call_id":…,"content":…}` | a `toolResult` content block (`toolUseId`, `content:[{"json":…}]`) |
| then | call again | call `converse()` again |

Compare the two files side by side — same `get_weather`, different plumbing.

### Strict schemas

Add `"strict": true` to a `toolSpec` to force tool arguments to match your JSON
schema exactly (structured output enforcement). Handy when downstream code
depends on the argument shape.

### Path A note

On the OpenAI-compatible endpoint (Path A), the `tools` / `tool_choice` shape is
**unchanged** from Azure — function calling just works. Use the Converse rewrite
(Path B) when you want one portable tool surface across all Bedrock models.

---

## Server tools (quick view)

| Azure server tool | Bedrock | Guide |
|---|---|---|
| `{"type": "web_search"}` | **Web Search** built-in tool (OpenAI Responses API, GPT-5.x) — near drop-in | [server-tools.md](server-tools.md) |
| `{"type": "code_interpreter"}` | AgentCore Code Interpreter or Lambda-backed tool | [server-tools.md](server-tools.md) |
| `{"type": "file_search"}` | Bedrock Knowledge Bases / OpenSearch | [server-tools.md](server-tools.md) |

---

## Verify with `az2br`

```bash
az2br scan migration/03_tool_use/before
```

Flags the function-calling shape, `tool_choice`, and the `web_search` server
tool — each with a note pointing here or to `server-tools.md`.
