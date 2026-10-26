# Tool Mapping — Azure OpenAI tools → Amazon Bedrock

Azure OpenAI offers two kinds of tools:

1. **Client-side function calling** — you define a function schema, the model
   asks you to call it, you run it and return the result. Portable everywhere.
2. **Server-side (hosted) tools** — Azure runs the tool for you: `web_search`,
   `code_interpreter`, `file_search`. These are the ones that need the most
   thought when migrating.

This page maps both to Bedrock.

---

## Function calling → Bedrock `toolConfig`

Same concept, different wire shape. On Converse (Path B), tools go in
`toolConfig`; the model replies with a `toolUse` block; you execute and append a
`toolResult` block, then call again.

| Azure OpenAI | Bedrock Converse |
|---|---|
| `tools=[{"type":"function","function":{"name","description","parameters"}}]` | `toolConfig={"tools":[{"toolSpec":{"name","description","inputSchema":{"json": …}}}]}` |
| `tool_choice="auto"` | `toolChoice={"auto":{}}` |
| `tool_choice="required"` | `toolChoice={"any":{}}` |
| `tool_choice={"type":"function","function":{"name":"X"}}` | `toolChoice={"tool":{"name":"X"}}` *(Claude 3 / Nova only)* |
| model returns `message.tool_calls[]` | model returns a `toolUse` content block |
| you return `{"role":"tool","content":…}` | you append a `toolResult` content block |
| — | `"strict": true` on a `toolSpec` enforces the schema (structured output) |

On **Path A** (OpenAI-compatible endpoint), the `tools` / `tool_choice` shape is
unchanged — function calling works exactly as it does on Azure.

Full runnable example: [../migration/03_tool_use/](../migration/03_tool_use/).

---

## Server tools

| Azure server tool | Bedrock equivalent | Effort |
|---|---|---|
| `{"type": "web_search"}` | **Web Search** — built-in server tool: `{"type": "web_search"}` on the **OpenAI Responses API** (GPT-5.4 and later), on the **bedrock-mantle** endpoint | Low — near drop-in |
| `{"type": "code_interpreter"}` | AgentCore **Code Interpreter** sandbox, or a Lambda-backed tool | Medium |
| `{"type": "file_search"}` | **Bedrock Managed Knowledge Base** (managed RAG) or Amazon OpenSearch | Medium |

### `web_search` → Bedrock Web Search

Bedrock has a **built-in, server-side Web Search tool** — so this is now close
to a drop-in, not a rewrite. Add the same tool object to the `tools` array:

```python
# client = OpenAI(base_url="https://bedrock-mantle.{region}.api.aws/openai/v1", ...)
response = client.responses.create(
    model="openai.gpt-5.6-terra",      # bare id on mantle (us./global. profile ids are bedrock-runtime only)
    input="What did AWS announce at re:Invent this year?",
    tools=[{"type": "web_search", "external_web_access": False}],
)
```

Requirements and constraints (verify current details in the Bedrock User Guide):

- **OpenAI Responses API only**, on the `bedrock-mantle` endpoint. **Not**
  available through Converse or InvokeModel, and **not** through the Responses
  API on `bedrock-runtime` (server-side tools are mantle-only). Mantle
  authorizes inference with `bedrock-mantle:CreateInference` rather than
  `bedrock:InvokeModel`.
- Supported on **OpenAI GPT-5.4 and later** in the commercial US Regions (narrower
  list in GovCloud).
- IAM permissions: `bedrock-websearch:InvokeSearch` (discover sources),
  `bedrock-websearch:InvokeFetch` (retrieve cached page content),
  `bedrock-websearch:ExternalWebAccess` (live outbound retrieval; set
  `external_web_access: false` to keep requests inside the AWS boundary).
- Regions: `us-east-1`, `us-east-2`, `us-west-2`, and GovCloud `us-gov-west-1`.
  If your model or Region isn't supported, fall back to a client-side function
  tool backed by a search provider, or the AgentCore Gateway Web Search connector.
- Optional controls: `search_context_size`, `filters` (`allowed_domains` /
  `blocked_domains`, up to 100 each), and `user_location`. See
  [../migration/03_tool_use/server-tools.md](../migration/03_tool_use/server-tools.md).

### `code_interpreter` → AgentCore Code Interpreter / Lambda

There is no "run this arbitrary code" flag on the base inference APIs. Two
options:

- **AgentCore Code Interpreter** — a managed, isolated sandbox for agent
  code execution.
- **A client-side tool** — define a `run_code` function tool whose backing
  implementation executes in an **AWS Lambda** sandbox you control (tighter
  blast radius, your own dependencies).

### `file_search` → Managed Knowledge Base / OpenSearch

Azure's `file_search` is managed RAG over uploaded files. On Bedrock:

- **Bedrock Managed Knowledge Base** (AWS-recommended) — fully managed RAG: point
  it at an S3 data source and it parses, chunks, embeds, and retrieves for you.
  See [../migration/05_data_plane/](../migration/05_data_plane/).
- **Amazon OpenSearch** (Serverless or managed) — when you want to own the index
  and retrieval logic directly.

---

## Decision guide

- **Function calling** → `toolConfig` (Converse) or unchanged (Path A). Always
  portable.
- **`web_search`** → Bedrock Web Search on the Responses API (bedrock-mantle
  endpoint) with a GPT-5.4+ model, if your Region supports it; else a
  client-side search tool.
- **`code_interpreter`** → AgentCore Code Interpreter or a Lambda-backed tool.
- **`file_search`** → Bedrock Managed Knowledge Base or OpenSearch (DIY).

See also: [api-mapping.md](api-mapping.md), [model-mapping.md](model-mapping.md).
