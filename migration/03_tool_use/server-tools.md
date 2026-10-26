# Server Tools — Azure OpenAI → Amazon Bedrock

Azure's server-side (hosted) tools run inside Azure. Each maps to a different
Bedrock capability. This is the companion detail to
[../../docs/tool-mapping.md](../../docs/tool-mapping.md).

---

## `web_search` → Bedrock Web Search (near drop-in)

Bedrock has a **built-in, server-side Web Search tool**, so this is close to a
drop-in rather than a rewrite. Add the same tool object to the `tools` array on
the **OpenAI Responses API**, called on the **`bedrock-mantle`** endpoint:

```python
from aws_bedrock_token_generator import provide_token
from openai import OpenAI

region = "us-east-1"
client = OpenAI(
    base_url=f"https://bedrock-mantle.{region}.api.aws/openai/v1",
    api_key=provide_token(region=region),
)

response = client.responses.create(
    model="openai.gpt-5.6-terra",           # bare id on mantle
    input="What did AWS announce most recently about Bedrock?",
    tools=[{"type": "web_search", "external_web_access": False}],
)
print(response.output_text)
```

**Requirements / limits** (confirm current details in the Bedrock User Guide):

- **OpenAI Responses API only**, on the `bedrock-mantle` endpoint. Server-side
  tools like Web Search are **not** available when the Responses API is called
  on `bedrock-runtime`, and not available through Converse or InvokeModel.
- Mantle authorizes inference with the `bedrock-mantle:CreateInference` IAM
  action, plus `bedrock-mantle:CallWithBearerToken` when you authenticate with a
  Bedrock API key (the OpenAI SDK path — granting only `CreateInference` fails
  403), instead of `bedrock:InvokeModel` on `bedrock-runtime`. Mantle takes the
  bare model id (`openai.gpt-5.6-terra`); the `us.`/`global.` inference-profile
  id is a `bedrock-runtime` concept and 404s on mantle.
- Supported on **OpenAI GPT-5.4 and later** in the commercial US Regions. In
  AWS GovCloud (US) it is limited to `openai.gpt-5.6-terra`,
  `openai.gpt-5.6-luna`, and `openai.gpt-5.4`.
- IAM: `bedrock-websearch:InvokeSearch` (discover sources),
  `bedrock-websearch:InvokeFetch` (retrieve cached page content),
  `bedrock-websearch:ExternalWebAccess` (live outbound retrieval). Set
  `external_web_access: false` to keep request data inside the AWS boundary
  (search + cached fetch still work). `AmazonBedrockFullAccess` grants Search and
  Fetch but not `ExternalWebAccess`; the `AmazonBedrockExternalWebSearchReadOnly`
  and `...FullAccess` managed policies add it.
- Regions: US East (N. Virginia) `us-east-1`, US East (Ohio) `us-east-2`,
  US West (Oregon) `us-west-2`, and AWS GovCloud (US-West) `us-gov-west-1`.
  Web Search is strictly regional; queries are not routed across Regions.

**Controls**

- `search_context_size`: `low` (up to 5 results), `medium` (default, up to 11),
  or `high` (up to 25).
- `filters`: restrict results to up to 100 `allowed_domains` and/or exclude up
  to 100 `blocked_domains`. Omit the scheme; each entry also matches subdomains.
- `user_location`: localize results, e.g.
  `{"type": "approximate", "country": "US"}` (`country` is an ISO 3166-1 alpha-2
  code; `region`, `city`, and an IANA `timezone` are optional). Defaults to the
  United States when omitted.

```python
tools=[{
    "type": "web_search",
    "external_web_access": False,
    "filters": {"allowed_domains": ["aws.amazon.com", "docs.aws.amazon.com"]},
    "user_location": {"type": "approximate", "country": "DE"},
}]
```

**Not on mantle, or want any model?** AgentCore Gateway offers a managed **Web
Search** MCP connector that works with any MCP-compatible agent framework
(Strands, LangChain, LangGraph, CrewAI) and any model, with domain and
published-date filters. It suits the Converse/Strands path in
[../04_agents/](../04_agents/).

**Fallback** if your model or Region isn't supported: define a client-side
`web_search` **function tool** whose implementation calls a search provider you
choose, and run the normal tool-use loop (see the Converse example in
[`after/`](after/)).

---

## `code_interpreter` → AgentCore Code Interpreter or Lambda

No inference-API flag runs arbitrary code. Options:

- **AgentCore Code Interpreter** — a managed, isolated sandbox for agent
  code execution. Best when you're already building on AgentCore
  (see [../04_agents/](../04_agents/)).
- **A client-side `run_code` function tool** backed by **AWS Lambda** — you
  control the runtime, dependencies, timeout, and blast radius. Wire it through
  the same `toolConfig` loop as any other function tool.
- **A server-side custom tool** on the Responses API — register a Lambda function
  (or an AgentCore Gateway) and Bedrock runs it for you, so you skip the client
  loop. Model support is limited today; check the Bedrock User Guide topic
  "Server-side tool use" for your model before choosing this.

---

## `file_search` → Managed Knowledge Base or OpenSearch

Azure `file_search` is managed RAG over uploaded files. On Bedrock:

- **Bedrock Managed Knowledge Base** (recommended by AWS) — fully managed RAG.
  Point it at an S3 data source and it parses, chunks, embeds, indexes, and
  retrieves, with no vector store or embedding model to choose. Query with
  `retrieve` (using `managedSearchConfiguration`), or use agentic retrieval for a
  grounded, multi-hop answer in one call (`retrieve_and_generate` is not
  supported on Managed KBs). See
  [../05_data_plane/](../05_data_plane/).
- **Customer-managed Knowledge Base** — when you need to pick the vector store,
  embedding model, or parsing yourself.
- **Amazon OpenSearch** (Serverless or managed) — when you want to own the
  index and retrieval logic.

---

## Summary

| Azure server tool | Bedrock target | Effort |
|---|---|---|
| `web_search` | Web Search built-in tool (Responses API on bedrock-mantle, GPT-5.4 and later) | Low — near drop-in |
| `code_interpreter` | AgentCore Code Interpreter / Lambda tool | Medium |
| `file_search` | Managed Knowledge Base / OpenSearch | Medium |
