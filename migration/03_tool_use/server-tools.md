# Server Tools — Azure OpenAI → Amazon Bedrock

Azure's server-side (hosted) tools run inside Azure. Each maps to a different
Bedrock capability. This is the companion detail to
[../../docs/tool-mapping.md](../../docs/tool-mapping.md).

---

## `web_search` → Bedrock Web Search (near drop-in)

Bedrock now has a **built-in, server-side Web Search tool**, so this is close to
a drop-in rather than a rewrite. Add the same tool object to the `tools` array
on the **OpenAI Responses API**:

```python
from aws_bedrock_token_generator import provide_token
from openai import OpenAI

region = "us-east-1"
client = OpenAI(
    base_url=f"https://bedrock-mantle.{region}.api.aws/openai/v1",
    api_key=provide_token(region=region),
)

response = client.responses.create(
    model="us.openai.gpt-5.6-terra",
    input="What did AWS announce most recently about Bedrock?",
    tools=[{"type": "web_search"}],
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
  bare model id (`openai.gpt-5.6-terra`); the `us.` inference-profile id is a
  `bedrock-runtime` concept and 404s on mantle.
- Supported on **OpenAI GPT-5.4, GPT-5.5, GPT-5.6** (Sol / Terra / Luna).
- IAM: `bedrock-websearch:InvokeSearch` (discover sources),
  `bedrock-websearch:InvokeFetch` (retrieve cached page content),
  `bedrock-websearch:ExternalWebAccess` (live outbound retrieval). Set
  `external_web_access: false` to keep request data inside the AWS boundary
  (search + cached fetch still work).
- Regionally limited (US East/West and GovCloud US-West at time of writing).

**Fallback** if your model or Region isn't supported: define a client-side
`web_search` **function tool** whose implementation calls a search provider you
choose, and run the normal tool-use loop (see the Converse example in
[`after/`](after/)).

---

## `code_interpreter` → AgentCore Code Interpreter or Lambda

No inference-API flag runs arbitrary code. Two options:

- **AgentCore Runtime Code Interpreter** — a managed, isolated sandbox for agent
  code execution. Best when you're already building on AgentCore
  (see [../04_agents/](../04_agents/)).
- **A client-side `run_code` function tool** backed by **AWS Lambda** — you
  control the runtime, dependencies, timeout, and blast radius. Wire it through
  the same `toolConfig` loop as any other function tool.

---

## `file_search` → Knowledge Bases or OpenSearch

Azure `file_search` is managed RAG over uploaded files. On Bedrock:

- **Bedrock Knowledge Bases** — fully managed RAG. Point it at an S3 data
  source, choose an embedding model (`embeddingModelArn`), and it chunks,
  embeds, indexes, and retrieves. Retrieve with `retrieve` /
  `retrieve_and_generate`. See [../05_data_plane/](../05_data_plane/).
- **Amazon OpenSearch** (Serverless or managed) — when you want to own the
  index and retrieval logic. Also the vector store Knowledge Bases can create
  for you.

---

## Summary

| Azure server tool | Bedrock target | Effort |
|---|---|---|
| `web_search` | Web Search built-in tool (Responses API on bedrock-mantle, GPT-5.x) | Low — near drop-in |
| `code_interpreter` | AgentCore Code Interpreter / Lambda tool | Medium |
| `file_search` | Knowledge Bases / OpenSearch | Medium |
