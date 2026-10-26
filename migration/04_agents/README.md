# 04 — Agents: Azure → Amazon Bedrock AgentCore + Strands

Agent workloads are the highest-effort migration because the *framework*, not
just the model call, is coupled to Azure. This section covers the three common
starting points and two targets.

## Targets

- **Framework-native swap (fastest):** keep your framework (LangChain, AutoGen,
  Semantic Kernel) and swap the **Azure model connector** for the Bedrock one.
  Smallest diff; you keep your orchestration code.
- **Strands Agents + AgentCore Runtime (strategic):** adopt the
  [Strands Agents SDK](https://strandsagents.com) for the agent loop and deploy
  on **Amazon Bedrock AgentCore Runtime** — a managed, serverless runtime for
  agents with memory, identity, gateways, and tool sandboxes. Best when you want
  a first-class AWS agent platform rather than framework glue.
- **AgentCore harness (managed agent loop):** declare a model, system prompt, and
  tools, and AgentCore runs the agent loop for you — no orchestration code.
  Generally available since June 2026, and the closest analog to Azure AI Agent
  Service, which also runs the loop server-side. You can export the harness to
  Strands code if you later need full control.
- **Bedrock Managed Agents, powered by OpenAI (public preview):** OpenAI's agent
  harness as an AWS service, so teams building on OpenAI models can keep that
  agent experience while running entirely inside AWS with IAM identity,
  CloudTrail logging, durable sessions, skills, and MCP tools. You choose where
  tools run: your own host or container, or an AgentCore Runtime in your account.
  It is in **public preview**, so details may change: today it uses the
  `bedrock-mantle` endpoint in us-east-1, us-east-2, and us-west-2, takes text
  input, and uses the OpenAI models available in your Region. Check the
  [Bedrock User Guide](https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-managed-agents-openai.html)
  for the current surface before you plan around it.
  A worked Azure AI Agent Service migration, with live-verified IAM and gotchas, is in
  [`azure_ai_agent_service/`](azure_ai_agent_service/README.md#bedrock-managed-agents-preview).

### Choosing between the managed options

Both options give you a managed agent loop on AWS. Pick by the model family and
the operating model you want.

| | **AgentCore harness** | **Bedrock Managed Agents, powered by OpenAI** |
|---|---|---|
| Status | Generally available | Public preview (may change) |
| Models | Model-agnostic; configure the model per agent | OpenAI models on Bedrock |
| Where tools run | AgentCore provisions the compute, memory, and identity for you | The environment you choose: your own compute or an AgentCore Runtime |
| Good fit | A first managed home for Azure AI Agent Service workloads; mixed or non-OpenAI models; export to Strands for more control | Teams standardized on OpenAI's agent harness who want it inside AWS governance |

They also compose: Managed Agents can use AgentCore Runtime as its execution
environment.

You can do the swap first to get onto Bedrock, then migrate to Strands or the
harness where it pays off.

## Framework guides

| Starting point | Guide | Framework-native target |
|---|---|---|
| LangChain (`AzureChatOpenAI`) | [`langchain/`](langchain/) | `ChatBedrockConverse` (`langchain-aws`) |
| AutoGen (Azure client) | [`autogen/`](autogen/) | AG2 Classic (`pip install "autogen[openai]"`) via Bedrock's OpenAI-compatible endpoint. Microsoft's `autogen-agentchat` is archived and not covered |
| Semantic Kernel (`AzureChatCompletion`) | [`semantic_kernel/`](semantic_kernel/) | OpenAI-compatible endpoint for OpenAI models; native Bedrock connector for non-OpenAI models. SK is now Microsoft Agent Framework |
| Azure AI Agent Service | *(see below)* | AgentCore harness, or AgentCore + Strands |

## Azure AI Agent Service → AgentCore + Strands

Azure AI Agent Service (the `AIProjectClient` / `azure.ai.projects` managed agent
runtime) has no drop-in SDK equivalent — it's a managed runtime, so the
migration target is also a managed runtime: **AgentCore Runtime** with a Strands
agent. The shape:

```python
from strands import Agent
from strands_tools import file_read, file_write
from bedrock_agentcore.runtime import BedrockAgentCoreApp

agent = Agent(tools=[file_read, file_write])   # your model + tools
app = BedrockAgentCoreApp()

@app.entrypoint
def agent_invocation(payload, context):
    user_message = payload.get("prompt", "")
    result = agent(user_message)
    return {"result": result.message}

app.run()
```

You then deploy the container to AgentCore Runtime. AgentCore adds managed
Memory, Identity, Gateway (MCP tools), and Code Interpreter — the pieces Azure
AI Agent Service bundled, now as composable AWS services. AgentCore Gateway can
also front model traffic (including `InvokeModel`) and fail over to a fallback
model when the primary is throttled or unavailable.

If you'd rather not write the loop at all, the **AgentCore harness** is the
closest managed equivalent: you configure the model, instructions, and tools and
AgentCore handles compute, memory, identity, and observability.

## The common denominator

Under every framework, the model call becomes a Bedrock call:

- **Model IDs** → [../../docs/model-mapping.md](../../docs/model-mapping.md)
- **Auth** → standard AWS credentials (IAM), not Azure keys/Entra
- **Tools** → Bedrock `toolConfig` / server tools → [../03_tool_use/](../03_tool_use/)

## Verify with `az2br`

```bash
az2br scan migration/04_agents
```

Flags `AzureChatOpenAI`, `AzureChatCompletion`, and Azure AI Agent Service
references across the subdirectories.
