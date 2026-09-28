# 04 — Agents: Azure → Amazon Bedrock AgentCore + Strands

Agent workloads are the highest-effort migration because the *framework*, not
just the model call, is coupled to Azure. This section covers the three common
starting points and two targets.

## Two targets

- **Framework-native swap (fastest):** keep your framework (LangChain, AutoGen,
  Semantic Kernel) and swap the **Azure model connector** for the Bedrock one.
  Smallest diff; you keep your orchestration code.
- **Strands Agents + AgentCore Runtime (strategic):** adopt the
  [Strands Agents SDK](https://strandsagents.com) for the agent loop and deploy
  on **Amazon Bedrock AgentCore Runtime** — a managed, serverless runtime for
  agents with memory, identity, gateways, and tool sandboxes. Best when you want
  a first-class AWS agent platform rather than framework glue.

You can do the swap first to get onto Bedrock, then migrate to Strands where it
pays off.

## Framework guides

| Starting point | Guide | Framework-native target |
|---|---|---|
| LangChain (`AzureChatOpenAI`) | [`langchain/`](langchain/) | `ChatBedrockConverse` (`langchain-aws`) |
| AutoGen (Azure client) | [`autogen/`](autogen/) | Bedrock via AutoGen's OpenAI-compatible client / `langchain-aws` |
| Semantic Kernel (`AzureChatCompletion`) | [`semantic_kernel/`](semantic_kernel/) | Bedrock connector for SK |
| Azure AI Agent Service | *(see below)* | AgentCore + Strands |

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
AI Agent Service bundled, now as composable AWS services.

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
