# LangChain: AzureChatOpenAI → ChatBedrockConverse

- **Before:** [`before/azure_langchain.py`](before/azure_langchain.py)
- **After:** [`after/bedrock_langchain.py`](after/bedrock_langchain.py)

The migration is a **connector swap**. Replace the model object; leave your
chains, agents, tools, and `.invoke()` calls alone.

```python
# before
from langchain_openai import AzureChatOpenAI
llm = AzureChatOpenAI(azure_endpoint=..., api_key=..., azure_deployment="my-gpt-4o",
                      api_version="2024-10-21", temperature=0.2)

# after
from langchain_aws import ChatBedrockConverse
llm = ChatBedrockConverse(model="openai.gpt-oss-120b-1:0",
                          region_name="us-east-1", temperature=0.2)
```

```bash
pip install langchain-aws
```

Notes:

- **Tested with `langchain-aws` 1.9.1** against `openai.gpt-oss-120b-1:0` and
  `us.openai.gpt-6-sol` (the sample drops the hardcoded `temperature` for
  frontier GPT models). Recent releases added `reasoning_effort` support for
  GPT-6 Sol/Luna on `ChatBedrockConverse` (1.8.0).
- `langchain-aws[openai]` also provides `ChatOpenAIMantle`, which targets the
  `bedrock-mantle` endpoint and defaults to the Responses API for GPT-5.x/6. It
  is an option when you need mantle-only features; it was not tested here.
- **`ChatBedrockConverse`** uses the Converse API, so it works across every
  Bedrock model — swapping models is just changing `model=`.
  (`ChatBedrock` also exists and uses InvokeModel; prefer `ChatBedrockConverse`
  for the portable path and unified tool use.)
- **Auth** is standard AWS credentials — no endpoint, key, or `api_version`.
- **Tools / function calling** via LangChain's `bind_tools(...)` continue to
  work; they compile to Bedrock `toolConfig` under the hood.
- **Model IDs:** [../../../docs/model-mapping.md](../../../docs/model-mapping.md).

For a first-class AWS agent runtime (managed memory, identity, tool sandboxes),
consider moving the orchestration to Strands + AgentCore — see the
[section README](../README.md).
