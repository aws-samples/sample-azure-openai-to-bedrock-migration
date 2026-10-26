# Semantic Kernel: Azure connector → Amazon Bedrock

- **Before:** [`before/azure_semantic_kernel.py`](before/azure_semantic_kernel.py)
- **After:** [`after/bedrock_semantic_kernel.py`](after/bedrock_semantic_kernel.py)

Semantic Kernel isolates the model behind a chat-completion **service** added to
the `Kernel`. Migrating to Bedrock is a service swap; your Kernel, plugins,
functions, and planners are unchanged.

Two ways to do it:

### 1. OpenAI-compatible endpoint (shown in `after/`, most portable)

`OpenAIChatCompletion` accepts a custom `async_client`, so point a standard
OpenAI async client at Bedrock's OpenAI-compatible endpoint:

```python
from openai import AsyncOpenAI
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

client = AsyncOpenAI(
    base_url=f"https://bedrock-runtime.{region}.amazonaws.com/openai/v1",
    api_key=provide_token(region=region),
)
kernel.add_service(OpenAIChatCompletion(ai_model_id="openai.gpt-oss-120b-1:0",
                                        async_client=client))
```

```bash
pip install semantic-kernel openai aws-bedrock-token-generator
```

### 2. Native Bedrock connector (SigV4 auth)

Semantic Kernel also ships an Amazon Bedrock connector
(`BedrockChatCompletion`) that authenticates with your AWS credentials directly
— no bearer token. **It does not support OpenAI models.** In `semantic-kernel`
1.45.0 it picks a request format from the model-ID prefix, covering AI21, Amazon,
Anthropic, Cohere, Meta, and Mistral only; passing `openai.gpt-oss-120b-1:0` or
`us.openai.gpt-6-sol` raises `Model ID ... does not contain a valid model
provider name`. Use it when you move to a non-OpenAI model; for OpenAI models use
the OpenAI-compatible route above. Its constructor varies across releases, so
check the version you have installed.

Notes:

- Semantic Kernel is now **Microsoft Agent Framework**, which Microsoft calls its
  enterprise-ready successor. The sample here is tested with `semantic-kernel`
  1.45.0 (Python) and is a lift-and-shift path.
- On this Chat Completions route, `openai.gpt-oss-120b-1:0` can return a
  `<reasoning>...</reasoning>` block inline in the reply. Use a GPT-5.6 / GPT-6
  model (for example `BEDROCK_MODEL_ID=global.openai.gpt-5.6-terra`) for clean
  output.
- Either way, `deployment_name` (Azure) becomes a Bedrock **model ID**
  ([../../../docs/model-mapping.md](../../../docs/model-mapping.md)).
- Endpoint / `api_key` / `api_version` all go away on the native connector; on
  the OpenAI-compatible path the key becomes a short-term Bedrock token.

For a managed AWS agent runtime, consider Strands + AgentCore — see the
[section README](../README.md).
