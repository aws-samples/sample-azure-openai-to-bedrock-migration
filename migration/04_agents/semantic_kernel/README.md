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
— no bearer token. Prefer it if you want native SigV4 auth. Its exact
constructor varies across `semantic-kernel` releases, so check the version you
have installed rather than copying a signature blindly.

Notes:

- Either way, `deployment_name` (Azure) becomes a Bedrock **model ID**
  ([../../../docs/model-mapping.md](../../../docs/model-mapping.md)).
- Endpoint / `api_key` / `api_version` all go away on the native connector; on
  the OpenAI-compatible path the key becomes a short-term Bedrock token.

For a managed AWS agent runtime, consider Strands + AgentCore — see the
[section README](../README.md).
