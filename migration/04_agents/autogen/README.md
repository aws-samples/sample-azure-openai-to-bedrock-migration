# AutoGen: Azure OpenAI → Amazon Bedrock

- **Before:** [`before/azure_autogen.py`](before/azure_autogen.py)
- **After:** [`after/bedrock_autogen.py`](after/bedrock_autogen.py)

AutoGen drives agents from an OpenAI-shaped `llm_config`. The cleanest Bedrock
path is to point that config at Bedrock's **OpenAI-compatible endpoint**
(Path A) — the agent graph (`AssistantAgent`, `UserProxyAgent`, group chats)
stays the same.

```python
# before — Azure
{"model": "my-gpt-4o", "api_type": "azure",
 "base_url": AZURE_ENDPOINT, "api_key": AZURE_KEY, "api_version": "2024-10-21"}

# after — Bedrock OpenAI-compatible endpoint
{"model": "openai.gpt-oss-120b-1:0", "api_type": "openai",
 "base_url": f"https://bedrock-runtime.{region}.amazonaws.com/openai/v1",
 "api_key": provide_token(region=region)}
```

```bash
pip install autogen-agentchat aws-bedrock-token-generator
```

What changes:

- `api_type` `"azure"` → `"openai"`; `base_url` → the Bedrock endpoint.
- `api_key` → a short-term Bedrock token from your AWS credentials
  (`aws-bedrock-token-generator`); drop `api_version`.
- `model` → a Bedrock model ID
  ([../../../docs/model-mapping.md](../../../docs/model-mapping.md)).

Notes:

- AutoGen package naming varies by version (`pyautogen` / `autogen-agentchat` /
  AG2). The config shape shown here is the common OpenAI-compatible form; adjust
  imports to your installed distribution.
- You can also back AutoGen with `langchain-aws` where a LangChain model is
  accepted, mirroring the [LangChain guide](../langchain/README.md).

For a managed AWS agent runtime, consider Strands + AgentCore — see the
[section README](../README.md).
