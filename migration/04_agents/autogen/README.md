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
pip install "autogen[openai]" aws-bedrock-token-generator
```

Tested with `autogen` 0.14.1 (AG2 Classic) against `bedrock-runtime`.

What changes:

- `api_type` `"azure"` → `"openai"`; `base_url` → the Bedrock endpoint.
- `api_key` → a short-term Bedrock token from your AWS credentials
  (`aws-bedrock-token-generator`); drop `api_version`.
- `model` → a Bedrock model ID
  ([../../../docs/model-mapping.md](../../../docs/model-mapping.md)).

Notes:

- **Pick the right AutoGen package.** The code here (`from autogen import
  AssistantAgent`, `llm_config`, `config_list`) is the **AG2 Classic** API,
  published on PyPI as `autogen`. Other distributions will not run it:
  - `autogen-agentchat` / `pyautogen` (Microsoft AutoGen 0.4+) import as
    `autogen_agentchat` and use `model_client=` instead of `llm_config`.
    Microsoft has archived AutoGen and recommends
    [Microsoft Agent Framework](https://github.com/microsoft/agent-framework)
    for new work.
  - `ag2` 1.x imports as `ag2` and no longer ships the `autogen` namespace.
  AG2 Classic is in maintenance mode (security and critical fixes only), so
  treat this as a lift-and-shift path rather than a target for new development.
- **Frontier GPT models** (`openai.gpt-5.x`, `gpt-6`) reject a non-default
  `temperature`; the sample omits it for those models.
- **`gpt-oss` shows its reasoning inline.** On this Chat Completions path the
  reply from `openai.gpt-oss-120b-1:0` can start with a `<reasoning>...</reasoning>`
  block inside the message content. Strip it, or use a GPT-5.6 / GPT-6 model
  (for example `BEDROCK_MODEL_ID=global.openai.gpt-5.6-terra`) for clean output.
- You can also back AutoGen with `langchain-aws` where a LangChain model is
  accepted, mirroring the [LangChain guide](../langchain/README.md).

For a managed AWS agent runtime, consider Strands + AgentCore — see the
[section README](../README.md).
