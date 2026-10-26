# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
AFTER — AutoGen agent backed by Amazon Bedrock (OpenAI-compatible endpoint).

AutoGen speaks the OpenAI config shape, so the cleanest Bedrock path is to point
its config_list at Bedrock's OpenAI-compatible endpoint (Path A). The agent
graph (AssistantAgent, UserProxyAgent, group chats) is unchanged — only the LLM
config changes.

Changes vs. ../before/azure_autogen.py:
  - api_type "azure" -> "openai"
  - base_url -> the Bedrock OpenAI-compatible endpoint
  - api_key -> a short-term Bedrock token (from AWS creds)
  - model -> a Bedrock model ID; drop api_version

Install (tested with autogen 0.14.1, "AG2 Classic"):
    pip install "autogen[openai]" aws-bedrock-token-generator

Which AutoGen? This sample uses the `autogen` import name, `llm_config`, and
`config_list`, which is the AG2 Classic API published on PyPI as `autogen`.
It does NOT work with `autogen-agentchat` (Microsoft AutoGen 0.4+, imported as
`autogen_agentchat`, now archived by Microsoft) or with `ag2` 1.x (imported as
`ag2`), neither of which provides `from autogen import AssistantAgent`.

Env vars:
    AWS_REGION (default us-east-1), BEDROCK_MODEL_ID (default openai.gpt-oss-120b-1:0),
    plus standard AWS credentials.
"""

import os

from autogen import AssistantAgent, UserProxyAgent
from aws_bedrock_token_generator import provide_token

REGION = os.environ.get("AWS_REGION", "us-east-1")
MODEL = os.environ.get("BEDROCK_MODEL_ID", "openai.gpt-oss-120b-1:0")

# Frontier GPT models (gpt-5.x / gpt-6) accept only the default temperature and
# reject an explicit value, so only set it for other models.
_FRONTIER = "openai.gpt-5" in MODEL.lower() or "openai.gpt-6" in MODEL.lower()

llm_config = {
    "config_list": [
        {
            "model": MODEL,
            "api_type": "openai",
            "base_url": f"https://bedrock-runtime.{REGION}.amazonaws.com/openai/v1",
            "api_key": provide_token(region=REGION),
        }
    ],
}
if not _FRONTIER:
    llm_config["temperature"] = 0.2

assistant = AssistantAgent(name="assistant", llm_config=llm_config)
user = UserProxyAgent(name="user", human_input_mode="NEVER", code_execution_config=False)


def run(prompt: str) -> str:
    result = user.initiate_chat(assistant, message=prompt, max_turns=1)
    return str(result.summary)


if __name__ == "__main__":
    print(run("In one sentence, what is Amazon Bedrock?"))
