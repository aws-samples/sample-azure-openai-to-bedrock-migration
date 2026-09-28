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

Install:
    pip install autogen-agentchat aws-bedrock-token-generator

Env vars:
    AWS_REGION (default us-east-1), BEDROCK_MODEL_ID (default openai.gpt-oss-120b-1:0),
    plus standard AWS credentials.
"""

import os

from autogen import AssistantAgent, UserProxyAgent
from aws_bedrock_token_generator import provide_token

REGION = os.environ.get("AWS_REGION", "us-east-1")

llm_config = {
    "config_list": [
        {
            "model": os.environ.get("BEDROCK_MODEL_ID", "openai.gpt-oss-120b-1:0"),
            "api_type": "openai",
            "base_url": f"https://bedrock-runtime.{REGION}.amazonaws.com/openai/v1",
            "api_key": provide_token(region=REGION),
        }
    ],
    "temperature": 0.2,
}

assistant = AssistantAgent(name="assistant", llm_config=llm_config)
user = UserProxyAgent(name="user", human_input_mode="NEVER", code_execution_config=False)


def run(prompt: str) -> str:
    result = user.initiate_chat(assistant, message=prompt, max_turns=1)
    return str(result.summary)


if __name__ == "__main__":
    print(run("In one sentence, what is Amazon Bedrock?"))
