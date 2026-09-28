# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
BEFORE — AutoGen agent backed by Azure OpenAI.

AutoGen (AG2 / autogen-agentchat) drives agents from an LLM config that points
at an Azure deployment. The Bedrock move swaps the client/config to Bedrock's
OpenAI-compatible endpoint. See ../after/bedrock_autogen.py.

Env vars:
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT
"""

import os

from autogen import AssistantAgent, UserProxyAgent

# Azure-style LLM config: api_type=azure + deployment + endpoint + api_version.
llm_config = {
    "config_list": [
        {
            "model": os.environ.get("AZURE_OPENAI_DEPLOYMENT", "my-gpt-4o"),
            "api_type": "azure",
            "base_url": os.environ["AZURE_OPENAI_ENDPOINT"],
            "api_key": os.environ["AZURE_OPENAI_API_KEY"],
            "api_version": "2024-10-21",
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
