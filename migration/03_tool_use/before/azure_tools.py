# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
BEFORE — Azure OpenAI function calling + a server-side web_search tool.

Shows both tool kinds in one file:
  - a client-side function tool (get_weather) — the classic tool-use loop, and
  - an Azure server-side tool ({"type": "web_search"}) that Azure runs for you.

Running az2br over this flags the function-calling shape, tool_choice, and the
web_search server tool. See ../after/ for the Bedrock Converse rewrite and
../server-tools.md for the server-tool mapping.

Env vars:
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT
"""

import json
import os

from openai import AzureOpenAI

client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version="2024-10-21",
)
DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "my-gpt-4o")

# Client-side function tool + an Azure server-side web_search tool.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {"type": "web_search"},
]


def get_weather(city: str) -> dict:
    """Pretend backing implementation."""
    return {"city": city, "temp_c": 18, "conditions": "cloudy"}


def run(user_msg: str) -> str:
    messages = [{"role": "user", "content": user_msg}]
    response = client.chat.completions.create(
        model=DEPLOYMENT, messages=messages, tools=TOOLS, tool_choice="auto"
    )
    msg = response.choices[0].message

    # If the model asked for a tool, run it and send the result back.
    if msg.tool_calls:
        messages.append(msg)
        for call in msg.tool_calls:
            if call.function.name == "get_weather":
                args = json.loads(call.function.arguments)
                result = get_weather(**args)
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)}
                )
        final = client.chat.completions.create(model=DEPLOYMENT, messages=messages)
        return final.choices[0].message.content
    return msg.content


if __name__ == "__main__":
    print(run("What's the weather in Seattle?"))
