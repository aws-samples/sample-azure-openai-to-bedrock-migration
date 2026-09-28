# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
AFTER — Bedrock Converse tool use (client-side function calling).

Ports the client-side get_weather function from ../before/azure_tools.py to the
Converse toolConfig shape:

  tools=[{"type":"function","function":{... "parameters": schema}}]
    ->  toolConfig={"tools":[{"toolSpec":{"name","description",
                                          "inputSchema":{"json": schema}}}]}
  tool_choice="auto"  ->  toolChoice={"auto":{}}

The loop also changes shape:
  - the model returns a `toolUse` content block (not message.tool_calls),
  - you append a `toolResult` content block (not a {"role":"tool"} message),
  - then call converse() again.

The Azure `web_search` server tool is intentionally NOT ported here — it moves
to the OpenAI Responses API path (see ../server-tools.md), not Converse.

Env vars:
    AWS_REGION (default: us-east-1), BEDROCK_MODEL_ID (default: openai.gpt-oss-120b-1:0),
    plus standard AWS credentials.
"""

import os

import boto3

client = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))
MODEL = os.environ.get("BEDROCK_MODEL_ID", "openai.gpt-oss-120b-1:0")

TOOL_CONFIG = {
    "tools": [
        {
            "toolSpec": {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    }
                },
            }
        }
    ],
    "toolChoice": {"auto": {}},
}


def get_weather(city: str) -> dict:
    """Pretend backing implementation (same as the Azure version)."""
    return {"city": city, "temp_c": 18, "conditions": "cloudy"}


def run(user_msg: str) -> str:
    messages = [{"role": "user", "content": [{"text": user_msg}]}]
    response = client.converse(modelId=MODEL, messages=messages, toolConfig=TOOL_CONFIG)

    # The model may ask for a tool via a toolUse content block.
    out_message = response["output"]["message"]
    tool_uses = [b["toolUse"] for b in out_message["content"] if "toolUse" in b]

    if response.get("stopReason") == "tool_use" and tool_uses:
        messages.append(out_message)  # echo the assistant turn back
        tool_results = []
        for tu in tool_uses:
            if tu["name"] == "get_weather":
                result = get_weather(**tu["input"])
                tool_results.append(
                    {
                        "toolResult": {
                            "toolUseId": tu["toolUseId"],
                            "content": [{"json": result}],
                        }
                    }
                )
        messages.append({"role": "user", "content": tool_results})
        final = client.converse(modelId=MODEL, messages=messages, toolConfig=TOOL_CONFIG)
        return _reply_text(final["output"]["message"])

    return _reply_text(out_message)


def _reply_text(message: dict) -> str:
    """Join all text blocks in a Converse message.

    A reasoning model (e.g. OpenAI gpt-oss) may emit a reasoningContent block
    before the text block, so scan rather than assume content[0].
    """
    return "".join(b["text"] for b in message.get("content", []) if "text" in b)


if __name__ == "__main__":
    print(run("What's the weather in Seattle?"))
