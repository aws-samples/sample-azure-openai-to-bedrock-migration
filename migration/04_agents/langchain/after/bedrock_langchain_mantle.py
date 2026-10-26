# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
AFTER (alternative) — LangChain on Amazon Bedrock via ChatOpenAIMantle.

`ChatOpenAIMantle` (langchain-aws) talks to the `bedrock-mantle` endpoint with
the OpenAI SDK and, for GPT-5.x / GPT-6, defaults to the Responses API. Choose it
over `ChatBedrockConverse` (see bedrock_langchain.py) when you want:

  - the OpenAI Responses API surface you had on Azure, or
  - mantle-only server-side tools such as the built-in Web Search tool.

Use `ChatBedrockConverse` instead when you want one portable interface across
the whole Bedrock model catalog (Claude, Nova, Llama, ...) and SigV4 auth.

Auth: no static key needed. With no `bedrock_api_key` set, ChatOpenAIMantle
derives a short-term Bedrock API key from your normal AWS credentials (env,
profile, or IAM role) and refreshes it. The IAM identity needs
`bedrock-mantle:CreateInference` and `bedrock-mantle:CallWithBearerToken`; Web
Search adds `bedrock-websearch:InvokeSearch` and `bedrock-websearch:InvokeFetch`.

Model id: mantle takes the **bare** id (`openai.gpt-5.6-terra`). The `us.` /
`global.` inference-profile ids are bedrock-runtime concepts and 404 here.

Frontier GPT models accept only the default temperature, so none is set. Use
`reasoning_effort` / `verbosity` instead (docs/model-mapping.md).

Install (tested with langchain-aws 1.9.1):
    pip install "langchain-aws[openai]"

Env vars:
    AWS_REGION (default us-east-1; Web Search needs us-east-1, us-east-2, or
    us-west-2), BEDROCK_MODEL_ID (default openai.gpt-5.6-terra),
    plus standard AWS credentials.
"""

import os

from langchain_aws import ChatOpenAIMantle
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

REGION = os.environ.get("AWS_REGION", "us-east-1")
MODEL = os.environ.get("BEDROCK_MODEL_ID", "openai.gpt-5.6-terra")

llm = ChatOpenAIMantle(model=MODEL, region_name=REGION)


def _text(content) -> str:
    """Join the text blocks of a message (Responses API returns a list of blocks)."""
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")


def _citations(content) -> list[str]:
    """URLs the model cited, from the Responses API `annotations` on text blocks."""
    if isinstance(content, str):
        return []
    return [
        a["url"]
        for b in content
        if isinstance(b, dict)
        for a in (b.get("annotations") or [])
        if a.get("type") == "url_citation"
    ]


def run(prompt: str) -> str:
    return _text(llm.invoke(prompt).content)


def search(prompt: str) -> tuple[str, list[str]]:
    """Bedrock's built-in server-side Web Search (the Azure `web_search` replacement)."""
    web = llm.bind_tools(
        [{"type": "web_search", "search_context_size": "low", "external_web_access": False}]
    )
    reply = web.invoke(prompt)
    return _text(reply.content), _citations(reply.content)


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It is 21 C and sunny in {city}."


def call_tool(prompt: str) -> str:
    """One round of client-side function calling with the same LangChain tool API."""
    with_tools = llm.bind_tools([get_weather])
    messages = [HumanMessage(prompt)]
    reply = with_tools.invoke(messages)
    messages.append(reply)
    for call in reply.tool_calls:
        messages.append(ToolMessage(get_weather.invoke(call["args"]), tool_call_id=call["id"]))
    return _text(with_tools.invoke(messages).content) if reply.tool_calls else _text(reply.content)


if __name__ == "__main__":
    print(run("In one sentence, what is Amazon Bedrock?"))
    answer, urls = search("In one sentence, what is Amazon Bedrock? Use web search.")
    print(answer, urls[:2])
    print(call_tool("What's the weather in Paris?"))
