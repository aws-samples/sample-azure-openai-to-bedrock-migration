# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
BEFORE — Azure AI Agent Service researcher agent.

Sourced from Azure-Samples/contoso-creative-writer (MIT).
Illustrates the patterns az2br flags as HIGH severity:

  [agents]       AIProjectClient / azure.ai.projects  (lines 10, 12, 38)
  [auth]         DefaultAzureCredential                 (lines 11, 39)
  [tools]        BingGroundingTool + functions.json      (lines 12, 53)
  [completions]  azure_deployment="gpt-4"               (line 63)

The agent lifecycle is managed by Azure AI Agent Service:
  create_agent → create_thread → create_message → create_and_process_run
  → list_messages → delete_agent

There is no explicit tool-call loop here — Azure AI Agent Service runs the
loop server-side and returns the final assistant message.  The Bedrock
equivalent must drive that loop explicitly (see ../after/bedrock_converse_loop.py).

Env vars required:
    AZURE_LOCATION, AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP,
    AZURE_AI_PROJECT_NAME  (used to build the connection string)
"""

from __future__ import annotations

import json
import os

from azure.ai.inference.prompts import PromptTemplate
from azure.ai.projects import AIProjectClient                         # [HIGH agents]
from azure.ai.projects.models import BingGroundingTool                # [HIGH agents/tools]
from azure.identity import DefaultAzureCredential                     # [HIGH auth]
from tenacity import retry, retry_if_result, stop_after_attempt, wait_exponential


def _build_connection_string() -> str:
    """Assemble the Azure AI Foundry project connection string from env vars."""
    location = os.environ["AZURE_LOCATION"]
    sub      = os.environ["AZURE_SUBSCRIPTION_ID"]
    group    = os.environ["AZURE_RESOURCE_GROUP"]
    proj     = os.environ["AZURE_AI_PROJECT_NAME"]
    return f"{location}.api.azureml.ms;{sub};{group};{proj}"


def research(instructions: str, feedback: str = "No feedback") -> dict:
    """
    Run the researcher agent via Azure AI Agent Service.

    Returns a dict:
        {"web": [...], "entities": [], "news": []}
    """
    conn_str = _build_connection_string()

    project_client = AIProjectClient.from_connection_string(      # [HIGH agents]
        credential=DefaultAzureCredential(),                        # [HIGH auth]
        conn_str=conn_str,
    )

    # Load system prompt from researcher.prompty
    prompt_template = PromptTemplate.from_prompty(file_path="researcher.prompty")
    messages = prompt_template.create_messages(instructions=instructions, feedback=feedback)

    # Attach Bing Search as a grounding tool
    bing_connection = project_client.connections.get(connection_name="bing-connection")
    bing = BingGroundingTool(connection_id=bing_connection.id)     # [HIGH agents/tools]

    with project_client:
        # Azure AI Agent Service manages the full agentic loop server-side
        agent = project_client.agents.create_agent(
            model="gpt-4",                                          # [MEDIUM completions]
            name="researcher-agent",
            instructions=messages[0]["content"],
            tools=bing.definitions,
        )
        thread  = project_client.agents.create_thread()
        project_client.agents.create_message(
            thread_id=thread.id,
            role="user",
            content=instructions,
        )

        @retry(
            retry=retry_if_result(
                lambda run: run.status == "failed"
                and (run.last_error or {}).get("code") == "rate_limit_exceeded"
            ),
            wait=wait_exponential(multiplier=1, min=4, max=60),
            stop=stop_after_attempt(10),
        )
        def _run() -> object:
            return project_client.agents.create_and_process_run(
                thread_id=thread.id, assistant_id=agent.id
            )

        _run()

        project_client.agents.delete_agent(agent.id)

        raw_msgs = project_client.agents.list_messages(thread_id=thread.id)
        response_text = raw_msgs.data[0]["content"][0]["text"]["value"]
        parsed = json.loads(response_text)

    return {
        "web":      parsed.get("web", []),
        "entities": parsed.get("entities", []),
        "news":     parsed.get("news", []),
    }


if __name__ == "__main__":
    result = research("What are the latest camping trends for winter?")
    print(json.dumps(result, indent=2))
