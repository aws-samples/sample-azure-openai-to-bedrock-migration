# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
AFTER — Path C: keep the OpenAI Responses API, point it at Amazon Bedrock (mantle).

Use this path if you're already calling `client.responses.create(...)` today
(or you're on Azure OpenAI Assistants, which maps conceptually to the Responses
API) and want to keep that surface without adopting Converse.

  - If you're on Chat Completions, see `bedrock_openai_endpoint.py` (Path A).
  - If you want a model-agnostic interface across the full Bedrock catalog,
    see `bedrock_converse.py` (Path B).

Why the `bedrock-mantle` endpoint here (not `bedrock-runtime`)? The Responses
API is available on both, but this example showcases the **mantle-only**
capabilities: server-side tools (Web Search, code interpreter), background
inference, and Projects/Workspaces. When the Responses API is called on
`bedrock-runtime`, server-side tools — including Web Search — are not available.

  1. `AzureOpenAI(...)`            -> `OpenAI(base_url=..., api_key=...)`
  2. Azure endpoint + API version -> the Bedrock mantle OpenAI-compatible endpoint
                                      https://bedrock-mantle.{region}.api.aws/openai/v1
  3. Azure key / deployment name  -> a Bedrock API key (bearer token) + a Bedrock model id

Auth: instead of storing a static key, generate a short-term Bedrock API key
from your normal AWS credentials with `aws-bedrock-token-generator`. The token
is derived from whatever credential chain boto3/AWS already uses (env vars,
profile, IAM role) and is valid for up to 12 hours. On mantle, the IAM identity
behind the request needs the `bedrock-mantle:CreateInference` action (instead of
`bedrock:InvokeModel`, which authorizes inference on `bedrock-runtime`).

Install:
    pip install openai aws-bedrock-token-generator

Env vars this expects (AWS):
    AWS_REGION            e.g. us-east-1  (or rely on your default AWS config)
    plus standard AWS credentials (profile, env vars, or IAM role)

Model id note: the AWS Web Search examples use the bare model id
(`openai.gpt-5.6-terra`) on the mantle endpoint, so that's what this file uses.
Verified live (us-east-1): mantle accepts the bare id and rejects the
cross-Region inference-profile id `us.openai.gpt-5.6-terra` with a 404
("model does not exist") — the `us.` prefix is a `bedrock-runtime` concept, not
a mantle one. On `bedrock-runtime` you'd use the profile id instead. See
../../docs/model-mapping.md.

IAM note: authenticating with a Bedrock API key / bearer token (what
`provide_token` + the OpenAI SDK do) requires BOTH
`bedrock-mantle:CreateInference` and `bedrock-mantle:CallWithBearerToken`.
Granting only `CreateInference` fails with a 403 ("not authorized to perform:
bedrock-mantle:CallWithBearerToken"). See the least-privilege example at
../../migration/04_agents/azure_ai_agent_service/least-privilege-role.json.

Web Search / Region note: Web Search is regional. At the time of writing it is
supported in the commercial US Regions us-east-1, us-east-2, and us-west-2 (and
GovCloud US-West), on the GPT-5.4/5.5/5.6 families. Confirm current availability
in the Bedrock User Guide.
"""

import os

from aws_bedrock_token_generator import provide_token
from openai import OpenAI

REGION = os.environ.get("AWS_REGION", "us-east-1")

# provide_token() mints a short-term Bedrock API key from your current AWS
# credentials, so no static key is stored in the app or environment.
client = OpenAI(
    base_url=f"https://bedrock-mantle.{REGION}.api.aws/openai/v1",
    api_key=provide_token(region=REGION),
)

# Bedrock model id — the AWS mantle examples use the bare id (no `us.` prefix).
MODEL = os.environ.get("BEDROCK_MODEL_ID", "openai.gpt-5.6-terra")


def respond_once() -> str:
    """Single Responses API call — the same call shape you use on OpenAI/Azure.

    This is the drop-in: keep `client.responses.create(...)`, change only the
    `base_url` (to mantle), the `api_key` (to a Bedrock token), and the `model`
    (to a Bedrock model id).
    """
    response = client.responses.create(
        model=MODEL,
        input="What is Amazon Bedrock? Answer in one sentence.",
    )
    return response.output_text


def respond_with_web_search() -> str:
    """Responses API call with the built-in, server-side Web Search tool.

    Web Search is a mantle-only server tool: the model decides when it needs
    current information, issues search queries against the Amazon-maintained web
    index, and grounds its answer with citations. This is the "near drop-in"
    the guide references — the same tool object you'd add on Azure/OpenAI.
    """
    response = client.responses.create(
        model=MODEL,
        input="What's the latest AWS Lambda cold start guidance?",
        tools=[{
            "type": "web_search",
            "search_context_size": "low",
            # Keep retrieval inside the AWS boundary. `external_web_access`
            # defaults to True (matching OpenAI), but AmazonBedrockFullAccess
            # grants only bedrock-websearch:InvokeSearch + :InvokeFetch, NOT
            # bedrock-websearch:ExternalWebAccess. If you leave the default True
            # without that permission, every live Fetch fails its backend
            # authorization check *before* the cache is read — silently: the
            # call still returns HTTP 200 with url_citation annotations from
            # Search-only observations, so the failure is easy to miss (the
            # denied InvokeFetch is only visible in CloudTrail). Setting False
            # avoids that failure and needs no extra IAM permission; Search and
            # cached Fetch both still work.
            "external_web_access": False,
        }],
    )
    return response.output_text


if __name__ == "__main__":
    print("== responses.create (plain) ==")
    print(respond_once())
    print("\n== responses.create (web_search) ==")
    print(respond_with_web_search())
