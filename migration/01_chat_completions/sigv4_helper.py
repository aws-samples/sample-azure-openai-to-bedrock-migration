# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
sigv4_helper.py — SigV4 auth for the OpenAI Python SDK on Amazon Bedrock.

Use this when you are running on AWS compute (EC2, ECS, EKS, Lambda, SageMaker)
with an IAM role attached and want the OpenAI SDK to auto-sign requests using
the instance/task/pod role — no long-lived credentials required.

Usage:
    from sigv4_helper import BedrockSigV4Auth
    import boto3, httpx
    from openai import OpenAI

    session = boto3.Session()
    client = OpenAI(
        base_url=f"https://bedrock-runtime.{session.region_name}.amazonaws.com/openai/v1",
        api_key="not-used",  # required by SDK, ignored by SigV4 signer
        http_client=httpx.Client(auth=BedrockSigV4Auth(session, session.region_name)),
    )
    response = client.chat.completions.create(
        model="openai.gpt-oss-120b-1:0",
        messages=[{"role": "user", "content": "Hello"}],
    )

Credentials are resolved by boto3's standard chain:
    - Environment variables (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY)
    - ~/.aws/credentials profile
    - IAM instance role (EC2), task role (ECS), pod identity (EKS)
    - SageMaker execution role

No credential is stored in code or config; every request is signed with a
request-scoped SigV4 signature that expires after 5 minutes.
"""

from __future__ import annotations

import boto3
import httpx
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest


class BedrockSigV4Auth(httpx.Auth):
    """
    httpx.Auth implementation that signs each request with AWS SigV4.

    Works with any credential source boto3 supports — env vars locally,
    instance role on EC2, task role on ECS, pod identity on EKS.
    """

    def __init__(self, session: boto3.Session | None = None, region: str | None = None):
        self._session = session or boto3.Session()
        self._region = region or self._session.region_name or "us-east-1"
        self._creds = self._session.get_credentials()

    def auth_flow(self, request: httpx.Request):  # type: ignore[override]
        aws_req = AWSRequest(
            method=request.method,
            url=str(request.url),
            data=request.content,
            headers=dict(request.headers),
        )
        SigV4Auth(self._creds, "bedrock", self._region).add_auth(aws_req)
        for key, value in aws_req.headers.items():
            request.headers[key] = value
        yield request


# ── Convenience factory ────────────────────────────────────────────────────────

def make_bedrock_openai_client(
    region: str | None = None,
    profile: str | None = None,
    **openai_kwargs,
):
    """
    Return an OpenAI client pre-configured for Bedrock with SigV4 auth.

    Args:
        region:  AWS region (defaults to AWS_DEFAULT_REGION or us-east-1).
        profile: Named boto3 credential profile (for local dev; omit on EC2/ECS).
        **openai_kwargs: Extra kwargs forwarded to OpenAI() (e.g. timeout=60).

    Example:
        from sigv4_helper import make_bedrock_openai_client
        client = make_bedrock_openai_client()
        response = client.chat.completions.create(...)
    """
    from openai import OpenAI  # lazy import — not required if using class directly

    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    r = region or session.region_name or "us-east-1"
    return OpenAI(
        base_url=f"https://bedrock-runtime.{r}.amazonaws.com/openai/v1",
        api_key="not-used",  # nosec B105 - placeholder; required field, ignored when SigV4 signer is active
        http_client=httpx.Client(auth=BedrockSigV4Auth(session, r)),
        **openai_kwargs,
    )
