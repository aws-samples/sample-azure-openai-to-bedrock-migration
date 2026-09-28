# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
providers.py — the network layer for record and replay.

Three providers, each turns a case (messages + params) into a normalized
response dict:

  - OpenAISource       record from Azure OpenAI or any OpenAI-compatible source
  - BedrockEndpoint    replay on Bedrock's OpenAI-compatible endpoint (Path A)
  - BedrockConverse    replay on the Bedrock Converse API (Path B)

`openai` and `boto3` are imported lazily inside each provider, so the rest of
the harness (cases, normalize, diff, and the offline test) has no hard
dependency on them. Configuration is read from environment variables so no
secrets live in code.
"""

from __future__ import annotations

import os
import sys
from typing import Any

from normalize import normalize_converse, normalize_openai


def _is_frontier_openai(model_id: str) -> bool:
    """True for OpenAI frontier models (GPT-5.x / GPT-6) on Bedrock.

    Matches ids like ``openai.gpt-5.6-terra`` and ``us.openai.gpt-6-astra``,
    with or without a ``us.`` / ``global.`` inference-profile prefix. These
    models accept only the default temperature and reject ``top_p``.
    """
    mid = (model_id or "").lower()
    return "openai.gpt-5" in mid or "openai.gpt-6" in mid


def _drop_unsupported_sampling(model_id: str, params: dict[str, Any]) -> dict[str, Any]:
    """Drop temperature/top_p for frontier OpenAI models (they only allow defaults).

    Returns a shallow copy with the offending keys removed; logs a one-line
    warning to stderr. gpt-oss and other models pass through unchanged. Mirrors
    LangChain ``ChatBedrockConverse``, which drops these and warns.
    """
    if not _is_frontier_openai(model_id):
        return params
    dropped = [k for k in ("temperature", "top_p") if k in params]
    if not dropped:
        return params
    cleaned = {k: v for k, v in params.items() if k not in ("temperature", "top_p")}
    print(
        f"warning: {model_id} does not support {', '.join(dropped)}; "
        f"dropping (only default values are allowed)",
        file=sys.stderr,
    )
    return cleaned


def _to_dict(obj: Any) -> dict[str, Any]:
    """Coerce an OpenAI SDK response object into a plain dict."""
    if isinstance(obj, dict):
        return obj
    for attr in ("model_dump", "to_dict", "dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            return fn()
    raise TypeError(f"cannot coerce {type(obj).__name__} to dict")


class OpenAISource:
    """Record from Azure OpenAI (default) or any OpenAI-compatible base_url.

    Env vars:
        Azure (default):
            AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT
            AZURE_OPENAI_API_VERSION (default: 2024-10-21)
        Generic OpenAI-compatible (set SOURCE_BASE_URL to switch):
            SOURCE_BASE_URL, SOURCE_API_KEY, SOURCE_MODEL
    """

    kind = "openai"

    def __init__(self) -> None:
        base_url = os.environ.get("SOURCE_BASE_URL")
        if base_url:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=base_url,
                api_key=os.environ.get("SOURCE_API_KEY", "unused"),
            )
            self._model = os.environ.get("SOURCE_MODEL", "gpt-4o")
        else:
            from openai import AzureOpenAI

            self._client = AzureOpenAI(
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_key=os.environ["AZURE_OPENAI_API_KEY"],  # nosec B105 — read from env, not hardcoded
                api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            )
            self._model = os.environ["AZURE_OPENAI_DEPLOYMENT"]

    def run(self, messages: list[dict[str, Any]], params: dict[str, Any]) -> dict[str, Any]:
        resp = self._client.chat.completions.create(
            model=self._model, messages=messages, **params
        )
        return normalize_openai(_to_dict(resp))


class BedrockEndpoint:
    """Replay on Bedrock's OpenAI-compatible endpoint (Path A).

    Env vars:
        AWS_REGION (default: us-east-1)
        BEDROCK_MODEL_ID (default: us.openai.gpt-5.6-terra)
        plus standard AWS credentials for the token generator.
    """

    kind = "openai"

    def __init__(self) -> None:
        from openai import OpenAI

        region = os.environ.get("AWS_REGION", "us-east-1")
        api_key = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        if not api_key:
            # Prefer a short-term token derived from AWS credentials. The token
            # generator honors the standard boto3 credential chain (which
            # includes AWS_PROFILE), so no profile argument is passed here.
            from aws_bedrock_token_generator import provide_token

            api_key = provide_token(region=region)
        self._client = OpenAI(
            base_url=f"https://bedrock-runtime.{region}.amazonaws.com/openai/v1",
            api_key=api_key,
        )
        self._model = os.environ.get("BEDROCK_MODEL_ID", "us.openai.gpt-5.6-terra")

    def run(self, messages: list[dict[str, Any]], params: dict[str, Any]) -> dict[str, Any]:
        params = _drop_unsupported_sampling(self._model, params)
        if _is_frontier_openai(self._model) and "max_tokens" in params:
            # Frontier GPT models use max_completion_tokens on the OpenAI endpoint.
            params = dict(params)
            params["max_completion_tokens"] = params.pop("max_tokens")
        resp = self._client.chat.completions.create(
            model=self._model, messages=messages, **params
        )
        return normalize_openai(_to_dict(resp))


class BedrockConverse:
    """Replay on the Bedrock Converse API (Path B).

    Env vars:
        AWS_REGION (default: us-east-1)
        BEDROCK_MODEL_ID (default: us.openai.gpt-5.6-terra)
        plus standard AWS credentials.
    """

    kind = "converse"

    def __init__(self) -> None:
        import boto3

        region = os.environ.get("AWS_REGION", "us-east-1")
        profile = os.environ.get("AWS_PROFILE")
        session = boto3.Session(region_name=region,
                                **({"profile_name": profile} if profile else {}))
        self._client = session.client("bedrock-runtime")
        self._model = os.environ.get("BEDROCK_MODEL_ID", "us.openai.gpt-5.6-terra")

    @staticmethod
    def _to_converse(messages: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
        """Split OpenAI-style messages into Converse system + messages."""
        system: list[dict[str, Any]] = []
        converse_msgs: list[dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            content = m.get("content", "")
            if role == "system":
                system.append({"text": content})
            else:
                converse_msgs.append({"role": role, "content": [{"text": content}]})
        return system, converse_msgs

    @staticmethod
    def _to_inference_config(params: dict[str, Any]) -> dict[str, Any]:
        """Map OpenAI-style params onto Converse inferenceConfig."""
        cfg: dict[str, Any] = {}
        if "temperature" in params:
            cfg["temperature"] = params["temperature"]
        if "max_tokens" in params:
            cfg["maxTokens"] = params["max_tokens"]
        if "top_p" in params:
            cfg["topP"] = params["top_p"]
        if "stop" in params:
            cfg["stopSequences"] = params["stop"]
        return cfg

    def run(self, messages: list[dict[str, Any]], params: dict[str, Any]) -> dict[str, Any]:
        params = _drop_unsupported_sampling(self._model, params)
        system, converse_msgs = self._to_converse(messages)
        kwargs: dict[str, Any] = {"modelId": self._model, "messages": converse_msgs}
        if system:
            kwargs["system"] = system
        cfg = self._to_inference_config(params)
        if cfg:
            kwargs["inferenceConfig"] = cfg
        resp = self._client.converse(**kwargs)
        return normalize_converse(resp)


# Registry used by the CLI --provider flag.
PROVIDERS = {
    "source": OpenAISource,
    "bedrock-endpoint": BedrockEndpoint,
    "bedrock-converse": BedrockConverse,
}


def get_provider(name: str):
    try:
        return PROVIDERS[name]()
    except KeyError:
        raise ValueError(
            f"unknown provider {name!r}; choose from {sorted(PROVIDERS)}"
        ) from None
