# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
AFTER — Amazon Bedrock Managed Agents (BMA, public preview) researcher agent.

A fourth migration target for ../before/azure_researcher.py. Unlike the other
three, you do not write or own the agent loop: BMA runs it as a service, the
way Azure AI Agent Service did. What changes is *where tools run*. Azure ran
Bing grounding for you; BMA runs your tools in an execution environment YOU
provide. Here that is your own machine, with the Codex exec server attached and
the three search tools served by ./bma_research_mcp.py.

PREVIEW: APIs and IAM action names can change. Read the caveats in
../README.md before building on this. For a plain function-calling research
task the Responses API loop (bedrock_responses_api_loop.py) is simpler; BMA
earns its keep on longer, stateful work that needs files, shell and skills.

Concept map (Azure AI Agent Service -> BMA):

  create_agent(model, instructions, tools)  agent{} block of POST …/agents/sessions
  create_thread()                           the session (same call)
  create_message(thread_id, ...)            POST …/events  agent.session.input.message
  create_and_process_run()                  a turn: async; wait for the new final answer
  list_messages()                           GET …/items (cursor-paginated)
  delete_agent()/delete_thread()            DELETE …/sessions/{id}
  BingGroundingTool / functions.json        stdio MCP server in the execution environment
  DefaultAzureCredential                    AWS SigV4, service "bedrock-mantle"
  retry on rate_limit_exceeded (tenacity)   backoff + jitter on HTTP 429

Same contract as the other targets: research() returns
{"web": [...], "entities": [...], "news": [...]}.

Run it (needs the one-time IAM stack in ../bma_iam_cdk and the Codex CLI, version
0.154.0 or later, on PATH or in CODEX_BIN):

    pip install boto3 requests
    AWS_PROFILE=<profile that can assume BmaClientRole> python bedrock_managed_agents_researcher.py

Env vars:
    AWS_PROFILE      Profile used for API calls AND handed to `codex exec-server`.
                     Use the least-privilege BmaClientRole, not an admin profile.
    AWS_REGION       us-east-1 (default), us-east-2 or us-west-2 (preview Regions).
    BMA_MODEL        default openai.gpt-5.6-luna — the BARE model id. Cross-Region
                     inference-profile ids (us.…) are not supported by the preview.
    BMA_ROLE_ARN     Session role BMA assumes (default: the role from ../bma_iam_cdk).
    BMA_ATTACH       "managed" (default: this script starts the exec server) or
                     "external" (you start `codex exec-server` yourself).
    CODEX_BIN        Codex executable (default: codex).
    BMA_WORKSPACE    Workspace directory (default: a fresh temp directory).
    BMA_TURN_TIMEOUT Seconds to wait for a turn (default 300).
    TAVILY_API_KEY   Optional; forwarded to the MCP server by NAME only.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Iterator

REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
ENDPOINT = f"https://bedrock-mantle.{REGION}.api.aws"
MODEL = os.environ.get("BMA_MODEL", "openai.gpt-5.6-luna")
SESSION_ROLE_NAME = "BedrockManagedAgentsPreviewInferenceServiceRole"
MCP_SERVER = Path(__file__).with_name("bma_research_mcp.py").resolve()
MCP_LABEL = "research_search"
MCP_TOOLS = ["find_information", "find_entities", "find_news"]

INSTRUCTIONS = """\
You are a researcher agent. Use the search tools (find_information, find_entities,
find_news) to gather sources, then return ONLY a single JSON object and nothing else.

Rules:
- Do NOT write any prose, explanation, or <thinking> block.
- Do NOT wrap the JSON in markdown code fences.
- Your entire response must be exactly one JSON object with this structure:

{"web": [{"url": "...", "name": "...", "description": "..."}],
 "entities": [],
 "news": []}

Return 3-5 sources in "web". Begin your response with { and end with }.
"""


# ── Pure helpers (no network; unit-tested offline) ────────────────────────────

def build_session_body(
    *,
    model: str,
    instructions: str,
    role_arn: str,
    workspace: str,
    mcp_command: str,
    mcp_script: str,
    forward_env: list[str] | None = None,
) -> dict[str, Any]:
    """The POST /openai/v1/agents/sessions body.

    `agent` replaces Azure's create_agent; the session itself replaces the
    thread. The tool is an MCP server the exec server launches in the execution
    environment, so `command`, `args` and `cwd` must be ABSOLUTE paths there.
    `env_vars` names variables to inherit — never put a secret value in `env`.
    """
    return {
        "role_arn": role_arn,
        "agent": {
            "model": model,
            "instructions": instructions,
            "tools": [
                {
                    "type": "mcp",
                    "server_label": MCP_LABEL,
                    "transport": {
                        "type": "stdio",
                        "command": mcp_command,
                        "args": [mcp_script],
                        "cwd": workspace,
                        "env": {},
                        "env_vars": list(forward_env or []),
                    },
                    "allowed_tools": list(MCP_TOOLS),
                }
            ],
        },
        "environment": {
            "type": "self_hosted",
            "workspace_directory": workspace,
            "capability_directories": [f"{workspace}/skills"],
        },
        "stream": False,
    }


def final_answer(items: list[dict], exclude_turns: set[str]) -> tuple[str, str] | None:
    """Return (turn_id, text) of the newest assistant final answer from a turn
    not in `exclude_turns`, else None.

    Why not just wait for session.status == "idle"? Right after you submit a
    message the session can still read "idle" for an instant, so status alone
    can return before the turn has started. A *new* assistant message with
    phase "final_answer" is the unambiguous completion signal.
    """
    found: tuple[str, str] | None = None
    for item in items:
        if (
            item.get("type") == "message"
            and item.get("role") == "assistant"
            and item.get("phase", "final_answer") == "final_answer"
            and item.get("turn_id") not in exclude_turns
        ):
            text = "".join(
                part.get("text", "")
                for part in item.get("content", [])
                if part.get("type") == "output_text"
            )
            found = (item["turn_id"], text)
    return found


def extract_json_object(text: str) -> dict:
    """Parse the model's JSON, tolerating stray prose or code fences."""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}") + 1
        if start == -1 or end <= start:
            raise ValueError(f"researcher returned no JSON object; got: {text[:200]!r}")
        parsed = json.loads(text[start:end])
    if not isinstance(parsed, dict):
        raise ValueError(f"researcher JSON must be an object; got {type(parsed).__name__}")
    return parsed


def normalize(parsed: dict) -> dict:
    return {
        "web": parsed.get("web", []),
        "entities": parsed.get("entities", []),
        "news": parsed.get("news", []),
    }


def parse_failure(event: dict) -> str | None:
    """Return a human-readable message if this SSE event reports a failed turn.

    A failed turn is ONLY visible in the event stream. Observed live: when the
    session role is not allowed to invoke the model, the session still reads
    status "idle" with error null and the items hold just the user message — a
    client that only polls waits until its timeout with no explanation. The
    stream carries an `error` event and `agent.session.turn.failed` with the
    real reason (for example the missing bedrock-mantle:CreateInference).
    """
    kind = event.get("type")
    if kind == "error":
        err = event.get("error") or {}
    elif kind == "agent.session.turn.failed":
        err = (event.get("turn") or {}).get("error") or {}
    else:
        return None
    return f"{err.get('code', 'error')}: {err.get('message', 'turn failed')}"


# ── SigV4 REST client ─────────────────────────────────────────────────────────

class BmaApi:
    """Minimal SigV4 client for the BMA preview REST API.

    Azure used DefaultAzureCredential; here credentials come from the standard
    AWS chain (profile, env, instance/task role) and every request is signed for
    service "bedrock-mantle". No API key. boto3/requests are imported lazily so
    the pure helpers above stay stdlib-only.
    """

    RETRY_STATUS = {429}                 # throttled before processing: safe for any method
    RETRY_READ_STATUS = {500, 502, 503, 504}

    def __init__(self, region: str = REGION, endpoint: str = ENDPOINT, max_attempts: int = 6):
        import boto3

        self.region, self.endpoint, self.max_attempts = region, endpoint.rstrip("/"), max_attempts
        self._boto = boto3.Session()

    def account_id(self) -> str:
        return self._boto.client("sts", region_name=self.region).get_caller_identity()["Account"]

    def request(self, method: str, path: str, body: dict | None = None):
        import requests
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest

        if not path.startswith("/") or path.startswith("//"):
            raise ValueError("path must start with exactly one slash")
        data = json.dumps(body).encode() if body is not None else None
        for attempt in range(1, self.max_attempts + 1):
            creds = self._boto.get_credentials()
            if creds is None:
                raise RuntimeError("No AWS credentials found; set AWS_PROFILE.")
            url = self.endpoint + path
            headers = {"Content-Type": "application/json"} if data else {}
            signed = AWSRequest(method=method, url=url, data=data, headers=headers)
            SigV4Auth(creds.get_frozen_credentials(), "bedrock-mantle", self.region).add_auth(signed)
            resp = requests.request(method, url, data=data, headers=dict(signed.headers), timeout=(10, 90))
            retry = resp.status_code in self.RETRY_STATUS or (
                method == "GET" and resp.status_code in self.RETRY_READ_STATUS
            )
            if retry and attempt < self.max_attempts:
                # Exponential backoff with jitter (replaces tenacity on rate_limit_exceeded).
                time.sleep(min(30.0, 2 ** attempt) * random.uniform(0.5, 1.0))
                continue
            if not resp.ok:
                raise RuntimeError(f"BMA {method} {path} -> HTTP {resp.status_code}: {resp.text[:500]}")
            return resp
        raise AssertionError("unreachable")

    def json(self, method: str, path: str, body: dict | None = None) -> dict:
        resp = self.request(method, path, body)
        return resp.json() if resp.content else {}

    def stream(self, path: str):
        """Open the session's server-sent-event stream (returns once headers arrive,
        i.e. once the stream is established). Caller must close() it."""
        import requests
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest

        url = self.endpoint + path
        signed = AWSRequest(method="GET", url=url, headers={"Accept": "text/event-stream"})
        SigV4Auth(self._boto.get_credentials().get_frozen_credentials(), "bedrock-mantle", self.region).add_auth(signed)
        resp = requests.get(url, headers=dict(signed.headers), stream=True, timeout=(10, 300))
        if not resp.ok:
            resp.close()
            raise RuntimeError(f"BMA GET {path} -> HTTP {resp.status_code}")
        return resp


class _TurnWatcher:
    """Background reader of the SSE stream that records the first turn failure.

    Opened BEFORE the message is submitted (an idle session emits nothing, and
    events are not replayed), closed when the turn is done.
    """

    def __init__(self, api: BmaApi, path: str):
        self.failure: str | None = None
        self._resp = api.stream(path)
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "_TurnWatcher":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        try:
            self._resp.close()
        except Exception:
            pass
        self._thread.join(timeout=2)

    def _run(self) -> None:
        try:
            for line in self._resp.iter_lines(chunk_size=1, decode_unicode=True):
                if line and line.startswith("data:"):
                    try:
                        message = parse_failure(json.loads(line[5:]))
                    except json.JSONDecodeError:
                        continue
                    if message and not self.failure:
                        self.failure = message
        except Exception:
            pass  # closed by __exit__, or a dropped connection: polling still covers success


# ── Session: Azure agent + thread + run, as one object ────────────────────────

class ManagedAgentSession:
    """create session -> attach exec server -> send turns -> delete.

    Use as a context manager so the session and the exec server are always
    cleaned up (Azure needed delete_agent for the same reason).
    """

    def __init__(
        self,
        api: BmaApi | None = None,
        *,
        instructions: str = INSTRUCTIONS,
        model: str = MODEL,
        turn_timeout: float | None = None,
        poll_interval: float = 2.0,
    ):
        self.api = api or BmaApi()
        self.instructions, self.model = instructions, model
        self.turn_timeout = turn_timeout or float(os.environ.get("BMA_TURN_TIMEOUT", "300"))
        self.poll_interval = poll_interval
        self.session_id: str | None = None
        self._seen_turns: set[str] = set()
        self._codex: subprocess.Popen | None = None
        self._codex_log: str | None = None
        self._tmp_workspace: str | None = None

    # -- lifecycle ------------------------------------------------------------
    def __enter__(self) -> "ManagedAgentSession":
        try:
            self._create()
            if os.environ.get("BMA_ATTACH", "managed") == "managed":
                self._attach_exec_server()
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _create(self) -> None:
        workspace = os.environ.get("BMA_WORKSPACE")
        if not workspace:
            workspace = self._tmp_workspace = tempfile.mkdtemp(prefix="bma-researcher-")
        workspace = str(Path(workspace).resolve())
        Path(workspace, "skills").mkdir(parents=True, exist_ok=True)
        role_arn = os.environ.get("BMA_ROLE_ARN") or (
            f"arn:aws:iam::{self.api.account_id()}:role/{SESSION_ROLE_NAME}"
        )
        body = build_session_body(
            model=self.model,
            instructions=self.instructions,
            role_arn=role_arn,
            workspace=workspace,
            mcp_command=sys.executable,
            mcp_script=str(MCP_SERVER),
            forward_env=[k for k in ("TAVILY_API_KEY",) if os.environ.get(k)],
        )
        session = self.api.json("POST", "/openai/v1/agents/sessions", body)
        self.session_id = session["id"]
        self.environment_id = session["environment"]["id"]

    def _attach_exec_server(self, ready_timeout: float = 45.0) -> None:
        """Start `codex exec-server` and wait for its connection acknowledgement.

        A running process is not proof of a connected environment — wait for the
        "exec-server connected" log line. Codex signs with the same AWS profile.
        """
        codex = shutil.which(os.environ.get("CODEX_BIN", "codex"))
        if not codex:
            raise RuntimeError("Codex CLI not found (need 0.154.0+). Set CODEX_BIN or BMA_ATTACH=external.")
        cmd = [
            codex, "exec-server",
            "--remote", f"{self.api.endpoint}/openai/v1",
            "--environment-id", self.environment_id,
            "--remote-transport", "direct",
            "--aws-sigv4",
            "--aws-service", "bedrock-mantle",
            "--aws-region", self.api.region,
        ]
        if os.environ.get("AWS_PROFILE"):
            cmd += ["--aws-profile", os.environ["AWS_PROFILE"]]
        env = dict(os.environ)
        env.setdefault("RUST_LOG", "codex_exec_server=info,codex_cli=info")
        fd, self._codex_log = tempfile.mkstemp(prefix="bma-codex-", suffix=".log")
        with os.fdopen(fd, "w") as log:
            self._codex = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)

        deadline = time.time() + ready_timeout
        while time.time() < deadline:
            text = Path(self._codex_log).read_text(errors="replace")
            if "exec-server connected" in text:
                return
            if self._codex.poll() is not None:
                raise RuntimeError(f"codex exec-server exited early:\n{text[-800:]}")
            time.sleep(0.5)
        raise RuntimeError("codex exec-server did not report a connection in time")

    def close(self) -> None:
        if self.session_id:
            try:
                self.api.json("DELETE", f"/openai/v1/agents/sessions/{self.session_id}")
            except Exception as exc:  # cleanup must not mask the real error
                print(f"warning: could not delete session {self.session_id}: {exc}", file=sys.stderr)
            self.session_id = None
        if self._codex and self._codex.poll() is None:
            self._codex.terminate()
            try:
                self._codex.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._codex.kill()
        if self._codex_log:
            Path(self._codex_log).unlink(missing_ok=True)
            self._codex_log = None
        if self._tmp_workspace:
            shutil.rmtree(self._tmp_workspace, ignore_errors=True)
            self._tmp_workspace = None

    # -- turns ----------------------------------------------------------------
    def items(self) -> Iterator[dict]:
        """All items, oldest first, following the opaque `after` cursor."""
        from urllib.parse import urlencode

        after = None
        while True:
            query = {"limit": 100, "order": "asc"}
            if after:
                query["after"] = after
            page = self.api.json("GET", f"/openai/v1/agents/sessions/{self.session_id}/items?{urlencode(query)}")
            yield from page.get("data", [])
            if not page.get("has_more"):
                return
            after = page["last_id"]

    def send(self, text: str) -> str:
        """Submit one user message and return the agent's final answer text.

        Azure: create_message + create_and_process_run + list_messages. BMA:
        POST an input event (202, empty body), then wait for the turn's final
        answer. Submissions are NOT retried after an ambiguous failure — check
        the session's items first so a turn is never duplicated.
        """
        event = {
            "events": [{
                "type": "agent.session.input.message",
                "input": [{"role": "user", "content": [{"type": "input_text", "text": text}]}],
            }]
        }
        base = f"/openai/v1/agents/sessions/{self.session_id}"
        with _TurnWatcher(self.api, f"{base}/events") as watcher:
            self.api.json("POST", f"{base}/events", event)

            deadline = time.time() + self.turn_timeout
            while time.time() < deadline:
                session = self.api.json("GET", base)
                if session.get("status") == "failed":
                    raise RuntimeError(f"BMA session failed: {session.get('error')}")
                answer = final_answer(list(self.items()), self._seen_turns)
                if answer and session.get("status") == "idle":
                    self._seen_turns.add(answer[0])
                    return answer[1]
                if watcher.failure:
                    raise RuntimeError(f"BMA turn failed — {watcher.failure}")
                time.sleep(self.poll_interval)
        raise TimeoutError(f"no final answer within {self.turn_timeout:.0f}s (session {self.session_id})")


# ── Public entry point — same contract as the other targets' research() ──────

def research(instructions: str, feedback: str = "No feedback") -> dict:
    """Run the researcher agent via Bedrock Managed Agents.

    Drop-in replacement for the Azure/Converse/Responses versions:
        {"web": [...], "entities": [...], "news": [...]}
    """
    user_content = instructions
    if feedback and feedback.lower() != "no feedback":
        user_content += f"\n\nFeedback to incorporate: {feedback}"
    with ManagedAgentSession() as session:
        return normalize(extract_json_object(session.send(user_content)))


if __name__ == "__main__":
    # Two turns on ONE session: this is what an Azure thread gave you.
    with ManagedAgentSession() as session:
        first = normalize(extract_json_object(session.send("What are the latest camping trends for winter?")))
        print(json.dumps(first, indent=2))
        follow_up = session.send(
            "Now narrow that to lightweight tents. Same JSON shape, replacing the previous result."
        )
        print(json.dumps(normalize(extract_json_object(follow_up)), indent=2))
