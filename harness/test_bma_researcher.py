#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
test_bma_researcher.py — offline tests for the Bedrock Managed Agents example.

Pure stdlib, no network, no AWS credentials. Covers:
  * the STDIO MCP server that replaces the Azure researcher's function tools
    (protocol, schema parity with functions.json, input validation, and a real
    subprocess round trip over stdin/stdout, exactly as the exec server drives it);
  * the Managed Agents client logic: session request shape, final-answer
    detection, the "idle before the turn starts" race, cursor pagination, and
    turn failures that are visible only in the event stream.
The fixtures mirror payloads captured from the live preview (ids replaced).

Run:  python harness/test_bma_researcher.py   (exit 0 = all passed)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AFTER = ROOT / "migration/04_agents/azure_ai_agent_service/after"
sys.path.insert(0, str(AFTER))

import bma_research_mcp as mcp  # noqa: E402

EXPECTED_TOOLS = {"find_information", "find_entities", "find_news"}


def _call(name: str, arguments) -> dict:
    return mcp.handle(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
         "params": {"name": name, "arguments": arguments}}
    )


class TestProtocol(unittest.TestCase):
    def test_initialize_advertises_tools_capability(self):
        r = mcp.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        self.assertEqual(r["id"], 1)
        self.assertIn("tools", r["result"]["capabilities"])
        self.assertEqual(r["result"]["protocolVersion"], mcp.PROTOCOL_VERSION)

    def test_notifications_get_no_response(self):
        self.assertIsNone(mcp.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}))

    def test_unknown_method_is_method_not_found(self):
        r = mcp.handle({"jsonrpc": "2.0", "id": 7, "method": "nope"})
        self.assertEqual(r["error"]["code"], -32601)

    def test_ping(self):
        self.assertEqual(mcp.handle({"jsonrpc": "2.0", "id": 2, "method": "ping"})["result"], {})


class TestToolList(unittest.TestCase):
    def setUp(self):
        self.tools = mcp.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"]["tools"]

    def test_same_three_tools_as_the_azure_researcher(self):
        self.assertEqual({t["name"] for t in self.tools}, EXPECTED_TOOLS)

    def test_schema_matches_functions_json_inner_schema(self):
        # Only the envelope changes (parameters -> inputSchema); the JSON Schema
        # body is the one the Azure researcher's functions.json declared.
        for t in self.tools:
            schema = t["inputSchema"]
            self.assertEqual(schema["type"], "object")
            self.assertEqual(schema["required"], ["query"])
            self.assertEqual(set(schema["properties"]), {"query", "market"})
            self.assertFalse(schema["additionalProperties"])
            self.assertTrue(t["description"])


class TestToolCalls(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.pop("TAVILY_API_KEY", None)  # force the offline stub

    def tearDown(self):
        if self._saved is not None:
            os.environ["TAVILY_API_KEY"] = self._saved

    def test_each_tool_returns_url_name_description(self):
        for name in sorted(EXPECTED_TOOLS):
            r = _call(name, {"query": "winter camping tents"})
            payload = json.loads(r["result"]["content"][0]["text"])
            self.assertEqual(len(payload["results"]), 3, name)
            for item in payload["results"]:
                self.assertEqual(set(item), {"url", "name", "description"})

    def test_rejects_missing_or_blank_query(self):
        for args in ({}, {"query": ""}, {"query": "   "}, {"query": 5}, None, "text"):
            self.assertEqual(_call("find_information", args)["error"]["code"], -32602, args)

    def test_rejects_oversized_query(self):
        too_long = "x" * (mcp.MAX_QUERY_CHARS + 1)
        self.assertEqual(_call("find_information", {"query": too_long})["error"]["code"], -32602)

    def test_rejects_non_string_market(self):
        self.assertEqual(_call("find_news", {"query": "q", "market": 1})["error"]["code"], -32602)

    def test_rejects_unknown_tool(self):
        self.assertEqual(_call("drop_tables", {"query": "q"})["error"]["code"], -32602)


class TestSubprocess(unittest.TestCase):
    """Drive the server the way the exec server does: newline-delimited JSON-RPC."""

    def test_roundtrip_over_stdio(self):
        env = {k: v for k, v in os.environ.items() if k != "TAVILY_API_KEY"}
        requests = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize"},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
             "params": {"name": "find_news", "arguments": {"query": "tents"}}},
        ]
        stdin = "\n".join(json.dumps(r) for r in requests) + "\nnot json\n"
        proc = subprocess.run(
            [sys.executable, str(AFTER / "bma_research_mcp.py")],
            input=stdin, capture_output=True, text=True, timeout=30, env=env, check=True,
        )
        lines = [json.loads(line) for line in proc.stdout.splitlines()]
        # initialize, tools/list, tools/call, then the parse error — no reply for the notification.
        self.assertEqual([m.get("id") for m in lines], [1, 2, 3, None])
        self.assertEqual(lines[3]["error"]["code"], -32700)
        self.assertEqual(len(lines[1]["result"]["tools"]), 3)


# ── The BMA client/session logic (stdlib-only: boto3/requests import lazily) ──

import bedrock_managed_agents_researcher as bma  # noqa: E402


def _msg(turn, role, text, phase=None):
    item = {
        "id": f"item-{turn}-{role}", "type": "message", "role": role, "status": "completed",
        "turn_id": turn,
        "content": [{"type": "input_text" if role == "user" else "output_text", "text": text}],
    }
    if phase:
        item["phase"] = phase
    return item


# Item shapes captured from a live run of the preview (ids replaced).
def _turn_items(turn, answer):
    return [
        _msg(turn, "user", "question"),
        {"id": f"rs-{turn}", "type": "reasoning", "status": "completed", "summary": [], "turn_id": turn},
        {"id": f"call-{turn}", "type": "mcp_call", "status": "completed", "turn_id": turn,
         "server_label": "research_search", "name": "find_information",
         "arguments": json.dumps({"query": "q", "market": "en-US"}),
         "output": {"_meta": None, "content": [{"type": "text", "text": json.dumps({"results": []})}]}},
        _msg(turn, "assistant", answer, phase="final_answer"),
    ]


class TestPureHelpers(unittest.TestCase):
    def test_session_body_matches_the_live_accepted_shape(self):
        body = bma.build_session_body(
            model="openai.gpt-5.6-luna", instructions="i", role_arn="arn:aws:iam::123456789012:role/R",
            workspace="/ws", mcp_command="/usr/bin/python3", mcp_script="/x/bma_research_mcp.py",
            forward_env=["TAVILY_API_KEY"],
        )
        self.assertEqual(body["role_arn"], "arn:aws:iam::123456789012:role/R")
        self.assertEqual(body["agent"]["model"], "openai.gpt-5.6-luna")  # bare id, no us. prefix
        tool = body["agent"]["tools"][0]
        self.assertEqual(tool["type"], "mcp")
        self.assertEqual(tool["transport"]["type"], "stdio")
        self.assertEqual(tool["transport"]["env"], {})  # secrets are forwarded by NAME only
        self.assertEqual(tool["transport"]["env_vars"], ["TAVILY_API_KEY"])
        self.assertEqual(tool["allowed_tools"], ["find_information", "find_entities", "find_news"])
        self.assertTrue(tool["transport"]["command"].startswith("/"))
        self.assertTrue(tool["transport"]["cwd"].startswith("/"))
        self.assertEqual(body["environment"]["type"], "self_hosted")
        self.assertFalse(body["stream"])

    def test_final_answer_reads_only_assistant_final_answer_text(self):
        items = _turn_items("turn_1", '{"web": []}')
        self.assertEqual(bma.final_answer(items, set()), ("turn_1", '{"web": []}'))

    def test_final_answer_ignores_turns_already_consumed(self):
        items = _turn_items("turn_1", "old") + _turn_items("turn_2", "new")
        self.assertEqual(bma.final_answer(items, {"turn_1"}), ("turn_2", "new"))
        self.assertIsNone(bma.final_answer(items, {"turn_1", "turn_2"}))

    def test_final_answer_none_while_turn_is_still_running(self):
        running = [_msg("turn_1", "user", "q"),
                   {"type": "reasoning", "id": "r", "turn_id": "turn_1", "status": "in_progress", "summary": []}]
        self.assertIsNone(bma.final_answer(running, set()))

    def test_final_answer_skips_non_final_assistant_messages(self):
        items = [_msg("t", "assistant", "thinking out loud", phase="commentary")]
        self.assertIsNone(bma.final_answer(items, set()))

    def test_extract_json_object_tolerates_fences_and_prose(self):
        self.assertEqual(bma.extract_json_object('{"web": []}'), {"web": []})
        self.assertEqual(bma.extract_json_object('Here:\n```json\n{"web": [1]}\n```'), {"web": [1]})
        with self.assertRaises(ValueError):
            bma.extract_json_object("no json here")
        with self.assertRaises(ValueError):
            bma.extract_json_object("[1, 2]")

    def test_normalize_fills_the_contract_keys(self):
        self.assertEqual(bma.normalize({"web": [{"url": "u"}]}), {"web": [{"url": "u"}], "entities": [], "news": []})


class FakeApi:
    """Scripted stand-in for BmaApi. `statuses` is consumed one per session GET."""

    endpoint, region = "https://bedrock-mantle.us-east-1.api.aws", "us-east-1"

    def __init__(self, statuses, item_pages, stream_lines=()):
        self.statuses, self.item_pages, self.calls = list(statuses), list(item_pages), []
        self.stream_lines = list(stream_lines)

    def stream(self, path):
        self.calls.append(("STREAM", path, None))
        lines = self.stream_lines

        class _Resp:
            def iter_lines(self, **_):
                yield from lines

            def close(self):
                pass

        return _Resp()

    def json(self, method, path, body=None):
        self.calls.append((method, path, body))
        if path.endswith("/events") and method == "POST":
            return {}
        if "/items" in path:
            return self.item_pages.pop(0) if len(self.item_pages) > 1 else self.item_pages[0]
        status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        return {"id": "sess_x", "status": status, "error": {"message": "boom"} if status == "failed" else None}


def _page(items, has_more=False, last_id=None):
    return {"object": "list", "data": items, "has_more": has_more, "first_id": None, "last_id": last_id}


def _session(api, **kw):
    s = bma.ManagedAgentSession(api=api, poll_interval=0, **kw)
    s.session_id = "sess_x"
    return s


class TestSend(unittest.TestCase):
    def test_returns_answer_and_submits_the_documented_event(self):
        api = FakeApi(["idle"], [_page(_turn_items("turn_1", "done"))])
        self.assertEqual(_session(api).send("hello"), "done")
        method, path, body = next(c for c in api.calls if c[0] == "POST" and c[1].endswith("/events"))
        self.assertEqual((method, path), ("POST", "/openai/v1/agents/sessions/sess_x/events"))
        self.assertEqual(body["events"][0]["type"], "agent.session.input.message")
        self.assertEqual(body["events"][0]["input"][0]["content"], [{"type": "input_text", "text": "hello"}])

    def test_idle_before_the_turn_starts_is_not_mistaken_for_done(self):
        # Observed live: status can read "idle" right after submit. The first two
        # polls see only the previous turn; the new answer lands on the third.
        old = _turn_items("turn_1", "old")
        pages = [_page(old), _page(old), _page(old + _turn_items("turn_2", "new"))]
        api = FakeApi(["idle", "in_progress", "idle"], pages)
        s = _session(api)
        s._seen_turns.add("turn_1")
        self.assertEqual(s.send("again"), "new")

    def test_second_turn_does_not_return_the_first_answer(self):
        first = _turn_items("turn_1", "first")
        api = FakeApi(["idle"], [_page(first)])
        s = _session(api)
        self.assertEqual(s.send("one"), "first")
        api.item_pages = [_page(first + _turn_items("turn_2", "second"))]
        self.assertEqual(s.send("two"), "second")

    def test_failed_session_raises_with_details(self):
        with self.assertRaisesRegex(RuntimeError, "boom"):
            _session(FakeApi(["failed"], [_page([])])).send("x")

    def test_times_out_instead_of_hanging(self):
        with self.assertRaises(TimeoutError):
            _session(FakeApi(["in_progress"], [_page([])]), turn_timeout=0.05).send("x")

    def test_items_follow_the_opaque_cursor(self):
        a, b = _turn_items("t1", "a"), _turn_items("t2", "b")
        api = FakeApi(["idle"], [_page(a, has_more=True, last_id="cursor/1=="), _page(b)])
        got = list(_session(api).items())
        self.assertEqual(len(got), len(a) + len(b))
        second = [c for c in api.calls if "/items" in c[1]][1][1]
        self.assertIn("after=cursor%2F1%3D%3D", second)  # URL-encoded, per the docs


# Failure payloads captured live (model denied by the session role); account id replaced.
_DENIED = (
    "model inference was denied under this session's role: access_denied: "
    "User: arn:aws:sts::123456789012:assumed-role/BedrockManagedAgentsPreviewInferenceServiceRole/x "
    "is not authorized to perform: bedrock-mantle:CreateInference"
)
_ERROR_EVENT = {"type": "error", "session_id": "sess_x",
                "error": {"type": "authentication_error", "code": "authentication_error", "message": _DENIED}}
_TURN_FAILED_EVENT = {"type": "agent.session.turn.failed", "turn_id": "turn_1",
                      "turn": {"id": "turn_1", "status": "failed",
                               "error": {"code": "authentication_error", "message": _DENIED}}}


class TestTurnFailureIsVisibleOnlyInTheStream(unittest.TestCase):
    def test_parse_failure_reads_error_and_turn_failed_events(self):
        for event in (_ERROR_EVENT, _TURN_FAILED_EVENT):
            self.assertIn("CreateInference", bma.parse_failure(event))
            self.assertTrue(bma.parse_failure(event).startswith("authentication_error:"))

    def test_parse_failure_ignores_healthy_events(self):
        for kind in ("agent.session.in_progress", "agent.session.turn.completed",
                     "agent.session.turn.output_text.delta", "agent.session.idle"):
            self.assertIsNone(bma.parse_failure({"type": kind}))

    def test_denied_turn_raises_fast_instead_of_timing_out(self):
        # Observed live: status stays "idle", error is null, items hold only the user
        # message — polling alone can never see this. The stream does.
        user_only = _page([_msg("turn_1", "user", "q")])
        lines = ["event: error", "data: " + json.dumps(_ERROR_EVENT), "",
                 "event: agent.session.turn.failed", "data: " + json.dumps(_TURN_FAILED_EVENT), ""]
        api = FakeApi(["idle"], [user_only], stream_lines=lines)
        with self.assertRaisesRegex(RuntimeError, "CreateInference"):
            _session(api, turn_timeout=5).send("hi")

    def test_stream_is_opened_before_the_message_is_submitted(self):
        api = FakeApi(["idle"], [_page(_turn_items("turn_1", "ok"))])
        _session(api).send("hi")
        order = [c[0] for c in api.calls if c[0] in ("STREAM", "POST")]
        self.assertEqual(order, ["STREAM", "POST"])  # events are not replayed


if __name__ == "__main__":
    unittest.main(verbosity=2)
