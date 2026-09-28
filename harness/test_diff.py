#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
test_diff.py — offline tests for the parity layer.

Runs with pure stdlib (no network, no openai/boto3). Exercises:
  - normalize_openai / normalize_converse produce the same shape from the two
    different provider responses,
  - a Converse response and an equivalent OpenAI response are judged at parity,
  - the similarity threshold catches a real regression,
  - the --exact flag distinguishes near-identical from identical text,
  - structural checks catch finish_reason and tool_call divergence.

Run:  python harness/test_diff.py   (exit 0 = all passed)
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import diff as diffmod
import normalize as norm

# --- Synthetic raw provider responses -------------------------------------

OPENAI_RESP = {
    "choices": [
        {
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "The answer is 391."},
        }
    ],
    "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
}

# Same meaning, Converse shape, "end_turn" -> "stop".
CONVERSE_RESP = {
    "output": {"message": {"role": "assistant", "content": [{"text": "The answer is 391."}]}},
    "stopReason": "end_turn",
    "usage": {"inputTokens": 22, "outputTokens": 5, "totalTokens": 27},
}

CONVERSE_RESP_REWORDED = {
    "output": {"message": {"role": "assistant", "content": [{"text": "The answer is 391."}]}},
    "stopReason": "end_turn",
    "usage": {"inputTokens": 22, "outputTokens": 5, "totalTokens": 27},
}

CONVERSE_RESP_WRONG = {
    "output": {"message": {"role": "assistant", "content": [{"text": "I cannot help with that request."}]}},
    "stopReason": "end_turn",
    "usage": {"inputTokens": 22, "outputTokens": 6, "totalTokens": 28},
}

# Tool-use responses in both shapes.
OPENAI_TOOL = {
    "choices": [
        {
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"function": {"name": "get_weather", "arguments": '{"city": "Seattle"}'}}
                ],
            },
        }
    ],
    "usage": {"prompt_tokens": 30, "completion_tokens": 10, "total_tokens": 40},
}

CONVERSE_TOOL = {
    "output": {
        "message": {
            "role": "assistant",
            "content": [{"toolUse": {"name": "get_weather", "input": {"city": "Seattle"}}}],
        }
    },
    "stopReason": "tool_use",
    "usage": {"inputTokens": 30, "outputTokens": 10, "totalTokens": 40},
}


class TestNormalize(unittest.TestCase):
    def test_openai_shape(self):
        n = norm.normalize_openai(OPENAI_RESP)
        self.assertEqual(n["text"], "The answer is 391.")
        self.assertEqual(n["finish_reason"], "stop")
        self.assertEqual(n["tool_calls"], [])
        self.assertEqual(n["usage"]["input_tokens"], 20)
        self.assertEqual(n["usage"]["output_tokens"], 5)

    def test_converse_shape(self):
        n = norm.normalize_converse(CONVERSE_RESP)
        self.assertEqual(n["text"], "The answer is 391.")
        self.assertEqual(n["finish_reason"], "stop")  # end_turn -> stop
        self.assertEqual(n["usage"]["input_tokens"], 22)

    def test_finish_reason_mapping(self):
        self.assertEqual(norm.normalize_finish_reason("max_tokens"), "length")
        self.assertEqual(norm.normalize_finish_reason("length"), "length")
        self.assertEqual(norm.normalize_finish_reason("tool_use"), "tool_use")
        self.assertEqual(norm.normalize_finish_reason("tool_calls"), "tool_use")
        self.assertEqual(norm.normalize_finish_reason(None), "unknown")

    def test_tool_calls_normalize_equivalently(self):
        a = norm.normalize_openai(OPENAI_TOOL)
        b = norm.normalize_converse(CONVERSE_TOOL)
        # OpenAI decodes the JSON-string arguments; both end up as the same dict.
        self.assertEqual(a["tool_calls"], [{"name": "get_weather", "arguments": {"city": "Seattle"}}])
        self.assertEqual(a["tool_calls"], b["tool_calls"])
        self.assertEqual(a["finish_reason"], b["finish_reason"])


class TestDiff(unittest.TestCase):
    def test_cross_provider_parity(self):
        src = norm.normalize_openai(OPENAI_RESP)
        tgt = norm.normalize_converse(CONVERSE_RESP)
        result = diffmod.compare("math", src, tgt)
        self.assertTrue(result.passed, result.reasons)
        self.assertEqual(result.similarity, 1.0)

    def test_regression_is_caught(self):
        src = norm.normalize_openai(OPENAI_RESP)
        tgt = norm.normalize_converse(CONVERSE_RESP_WRONG)
        result = diffmod.compare("math", src, tgt, threshold=0.60)
        self.assertFalse(result.passed)
        self.assertTrue(any("similarity" in r for r in result.reasons))

    def test_exact_flag(self):
        src = norm.normalize_openai(OPENAI_RESP)
        tgt = norm.normalize_converse(CONVERSE_RESP_REWORDED)
        # Identical normalized text -> exact passes.
        self.assertTrue(diffmod.compare("math", src, tgt, exact=True).passed)

    def test_exact_flag_rejects_near_miss(self):
        src = norm.normalize_openai(OPENAI_RESP)
        near = norm.normalize_openai(
            {
                "choices": [
                    {"finish_reason": "stop",
                     "message": {"role": "assistant", "content": "The answer is 392."}}
                ],
                "usage": {},
            }
        )
        # High similarity but not identical -> exact fails, similarity passes.
        self.assertFalse(diffmod.compare("math", src, near, exact=True).passed)
        self.assertTrue(diffmod.compare("math", src, near, threshold=0.60).passed)

    def test_finish_reason_divergence_caught(self):
        src = norm.normalize_openai(OPENAI_RESP)  # stop
        tgt = norm.normalize_converse(
            {
                "output": {"message": {"role": "assistant", "content": [{"text": "The answer is 391."}]}},
                "stopReason": "max_tokens",  # -> length
                "usage": {},
            }
        )
        result = diffmod.compare("math", src, tgt)
        self.assertFalse(result.passed)
        self.assertTrue(any("finish_reason" in r for r in result.reasons))

    def test_tool_call_parity(self):
        src = norm.normalize_openai(OPENAI_TOOL)
        tgt = norm.normalize_converse(CONVERSE_TOOL)
        result = diffmod.compare("weather", src, tgt)
        self.assertTrue(result.passed, result.reasons)


if __name__ == "__main__":
    unittest.main(verbosity=2)
