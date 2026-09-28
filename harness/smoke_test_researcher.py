#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""
smoke_test_researcher.py — validate the researcher agent Converse loop on Bedrock.

Tests two things:
  1. bedrock_converse_loop.research() — the migrated Azure AI Agent Service researcher
     Verifies: tool calls are made, the loop completes, output is valid JSON
  2. harness BedrockConverse provider — basic chat parity cases
     Verifies: the harness replay path works end-to-end

Usage:
    # With your AWS profile (no Azure needed):
    cd azure-oai-to-bedrock-playbook
    AWS_PROFILE=your-profile \\
    AWS_REGION=us-east-1 \\
    python harness/smoke_test_researcher.py

    # Override model (optional — us.openai.gpt-5.6-terra is the default):
    BEDROCK_MODEL_ID=us.openai.gpt-5.6-terra \\
    AWS_PROFILE=your-profile \\
    python harness/smoke_test_researcher.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Resolve playbook root so we can import from migration/ and harness/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "migration/04_agents/azure_ai_agent_service/after"))

PROFILE  = os.environ.get("AWS_PROFILE", "")
REGION   = os.environ.get("AWS_REGION", "us-east-1")
MODEL    = os.environ.get("BEDROCK_MODEL_ID", "us.openai.gpt-5.6-terra")

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
SKIP = "\033[33m-\033[0m"
results: list[tuple[str, bool, str]] = []


def record(name: str, passed: bool, detail: str = "") -> None:
    results.append((name, passed, detail))
    mark = PASS if passed else FAIL
    print(f"  {mark} {name}" + (f"  — {detail}" if detail else ""))


# ── Test 1: Converse tool loop (researcher agent) ──────────────────────────────

print(f"\n{'='*60}")
print(f"  az2br migration smoke test")
print(f"  Profile : {PROFILE or '(default)'}")
print(f"  Region  : {REGION}")
print(f"  Model   : {MODEL}")
print(f"{'='*60}\n")

print("Test 1 — Researcher agent Converse tool loop")

try:
    import bedrock_converse_loop as researcher

    t0 = time.time()
    result = researcher.research(
        "What are the latest lightweight camping tent innovations in 2025?",
        feedback="No feedback",
    )
    elapsed = round(time.time() - t0, 1)

    # Validate output shape
    assert isinstance(result, dict), "result must be a dict"
    assert "web" in result,      "result must have 'web' key"
    assert "entities" in result, "result must have 'entities' key"
    assert "news" in result,     "result must have 'news' key"
    assert isinstance(result["web"], list), "'web' must be a list"
    assert len(result["web"]) > 0, "web results must not be empty"

    first = result["web"][0]
    assert "url" in first,         "each result needs 'url'"
    assert "name" in first,        "each result needs 'name'"
    assert "description" in first, "each result needs 'description'"

    record("tool loop completes",       True, f"{elapsed}s")
    record("output shape valid",        True, f"{len(result['web'])} web results")
    record("results have url/name/desc",True)

    # Show a sample result
    print(f"\n    Sample result:")
    print(f"    url:  {first['url'][:70]}")
    print(f"    name: {first['name'][:70]}")

except ImportError as e:
    record("import bedrock_converse_loop", False, str(e))
except AssertionError as e:
    record("output shape validation", False, str(e))
except Exception as e:
    record("tool loop execution", False, f"{type(e).__name__}: {e}")


# ── Test 2: Harness BedrockConverse provider ────────────────────────────────────

print("\nTest 2 — Harness BedrockConverse provider (basic cases)")

try:
    from providers import BedrockConverse

    provider = BedrockConverse()
    cases = [
        {"id": "math",   "messages": [{"role": "user", "content": "What is 17 * 23? Reply with only the number."}], "params": {"temperature": 0.0, "max_tokens": 16}},
        {"id": "list",   "messages": [{"role": "user", "content": "Name three AWS regions in the US, comma-separated."}], "params": {"temperature": 0.0, "max_tokens": 64}},
    ]

    for case in cases:
        t0 = time.time()
        resp = provider.run(case["messages"], case.get("params", {}))
        elapsed = round(time.time() - t0, 1)
        text = (resp.get("text") or "").strip()
        ok = bool(text)
        record(f"case '{case['id']}'", ok, f"{elapsed}s  →  \"{text[:60]}\"")

except Exception as e:
    record("harness provider", False, f"{type(e).__name__}: {e}")


# ── Summary ─────────────────────────────────────────────────────────────────────

passed = sum(1 for _, ok, _ in results if ok)
failed = len(results) - passed

print(f"\n{'='*60}")
print(f"  Results: {passed} passed, {failed} failed")
print(f"{'='*60}\n")

if failed:
    print("Failed tests:")
    for name, ok, detail in results:
        if not ok:
            print(f"  {FAIL} {name}: {detail}")
    sys.exit(1)
else:
    print("All checks passed.\n")
    sys.exit(0)
