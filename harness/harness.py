#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
harness.py — Azure→Bedrock chat parity harness (record / replay / diff).

Workflow:

  1. record   Call the source (Azure/OpenAI) for each case, save normalized
              responses to recordings/<label>/<case_id>.json.
  2. replay   Call a Bedrock target (endpoint or Converse) for each case, save
              to recordings/<label>/<case_id>.json.
  3. diff     Compare two recording sets case-by-case for parity, print a
              report, and exit non-zero if any case fails (CI-friendly).

`record` and `replay` need network + the openai/boto3 deps. `diff` is pure
stdlib and offline, so parity can be re-checked (and unit-tested) without
credentials.

Examples:
    # 1. record from Azure
    python harness.py record --label azure

    # 2. replay on the Bedrock OpenAI-compatible endpoint
    python harness.py replay --provider bedrock-endpoint --label bedrock

    # 3. compare
    python harness.py diff --source azure --target bedrock

    # stricter: require near-identical text
    python harness.py diff --source azure --target bedrock --threshold 0.85
    # strictest: exact (normalized) text match
    python harness.py diff --source azure --target bedrock --exact
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import diff as diffmod

HERE = Path(__file__).resolve().parent
DEFAULT_CASES = HERE / "cases.jsonl"
RECORDINGS_DIR = HERE / "recordings"


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if "id" not in case or "messages" not in case:
                raise SystemExit(f"{path}:{line_no}: case needs 'id' and 'messages'")
            case.setdefault("params", {})
            cases.append(case)
    if not cases:
        raise SystemExit(f"{path}: no cases found")
    return cases


def _label_dir(label: str) -> Path:
    return RECORDINGS_DIR / label


def _save(label: str, case_id: str, normalized: dict[str, Any]) -> Path:
    out_dir = _label_dir(label)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{case_id}.json"
    out_path.write_text(json.dumps(normalized, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def _load_recording(label: str, case_id: str) -> dict[str, Any] | None:
    path = _label_dir(label) / f"{case_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def cmd_capture(args: argparse.Namespace, provider_name: str) -> int:
    """Shared record/replay implementation."""
    from providers import get_provider  # lazy: only record/replay need network deps

    cases = load_cases(Path(args.cases))
    provider = get_provider(provider_name)
    print(f"▶ {args._verb} {len(cases)} case(s) via '{provider_name}' → label '{args.label}'\n")

    failures = 0
    for case in cases:
        cid = case["id"]
        try:
            normalized = provider.run(case["messages"], case.get("params", {}))
            path = _save(args.label, cid, normalized)
            preview = (normalized.get("text") or "").replace("\n", " ")[:60]
            print(f"  ✓ {cid:<12} → {path.name}  “{preview}…”")
        except Exception as exc:  # noqa: BLE001 — surface any provider error per-case
            failures += 1
            print(f"  ✗ {cid:<12} ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)

    print(f"\nDone. {len(cases) - failures} ok, {failures} failed. "
          f"Recordings in {_label_dir(args.label)}")
    return 1 if failures else 0


def cmd_diff(args: argparse.Namespace) -> int:
    cases = load_cases(Path(args.cases))
    print(f"▶ diff  source='{args.source}'  target='{args.target}'  "
          f"{'exact' if args.exact else f'threshold={args.threshold}'}\n")

    results: list[diffmod.DiffResult] = []
    missing = 0
    for case in cases:
        cid = case["id"]
        src = _load_recording(args.source, cid)
        tgt = _load_recording(args.target, cid)
        if src is None or tgt is None:
            missing += 1
            which = []
            if src is None:
                which.append(f"source '{args.source}'")
            if tgt is None:
                which.append(f"target '{args.target}'")
            print(f"  ⚠ {cid:<12} MISSING recording in {', '.join(which)}", file=sys.stderr)
            continue
        results.append(
            diffmod.compare(cid, src, tgt, threshold=args.threshold, exact=args.exact)
        )

    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed

    for r in results:
        mark = "✓" if r.passed else "✗"
        print(f"  {mark} {r.case_id:<12} similarity={r.similarity:.3f}")
        for reason in r.reasons:
            print(f"       - {reason}")

    print(f"\nParity: {passed} passed, {failed} failed, {missing} missing.")

    if args.report:
        report = {
            "source": args.source,
            "target": args.target,
            "threshold": args.threshold,
            "exact": args.exact,
            "summary": {"passed": passed, "failed": failed, "missing": missing},
            "results": [r.to_dict() for r in results],
        }
        Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Report written to {args.report}")

    return 1 if (failed or missing) else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harness.py",
        description="Azure→Bedrock chat parity harness (record / replay / diff).",
    )
    parser.add_argument("--cases", default=str(DEFAULT_CASES),
                        help=f"path to cases JSONL (default: {DEFAULT_CASES.name})")
    sub = parser.add_subparsers(dest="command", required=True)

    p_record = sub.add_parser("record", help="record responses from the source (Azure/OpenAI)")
    p_record.add_argument("--provider", default="source",
                          help="source provider name (default: source)")
    p_record.add_argument("--label", default="source", help="recording label / folder name")

    p_replay = sub.add_parser("replay", help="replay cases on a Bedrock target")
    p_replay.add_argument("--provider", required=True,
                          choices=["bedrock-endpoint", "bedrock-converse", "source"],
                          help="target provider to replay on")
    p_replay.add_argument("--label", required=True, help="recording label / folder name")

    p_diff = sub.add_parser("diff", help="compare two recording sets for parity")
    p_diff.add_argument("--source", required=True, help="source recording label")
    p_diff.add_argument("--target", required=True, help="target recording label")
    p_diff.add_argument("--threshold", type=float, default=diffmod.DEFAULT_THRESHOLD,
                        help=f"similarity threshold 0..1 (default: {diffmod.DEFAULT_THRESHOLD})")
    p_diff.add_argument("--exact", action="store_true",
                        help="require exact (normalized) text match instead of similarity")
    p_diff.add_argument("--report", help="write a JSON parity report to this path")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "record":
        args._verb = "record"
        return cmd_capture(args, args.provider)
    if args.command == "replay":
        args._verb = "replay"
        return cmd_capture(args, args.provider)
    if args.command == "diff":
        return cmd_diff(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
