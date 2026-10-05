"""Build the BFCL V4 Live single-tool routing slice from official JSONL files."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from laya_bench.datasets.bfcl import (  # noqa: E402
    DEFAULT_SEED,
    SOURCE_URL,
    convert_cases,
    load_jsonl,
    sha256_file,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True, help="Official BFCL live_multiple question JSONL")
    parser.add_argument("--answers", type=Path, required=True, help="Official BFCL live_multiple possible-answer JSONL")
    parser.add_argument("--output", type=Path, default=ROOT / "data/public/bfcl_v4_live_tool_selection.jsonl")
    parser.add_argument("--source-commit", required=True, help="40-character official BFCL repository commit SHA")
    parser.add_argument("--cases-sha256", required=True, help="Expected SHA256 of --cases")
    parser.add_argument("--answers-sha256", required=True, help="Expected SHA256 of --answers")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--limit", type=int, default=120)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-fA-F]{40}", args.source_commit):
        parser.error("--source-commit must be a full 40-character commit SHA")
    for path, expected in ((args.cases, args.cases_sha256), (args.answers, args.answers_sha256)):
        actual = sha256_file(path)
        if actual.lower() != expected.lower():
            parser.error(f"SHA256 mismatch for {path}: expected {expected}, got {actual}")
    rows = convert_cases(load_jsonl(args.cases), load_jsonl(args.answers), limit=args.limit, seed=args.seed)
    if not rows:
        parser.error("no unambiguous single-tool live_multiple cases found; refusing to fabricate labels")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({
        "cases": len(rows),
        "source_url": f"{SOURCE_URL}/tree/{args.source_commit}/berkeley-function-call-leaderboard/bfcl_eval/data",
        "source_commit": args.source_commit,
        "cases_sha256": sha256_file(args.cases),
        "answers_sha256": sha256_file(args.answers),
        "output": str(args.output),
        "seed": args.seed,
        "limit": args.limit,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
