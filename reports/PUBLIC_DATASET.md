# Public adjacent benchmark: BFCL V4 Live single-tool choice

The frozen file at data/public/bfcl_v4_live_tool_selection.jsonl contains 120 unambiguous single-function selection cases from official BFCL V4 Live multiple-function data. SHA256: 3a26991a890d3ec5ade104f5a070dd8e3c28d85e233c35ae2f7d0dee5f1fdbcd. Every row is held out as test; no threshold is tuned on it.

Source: https://github.com/EnlightenedAI/BFCL at commit 6ea57973c7a6097fd7c5915698c54c17c5b1b6c8. The official data README declares Apache-2.0. Question JSONL SHA256: 95fe6f0cc1a3555285bae6b3b549fc5c597781120eb9c4eecd990feac26c4ac4. Answer JSONL SHA256: e9237516245599c2f017d9d4553f5e83a53cd40d65c9028b14372402a549f0bb.

The converter accepts a case only when it has one text user request, at least two named and described function choices, and exactly one gold function call that matches a choice. It discards arguments, multi-call cases, and ambiguous records. This measures Laya's choice among described functions. It is not the BFCL leaderboard, function-call generation, argument accuracy, or a direct company-routing benchmark.

Rebuild with official files:

    python scripts/build_public_dataset.py --cases PATH_TO_QUESTIONS --answers PATH_TO_GOLD --source-commit 6ea57973c7a6097fd7c5915698c54c17c5b1b6c8 --cases-sha256 95fe6f0cc1a3555285bae6b3b549fc5c597781120eb9c4eecd990feac26c4ac4 --answers-sha256 e9237516245599c2f017d9d4553f5e83a53cd40d65c9028b14372402a549f0bb --seed 20261005 --limit 120
