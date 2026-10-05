from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from laya_bench.datasets.custom import DEFAULT_SEED, build_rows, write_jsonl


def test_dataset_is_deterministic_and_in_expected_size() -> None:
    first = build_rows()
    assert len(first) == 384
    assert first == build_rows(DEFAULT_SEED)
    assert first != build_rows(DEFAULT_SEED + 1)


def test_rows_have_unique_ids_and_valid_laya_schema() -> None:
    rows = build_rows()
    assert len({row["id"] for row in rows}) == len(rows)
    assert {"intent", "external_research", "expensive_reasoning", "search_type", "evidence_sufficiency", "direct_escalate"} == {row["task"] for row in rows}
    assert {row["gold"] for row in rows if row["task"] == "intent"} == {
        "company_screening", "company_comparison", "database_question",
        "research_request", "report_generation", "general_chat",
    }
    assert "hybrid_semantic_structured" in {row["gold"] for row in rows if row["task"] == "search_type"}
    assert all("Taskline" not in row["state"] for row in rows)
    for row in rows:
        assert set(row) == {"id", "task", "state", "question", "gold", "split", "tags"}
        assert row["split"] in {"dev", "test"}
        assert isinstance(row["state"], str) and row["state"]
        assert len(row["question"]) == 1
        question = next(iter(row["question"].values()))
        assert question["type"] in {"choice", "noul"}
        assert question["instructions"]
        if question["type"] == "noul":
            assert row["gold"] in {"yes", "no"}
        else:
            assert isinstance(question["criteria"], dict)
            assert set(question["criteria"]) >= {row["gold"]}
            assert all(isinstance(text, str) and text for text in question["criteria"].values())


def test_class_counts_and_split_template_families() -> None:
    rows = build_rows()
    assert Counter(row["task"] for row in rows) == {
        "intent": 64, "external_research": 64, "expensive_reasoning": 64,
        "search_type": 64, "evidence_sufficiency": 64, "direct_escalate": 64,
    }
    for task in {row["task"] for row in rows}:
        task_rows = [row for row in rows if row["task"] == task]
        counts = Counter(row["gold"] for row in task_rows)
        if task in {"external_research", "expensive_reasoning", "evidence_sufficiency"}:
            assert counts == {"yes": 32, "no": 32}
        if task in {"intent", "search_type", "direct_escalate"}:
            assert max(counts.values()) - min(counts.values()) <= 4
        assert Counter(row["split"] for row in task_rows) == {"dev": 32, "test": 32}
        template_splits: dict[str, set[str]] = {}
        for row in task_rows:
            template = next(tag for tag in row["tags"] if tag.startswith("template_"))
            template_splits.setdefault(template, set()).add(row["split"])
        assert len(template_splits) == 8
        assert all(len(splits) == 1 for splits in template_splits.values())
        assert {next(iter(splits)) for splits in template_splits.values()} == {"dev", "test"}


def test_counterfactual_pairs_are_adjacent_in_content_and_flip_gold() -> None:
    rows = build_rows()
    pairs: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        template = next(tag for tag in row["tags"] if tag.startswith("template_"))
        pair = next(tag for tag in row["tags"] if tag.startswith("pair_"))
        pairs.setdefault((row["task"], template, pair), []).append(row)
    assert len(pairs) == 6 * 8 * 4
    for pair in pairs.values():
        assert len(pair) == 2
        assert pair[0]["gold"] != pair[1]["gold"]
        assert pair[0]["split"] == pair[1]["split"]
        assert pair[0]["question"] == pair[1]["question"]


def test_jsonl_writer_round_trips() -> None:
    path = PROJECT_ROOT / "reports" / ".custom-dataset-test.jsonl"
    rows = build_rows()
    try:
        write_jsonl(path, rows)
        decoded = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert decoded == rows
    finally:
        path.unlink(missing_ok=True)
