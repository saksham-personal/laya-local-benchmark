from __future__ import annotations

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from laya_bench.datasets.bfcl import convert_cases  # noqa: E402


def _case(case_id: str = "live_multiple_1") -> dict:
    return {
        "id": case_id,
        "question": [[{"role": "user", "content": "Find a flight to Delhi."}]],
        "function": [
            {"name": "search_flights", "description": "Search available flights."},
            {"name": "search_hotels", "description": "Search available hotels."},
        ],
    }


def _answer(case_id: str = "live_multiple_1", gold: list | None = None) -> dict:
    if gold is None:
        gold = [{"search_flights": {"destination": ["Delhi"]}}]
    return {"id": case_id, "ground_truth": gold}


def test_converts_only_single_unambiguous_function_choice() -> None:
    rows = convert_cases([_case()], [_answer()])
    assert len(rows) == 1
    row = rows[0]
    assert set(row) == {"id", "task", "state", "question", "gold", "split", "tags"}
    assert row["gold"] == "search_flights"
    assert row["task"] == "tool_selection"
    assert row["question"]["decision"]["type"] == "choice"
    assert row["question"]["decision"]["criteria"] == {
        "search_flights": "Search available flights.",
        "search_hotels": "Search available hotels.",
    }
    assert "Find a flight to Delhi." in row["state"]


def test_rejects_argument_only_multicall_bad_and_non_live_cases() -> None:
    cases = [
        _case("live_multiple_1"),
        _case("live_multiple_2"),
        _case("live_multiple_3"),
        _case("simple_python_1"),
        _case("live_multiple_4"),
    ]
    answers = [
        _answer("live_multiple_1"),
        _answer("live_multiple_2", [{"search_flights": {"to": ["Delhi"]}, "search_hotels": {}}]),
        _answer("live_multiple_3", [{"unknown_function": {}}]),
        _answer("simple_python_1"),
        _answer("live_multiple_4", []),
    ]
    assert [row["id"] for row in convert_cases(cases, answers)] == ["live_multiple_1"]


def test_sampling_is_stable_and_caps_at_limit() -> None:
    cases = [_case(f"live_multiple_{number:03d}") for number in range(18)]
    answers = [_answer(case["id"]) for case in cases]
    first = convert_cases(cases, answers, limit=10, seed=20261005)
    second = convert_cases(cases, answers, limit=10, seed=20261005)
    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert len(first) == 10
    assert {row["split"] for row in first} == {"test"}
    assert "test" in {row["split"] for row in first}
