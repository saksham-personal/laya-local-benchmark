"""Strict adapter for single-tool BFCL V4 Live multiple-function cases.

This intentionally rejects multi-call, ambiguous, or malformed answers: the
benchmark schema measures function choice, not argument generation.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

DEFAULT_SEED = 20261005
SOURCE_URL = "https://github.com/EnlightenedAI/BFCL"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _question_text(value: Any) -> str:
    """Extract one user request; reject multi-turn or non-text prompts."""
    if isinstance(value, str):
        text = value.strip()
        if text:
            return text
        raise ValueError("empty question")
    if isinstance(value, list):
        messages: list[Any] = []

        def flatten(node: Any) -> None:
            if isinstance(node, list):
                for child in node:
                    flatten(child)
            elif isinstance(node, dict):
                messages.append(node)
        flatten(value)
        user_messages = [m for m in messages if m.get("role") == "user"]
        if len(user_messages) == 1 and isinstance(user_messages[0].get("content"), str):
            text = user_messages[0]["content"].strip()
            if text:
                return text
    raise ValueError("expected exactly one text user message")


def _gold_function_names(answer: dict[str, Any]) -> list[str]:
    truth = answer.get("ground_truth")
    if not isinstance(truth, list) or len(truth) != 1 or not isinstance(truth[0], dict):
        raise ValueError("gold must contain exactly one function call")
    if len(truth[0]) != 1:
        raise ValueError("gold call must identify exactly one function")
    name = next(iter(truth[0]))
    if not isinstance(name, str) or not name.strip():
        raise ValueError("gold function name must be non-empty text")
    return [name]


def convert_cases(
    cases: Iterable[dict[str, Any]],
    answers: Iterable[dict[str, Any]],
    *,
    limit: int = 120,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    """Return a deterministically sampled, strictly labeled routing slice."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    answer_by_id: dict[Any, dict[str, Any] | None] = {}
    for item in answers:
        if not isinstance(item, dict):
            continue
        answer_id = item.get("id")
        if not isinstance(answer_id, str):
            continue
        if answer_id in answer_by_id:
            answer_by_id[answer_id] = None
        else:
            answer_by_id[answer_id] = item
    eligible: list[dict[str, Any]] = []
    seen: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            continue
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id.startswith("live_multiple_") or case_id in seen:
            continue
        seen.add(case_id)
        answer = answer_by_id.get(case_id)
        if answer is None:
            continue
        try:
            names = _gold_function_names(answer)
            question = _question_text(case.get("question"))
            functions = case.get("function")
            if not isinstance(functions, list) or len(functions) < 2:
                continue
            criteria: dict[str, str] = {}
            for function in functions:
                if not isinstance(function, dict):
                    raise ValueError("malformed function schema")
                name = function.get("name")
                description = function.get("description")
                if not isinstance(name, str) or not name.strip() or not isinstance(description, str) or not description.strip():
                    raise ValueError("function name and description must be present")
                if name in criteria:
                    raise ValueError("duplicate function name")
                criteria[name] = description.strip()
            if names[0] not in criteria:
                continue
            eligible.append({
                "id": case_id,
                "task": "tool_selection",
                "state": "User request:\n" + question,
                "question": {"decision": {
                    "type": "choice",
                    "instructions": "Select the appropriate function for the user request.",
                    "criteria": criteria,
                }},
                "gold": names[0],
                "split": "test",
                "tags": ["public", "bfcl", "bfcl_v4_live", "live_multiple", "tool_selection"],
            })
        except (ValueError, TypeError):
            continue
    eligible.sort(key=lambda row: row["id"])
    random.Random(seed).shuffle(eligible)
    selected = eligible[:limit]
    # Keep this external benchmark entirely held out. Calibration thresholds
    # are fitted on the application-specific development split.
    return selected


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: expected JSON object")
                rows.append(value)
    return rows
