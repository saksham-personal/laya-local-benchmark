"""Quality metrics for Laya benchmark prediction rows (standard library only)."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence


def _labels(rows: Sequence[Mapping[str, Any]]) -> list[Any]:
    values = {r.get("gold") for r in rows} | {r.get("pred") for r in rows}
    return sorted(values, key=lambda x: (type(x).__name__, str(x)))


def _class_metrics(rows: Sequence[Mapping[str, Any]], labels: Sequence[Any]) -> dict[str, Any]:
    matrix = [[0 for _ in labels] for _ in labels]
    index = {label: i for i, label in enumerate(labels)}
    for row in rows:
        matrix[index[row["gold"]]][index[row["pred"]]] += 1
    per_class = {}
    f1_values, weighted_sum = [], 0.0
    total = len(rows)
    for label, i in index.items():
        tp = matrix[i][i]
        support = sum(matrix[i])
        predicted = sum(matrix[r][i] for r in range(len(labels)))
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[str(label)] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        f1_values.append(f1)
        weighted_sum += f1 * support
    correct = sum(matrix[i][i] for i in range(len(labels)))
    return {
        "count": total,
        "accuracy": correct / total if total else None,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else None,
        "weighted_f1": weighted_sum / total if total else None,
        "labels": list(labels),
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def _auc(rows: Sequence[Mapping[str, Any]], positive: Any) -> float | None:
    scored = []
    for row in rows:
        probs = row.get("probabilities")
        if not isinstance(probs, Mapping) or positive not in probs:
            return None
        try:
            score = float(probs[positive])
        except (TypeError, ValueError):
            return None
        if not math.isfinite(score):
            return None
        scored.append((score, row["gold"] == positive))
    pos = sum(y for _, y in scored)
    neg = len(scored) - pos
    if not pos or not neg:
        return None
    # Mann-Whitney statistic with average ranks for ties.
    ordered = sorted(scored, key=lambda x: x[0])
    rank_sum = 0.0
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        avg_rank = ((i + 1) + j) / 2
        rank_sum += avg_rank * sum(1 for _, y in ordered[i:j] if y)
        i = j
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def _probability_metrics(rows: Sequence[Mapping[str, Any]], labels: Sequence[Any], bins: int = 10) -> dict[str, Any]:
    result: dict[str, Any] = {"roc_auc": None, "brier": None, "ece": None, "ece_bins": bins}
    if len(labels) == 2:
        positive = labels[-1]
        result["roc_auc"] = _auc(rows, positive)
        usable = []
        for row in rows:
            probs = row.get("probabilities")
            if not isinstance(probs, Mapping) or positive not in probs:
                continue
            try:
                p = float(probs[positive])
            except (TypeError, ValueError):
                continue
            if math.isfinite(p) and 0 <= p <= 1:
                usable.append((row, p))
        if usable:
            result["brier"] = sum((p - (row["gold"] == positive)) ** 2 for row, p in usable) / len(usable)
        result["probability_count"] = len(usable)
    else:
        result["probability_count"] = 0
    # Selected-answer ECE applies to both binary and multiclass questions.
    confidence_rows = []
    for row in rows:
        try:
            confidence = float(row.get("confidence"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(confidence) and 0 <= confidence <= 1:
            confidence_rows.append((confidence, row.get("gold") == row.get("pred")))
    if confidence_rows:
        ece = 0.0
        for b in range(bins):
            members = [(c, correct) for c, correct in confidence_rows if (b / bins <= c < (b + 1) / bins) or (b == bins - 1 and c == 1)]
            if members:
                avg_conf = sum(c for c, _ in members) / len(members)
                avg_acc = sum(correct for _, correct in members) / len(members)
                ece += len(members) / len(confidence_rows) * abs(avg_conf - avg_acc)
        result["ece"] = ece
        result["ece_count"] = len(confidence_rows)
    else:
        result["ece_count"] = 0
    return result


def _evaluate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = _labels(rows)
    metrics = _class_metrics(rows, labels)
    metrics.update(_probability_metrics(rows, labels))
    return metrics


def evaluate_quality(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Return JSON-safe overall and per-task quality metrics."""
    materialized = [dict(row) for row in rows]
    for row in materialized:
        if "gold" not in row or "pred" not in row:
            raise ValueError("each prediction row must include gold and pred")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in materialized:
        grouped[str(row.get("task", "unknown"))].append(row)
    return {"overall": _evaluate(materialized), "per_task": {task: _evaluate(group) for task, group in sorted(grouped.items())}}


def _confidence(row: Mapping[str, Any]) -> float:
    try:
        value = float(row.get("confidence"))
    except (TypeError, ValueError):
        return float("-inf")
    return value if math.isfinite(value) else float("-inf")


def selective_report(rows: Iterable[Mapping[str, Any]], coverages: Sequence[float] = (0.5, 0.7, 0.8, 1.0)) -> dict[str, Any]:
    """Report confidence-ranked selective performance; fit thresholds on dev when available.

    A dev-derived cutoff is applied to test using confidence >= cutoff, so boundary ties
    are all retained and achieved test coverage can exceed its target.
    """
    materialized = [dict(r) for r in rows]
    splits: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in materialized:
        splits[str(r.get("split", "unknown"))].append(r)
    dev = splits.get("dev", [])
    result: dict[str, Any] = {"strategy": "dev_threshold" if dev and "test" in splits else "ranked_per_split", "by_split": {}}
    for split, group in sorted(splits.items()):
        ordered = sorted(enumerate(group), key=lambda pair: (-_confidence(pair[1]), pair[0]))
        entries = []
        for coverage in coverages:
            if not 0 < float(coverage) <= 1:
                raise ValueError("coverages must be in (0, 1]")
            if dev and "test" in splits and split == "test":
                dev_sorted = sorted(enumerate(dev), key=lambda pair: (-_confidence(pair[1]), pair[0]))
                k = max(1, math.ceil(float(coverage) * len(dev_sorted))) if dev_sorted else 0
                threshold = _confidence(dev_sorted[k - 1][1]) if k else float("inf")
                accepted = [r for r in group if _confidence(r) >= threshold]
                threshold_out: float | None = threshold if math.isfinite(threshold) else None
            else:
                k = math.ceil(float(coverage) * len(ordered))
                accepted = [r for _, r in ordered[:k]]
                threshold_out = _confidence(ordered[k - 1][1]) if k else None
            n = len(group)
            accepted_n = len(accepted)
            acc = sum(r.get("gold") == r.get("pred") for r in accepted) / accepted_n if accepted_n else None
            entries.append({"target_coverage": float(coverage), "threshold": threshold_out,
                            "accepted": accepted_n, "total": n,
                            "achieved_coverage": accepted_n / n if n else None,
                            "accepted_accuracy": acc, "accepted_error": 1 - acc if acc is not None else None,
                            "escalated": n - accepted_n, "escalation_rate": (n - accepted_n) / n if n else None})
        result["by_split"][split] = entries
    return result


def compare_variants(reference: Iterable[Mapping[str, Any]], candidate: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare matched predictions; no equivalence assumption is made."""
    ref = {r["id"]: dict(r) for r in reference}
    cand = {r["id"]: dict(r) for r in candidate}
    ref_ids, cand_ids = set(ref), set(cand)
    ids = sorted(ref_ids & cand_ids, key=lambda x: (type(x).__name__, str(x)))
    matched_ref = [ref[i] for i in ids]
    matched_cand = [cand[i] for i in ids]
    deltas = []
    for i in ids:
        a, b = ref[i].get("probabilities"), cand[i].get("probabilities")
        if not isinstance(a, Mapping) or not isinstance(b, Mapping):
            continue
        for label in set(a) & set(b):
            try:
                delta = abs(float(a[label]) - float(b[label]))
            except (TypeError, ValueError):
                continue
            if math.isfinite(delta):
                deltas.append(delta)
    ref_eval, cand_eval = _evaluate(matched_ref), _evaluate(matched_cand)
    recall_delta = {}
    for label in set(ref_eval["per_class"]) | set(cand_eval["per_class"]):
        left = ref_eval["per_class"].get(label, {}).get("recall")
        right = cand_eval["per_class"].get(label, {}).get("recall")
        recall_delta[label] = right - left if left is not None and right is not None else None
    agreement = sum(a.get("pred") == b.get("pred") for a, b in zip(matched_ref, matched_cand)) / len(ids) if ids else None
    return {
        "matched": len(ids), "reference_only_ids": sorted(ref_ids - cand_ids, key=str),
        "candidate_only_ids": sorted(cand_ids - ref_ids, key=str),
        "agreement": agreement,
        "probability_delta": {"count": len(deltas), "max": max(deltas) if deltas else None, "mean": sum(deltas) / len(deltas) if deltas else None},
        "accuracy_delta": cand_eval["accuracy"] - ref_eval["accuracy"] if ids else None,
        "per_class_recall_delta": recall_delta,
        "reference_accuracy": ref_eval["accuracy"], "candidate_accuracy": cand_eval["accuracy"],
    }
