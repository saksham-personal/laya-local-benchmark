import math

from laya_bench.metrics import compare_variants, evaluate_quality, selective_report
from laya_bench.reporting import render_html, render_markdown


def row(i, gold, pred, p, split="test", task="binary"):
    return {"id": i, "task": task, "split": split, "gold": gold, "pred": pred,
            "confidence": max(p, 1 - p), "probabilities": {"no": 1 - p, "yes": p}, "variant": "v"}


def test_quality_handles_perfect_and_missing_probability_and_one_class_auc():
    rows = [row("a", "yes", "yes", .9), row("b", "no", "no", .2)]
    result = evaluate_quality(rows)
    assert result["overall"]["accuracy"] == 1.0
    assert result["overall"]["per_class"]["yes"]["recall"] == 1.0
    assert result["overall"]["confusion_matrix"] == [[1, 0], [0, 1]]
    assert result["overall"]["roc_auc"] == 1.0
    missing = rows + [{**row("c", "no", "no", .1), "probabilities": None}]
    assert evaluate_quality(missing)["overall"]["probability_count"] == 2
    one_class = [row("x", "yes", "yes", .9), row("y", "yes", "yes", .8)]
    assert evaluate_quality(one_class)["overall"]["roc_auc"] is None


def test_selective_stable_ties_and_dev_threshold_applied_to_test():
    rows = [row(1, "yes", "yes", .8, "dev"), row(2, "no", "yes", .8, "dev"),
            row(3, "yes", "yes", .9, "test"), row(4, "no", "no", .8, "test"),
            row(5, "no", "yes", .8, "test")]
    result = selective_report(rows, coverages=(.5, 1.0))
    assert result["strategy"] == "dev_threshold"
    half = result["by_split"]["test"][0]
    assert half["threshold"] == .8 and half["accepted"] == 3  # ties pass threshold
    tied_only = selective_report([row(1, "yes", "yes", .7), row(2, "no", "no", .7)], (.5,))
    assert tied_only["by_split"]["test"][0]["accepted"] == 1  # stable row-order top-k


def test_comparison_matches_ids_reports_differences_and_recall_delta():
    left = [row("same", "yes", "yes", .8), row("ref", "no", "no", .2)]
    right = [row("same", "yes", "no", .6), row("cand", "no", "yes", .9)]
    result = compare_variants(left, right)
    assert result["matched"] == 1
    assert result["reference_only_ids"] == ["ref"]
    assert result["candidate_only_ids"] == ["cand"]
    assert result["agreement"] == 0
    assert math.isclose(result["probability_delta"]["mean"], .2)
    assert result["accuracy_delta"] == -1


def test_rendering_preserves_unknowns_and_escapes_html():
    data = {"known": .4, "unavailable": None, "label": "<script>alert(1)</script>"}
    md = render_markdown(data)
    page = render_html(data)
    assert "0.4" in md and "—" in md
    assert "&lt;script&gt;" in page and "<script>" not in page
