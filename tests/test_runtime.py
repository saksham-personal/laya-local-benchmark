from __future__ import annotations

import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from laya_bench.runtime import decode_answer, predict_row


class FakeAgent:
    def __init__(self, answer: dict):
        self.answer = answer

    def system_one(self, state, questions, **kwargs):
        assert state and "decision" in questions
        return {"answers": {"decision": self.answer}, "usage": {"input_tokens": 47}}


def test_choice_uses_answer_confidence_and_label_order():
    answer = {"type": "choice", "choice": "semantic_retrieval",
              "probabilities": {"structured_database": .2, "semantic_retrieval": .8},
              "confidence": .3, "answer_confidence": .8}
    row = {"id": "one", "task": "search_type", "split": "test", "gold": "semantic_retrieval",
           "state": "Find firms in descriptions", "question": {"decision": {"type": "choice"}}}
    prediction = predict_row(FakeAgent(answer), row, variant="english/onnx_fp32")
    assert prediction["pred"] == "semantic_retrieval"
    assert prediction["confidence"] == .8
    assert prediction["probabilities"]["structured_database"] == .2
    assert prediction["input_tokens"] == 47


def test_noul_normalization():
    pred, confidence, probs = decode_answer({"type": "noul", "noul": .18,
                                             "answer_confidence": .82})
    assert pred == "no"
    assert confidence == .82
    assert probs == pytest.approx({"no": .82, "yes": .18})
