"""Unit tests for src/evaluate.py."""

from __future__ import annotations

from src.config import DotDict
from src.evaluate import check_quality_gates


def make_config(min_accuracy: float, min_f1: float, min_precision: float, min_recall: float) -> DotDict:
    return DotDict(
        {
            "quality_gates": {
                "min_accuracy": min_accuracy,
                "min_f1_score": min_f1,
                "min_precision": min_precision,
                "min_recall": min_recall,
            }
        }
    )


def test_check_quality_gates_all_pass() -> None:
    metrics = {
        "test_accuracy": 0.95,
        "test_f1_score": 0.90,
        "test_precision": 0.92,
        "test_recall": 0.88,
    }
    config = make_config(0.80, 0.75, 0.70, 0.70)

    result = check_quality_gates(metrics, config)

    assert result.passed_gates is True
    assert result.failed_checks == {}


def test_check_quality_gates_fails_on_low_accuracy() -> None:
    metrics = {
        "test_accuracy": 0.50,
        "test_f1_score": 0.90,
        "test_precision": 0.92,
        "test_recall": 0.88,
    }
    config = make_config(0.80, 0.75, 0.70, 0.70)

    result = check_quality_gates(metrics, config)

    assert result.passed_gates is False
    assert "test_accuracy" in result.failed_checks


def test_check_quality_gates_missing_metric_fails() -> None:
    metrics = {"test_accuracy": 0.95}
    config = make_config(0.80, 0.75, 0.70, 0.70)

    result = check_quality_gates(metrics, config)

    assert result.passed_gates is False
    assert "test_f1_score" in result.failed_checks
