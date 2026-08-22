"""
Model evaluation module.

Responsibilities:
    * Compute standard classification metrics on the held-out test set.
    * Compare those metrics against the quality gate thresholds defined
      in `config/config.yaml`.
    * Log the evaluation metrics to the currently active MLflow run.
    * Return an explicit `passed_gates` boolean verdict used to decide
      whether the model should be registered.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlflow
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.config import DotDict
from src.exceptions import EvaluationError
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    """Structured result of a model evaluation against quality gates."""

    metrics: dict[str, float]
    thresholds: dict[str, float]
    passed_gates: bool
    failed_checks: dict[str, str] = field(default_factory=dict)


def compute_metrics(model: ClassifierMixin, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """
    Compute accuracy, weighted F1, precision, and recall on the test set.

    Args:
        model: A fitted scikit-learn classifier.
        X_test: Held-out feature matrix.
        y_test: Held-out ground-truth labels.

    Returns:
        A dictionary mapping metric name to its computed float value.

    Raises:
        EvaluationError: If prediction or metric computation fails.
    """
    try:
        y_pred = model.predict(X_test)
    except Exception as exc:
        raise EvaluationError(f"Model prediction failed during evaluation: {exc}") from exc

    try:
        metrics = {
            "test_accuracy": float(accuracy_score(y_test, y_pred)),
            "test_f1_score": float(f1_score(y_test, y_pred, average="weighted")),
            "test_precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "test_recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        }
    except ValueError as exc:
        raise EvaluationError(f"Failed to compute evaluation metrics: {exc}") from exc

    logger.info("Computed evaluation metrics: %s", metrics)
    return metrics


def check_quality_gates(metrics: dict[str, float], config: DotDict) -> EvaluationResult:
    """
    Compare computed metrics against configured minimum thresholds.

    Args:
        metrics: Dictionary of computed evaluation metrics (see
            `compute_metrics`).
        config: Loaded project configuration containing `quality_gates`.

    Returns:
        An `EvaluationResult` capturing the pass/fail verdict and details
        of any failed checks.
    """
    thresholds = {
        "test_accuracy": config.quality_gates.min_accuracy,
        "test_f1_score": config.quality_gates.min_f1_score,
        "test_precision": config.quality_gates.min_precision,
        "test_recall": config.quality_gates.min_recall,
    }

    failed_checks: dict[str, str] = {}
    for metric_name, threshold in thresholds.items():
        actual_value = metrics.get(metric_name)
        if actual_value is None:
            failed_checks[metric_name] = "metric not computed"
            continue
        if actual_value < threshold:
            failed_checks[metric_name] = f"{actual_value:.4f} < required {threshold:.4f}"

    passed_gates = len(failed_checks) == 0

    if passed_gates:
        logger.info("All quality gates PASSED.")
    else:
        logger.warning("Quality gates FAILED: %s", failed_checks)

    return EvaluationResult(
        metrics=metrics,
        thresholds=thresholds,
        passed_gates=passed_gates,
        failed_checks=failed_checks,
    )


def run_evaluation_pipeline(
    model: ClassifierMixin, X_test: pd.DataFrame, y_test: pd.Series, config: DotDict
) -> EvaluationResult:
    """
    Evaluate a trained model and log the results to the active MLflow run.

    This function assumes an MLflow run is already active (started by
    `train.run_training_pipeline`) and logs into that same run so that
    training and evaluation metrics live together.

    Args:
        model: A fitted scikit-learn classifier.
        X_test: Held-out feature matrix.
        y_test: Held-out ground-truth labels.
        config: Loaded project configuration.

    Returns:
        The `EvaluationResult`, including the explicit `passed_gates` verdict.

    Raises:
        EvaluationError: If metric computation fails.
    """
    metrics = compute_metrics(model, X_test, y_test)
    result = check_quality_gates(metrics, config)

    try:
        mlflow.log_metrics(result.metrics)
        mlflow.log_param("passed_gates", result.passed_gates)
        if result.failed_checks:
            mlflow.set_tag("failed_checks", str(result.failed_checks))
    except Exception as exc:
        raise EvaluationError(f"Failed to log evaluation results to MLflow: {exc}") from exc

    return result
