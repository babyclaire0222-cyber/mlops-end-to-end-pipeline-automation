"""
Visualization module.

Generates evaluation plots (confusion matrix, metric comparison bar chart,
and — when supported by the model — feature importances) and logs them as
MLflow artifacts, so they're visible alongside the run's metrics in the
MLflow UI without needing a separate dashboard.

Uses matplotlib in the non-interactive "Agg" backend, since this runs in
headless CI/container environments with no display available.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from src.logger import get_logger

logger = get_logger(__name__)


def plot_confusion_matrix(
    model: ClassifierMixin, X_test: pd.DataFrame, y_test: pd.Series, output_path: str
) -> str:
    """
    Render and save a confusion matrix for the model's test-set predictions.

    Args:
        model: A fitted scikit-learn classifier.
        X_test: Held-out feature matrix.
        y_test: Held-out ground-truth labels.
        output_path: File path (including filename) to save the PNG to.

    Returns:
        The path the plot was saved to.
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(confusion_matrix=cm)
    display.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title("Confusion Matrix (Test Set)")
    fig.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)

    logger.info("Saved confusion matrix plot to '%s'.", output_path)
    return output_path


def plot_metric_comparison(metrics: dict, thresholds: dict, output_path: str) -> str:
    """
    Render a bar chart comparing achieved metrics against their quality-gate
    thresholds, so it's visually obvious which (if any) metrics fell short.

    Args:
        metrics: Mapping of metric name -> achieved value (e.g. from
            `evaluate.compute_metrics`).
        thresholds: Mapping of metric name -> required minimum value (e.g.
            from `evaluate.check_quality_gates`).
        output_path: File path (including filename) to save the PNG to.

    Returns:
        The path the plot was saved to.
    """
    labels = list(thresholds.keys())
    achieved = [metrics.get(name, 0.0) for name in labels]
    required = [thresholds[name] for name in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_achieved = ax.bar(x - width / 2, achieved, width, label="Achieved", color="#2a9d8f")
    bars_required = ax.bar(x + width / 2, required, width, label="Required (gate)", color="#e76f51")

    ax.set_ylabel("Score")
    ax.set_title("Evaluation Metrics vs. Quality Gate Thresholds")
    ax.set_xticks(x)
    ax.set_xticklabels([label.replace("test_", "").replace("_", " ").title() for label in labels])
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.bar_label(bars_achieved, fmt="%.2f", padding=3)
    ax.bar_label(bars_required, fmt="%.2f", padding=3)
    fig.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)

    logger.info("Saved metric comparison plot to '%s'.", output_path)
    return output_path


def plot_feature_importance(model: ClassifierMixin, feature_names: list, output_path: str) -> str | None:
    """
    Render a horizontal bar chart of feature importances, if the model
    exposes `feature_importances_` (e.g. tree-based models). Silently
    skipped for model types that don't expose this attribute.

    Args:
        model: A fitted scikit-learn classifier.
        feature_names: Ordered list of feature column names matching the
            model's training data.
        output_path: File path (including filename) to save the PNG to.

    Returns:
        The path the plot was saved to, or None if the model has no
        `feature_importances_` attribute.
    """
    if not hasattr(model, "feature_importances_"):
        logger.info("Model has no feature_importances_ attribute; skipping feature importance plot.")
        return None

    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    top_n = min(20, len(order))
    order = order[:top_n]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.3)))
    ax.barh(
        [feature_names[i] for i in order][::-1],
        [importances[i] for i in order][::-1],
        color="#264653",
    )
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances")
    fig.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)

    logger.info("Saved feature importance plot to '%s'.", output_path)
    return output_path


def generate_and_log_evaluation_plots(
    model: ClassifierMixin,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    metrics: dict,
    thresholds: dict,
    plots_dir: str = "artifacts/plots",
) -> list[str]:
    """
    Generate all evaluation plots and log them as artifacts on the active
    MLflow run, under the "plots" artifact directory.

    Args:
        model: A fitted scikit-learn classifier.
        X_test: Held-out feature matrix.
        y_test: Held-out ground-truth labels.
        metrics: Achieved evaluation metrics.
        thresholds: Quality-gate thresholds for those metrics.
        plots_dir: Local directory to write PNGs to before uploading.

    Returns:
        A list of local file paths for the plots that were generated.
    """
    generated: list[str] = []

    cm_path = plot_confusion_matrix(model, X_test, y_test, f"{plots_dir}/confusion_matrix.png")
    generated.append(cm_path)

    metrics_path = plot_metric_comparison(metrics, thresholds, f"{plots_dir}/metrics_vs_thresholds.png")
    generated.append(metrics_path)

    importance_path = plot_feature_importance(
        model, list(X_test.columns), f"{plots_dir}/feature_importance.png"
    )
    if importance_path:
        generated.append(importance_path)

    for path in generated:
        mlflow.log_artifact(path, artifact_path="plots")

    logger.info("Logged %d evaluation plot(s) to the active MLflow run.", len(generated))
    return generated
