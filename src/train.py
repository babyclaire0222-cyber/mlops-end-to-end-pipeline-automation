"""
Model training module.

Responsibilities:
    * Train a scikit-learn classifier (RandomForest or GradientBoosting)
      on the processed training set.
    * Run stratified cross-validation for a robust performance estimate.
    * Log hyperparameters, execution time, CV scores, and the model binary
      to MLflow inside a single `mlflow.start_run()` context.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.config import DotDict
from src.exceptions import TrainingError
from src.logger import get_logger

logger = get_logger(__name__)

_SUPPORTED_MODELS = {
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
}


def build_model(model_type: str, hyperparameters: dict[str, Any], random_state: int) -> ClassifierMixin:
    """
    Instantiate a scikit-learn classifier based on configuration.

    Args:
        model_type: One of the keys in `_SUPPORTED_MODELS` ("random_forest",
            "gradient_boosting").
        hyperparameters: Keyword arguments forwarded to the estimator.
        random_state: Seed for reproducibility.

    Returns:
        An unfitted scikit-learn classifier instance.

    Raises:
        TrainingError: If `model_type` is not supported.
    """
    if model_type not in _SUPPORTED_MODELS:
        raise TrainingError(
            f"Unsupported model_type '{model_type}'. Supported types: {list(_SUPPORTED_MODELS)}"
        )

    estimator_cls = _SUPPORTED_MODELS[model_type]
    model = estimator_cls(random_state=random_state, **hyperparameters)
    logger.info("Instantiated model '%s' with hyperparameters: %s", model_type, hyperparameters)
    return model


def load_train_test_data(
    train_path: str, test_path: str, target_column: str
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load processed train/test CSVs and split them into features and target.

    Args:
        train_path: Path to the processed training CSV.
        test_path: Path to the processed test CSV.
        target_column: Name of the label column.

    Returns:
        A tuple of (X_train, y_train, X_test, y_test).

    Raises:
        TrainingError: If files are missing or malformed.
    """
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    except (FileNotFoundError, pd.errors.ParserError) as exc:
        raise TrainingError(f"Failed to load train/test data: {exc}") from exc

    if target_column not in train_df.columns or target_column not in test_df.columns:
        raise TrainingError(f"Target column '{target_column}' missing from train/test data.")

    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    return X_train, y_train, X_test, y_test


def run_training_pipeline(
    config: DotDict, train_path: str, test_path: str
) -> tuple[ClassifierMixin, pd.DataFrame, pd.Series, str]:
    """
    Train a classifier inside an MLflow run, logging params/metrics/artifacts.

    Args:
        config: Loaded project configuration.
        train_path: Path to the processed training CSV.
        test_path: Path to the processed test CSV.

    Returns:
        A tuple of (fitted_model, X_test, y_test, active_run_id). The MLflow
        run is left active (not ended) so that `evaluate.py` can log
        additional metrics into the same run before it is closed by the
        orchestrator.

    Raises:
        TrainingError: If training fails at any step.
    """
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)

    X_train, y_train, X_test, y_test = load_train_test_data(
        train_path, test_path, config.data.target_column
    )

    model = build_model(
        model_type=config.model.type,
        hyperparameters=dict(config.model.hyperparameters),
        random_state=config.project.random_state,
    )

    run = mlflow.start_run(run_name=f"{config.model.type}_training_run")
    try:
        logger.info("Started MLflow run: %s", run.info.run_id)

        mlflow.log_param("model_type", config.model.type)
        for param_name, param_value in config.model.hyperparameters.items():
            mlflow.log_param(param_name, param_value)
        mlflow.log_param("test_size", config.data.test_size)
        mlflow.log_param("random_state", config.project.random_state)

        cv = StratifiedKFold(
            n_splits=config.model.cross_validation.n_splits,
            shuffle=True,
            random_state=config.project.random_state,
        )

        start_time = time.perf_counter()
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=cv, scoring=config.model.cross_validation.scoring
        )
        model.fit(X_train, y_train)
        elapsed_seconds = time.perf_counter() - start_time

        mlflow.log_metric("cv_mean_score", float(cv_scores.mean()))
        mlflow.log_metric("cv_std_score", float(cv_scores.std()))
        mlflow.log_metric("training_execution_time_seconds", elapsed_seconds)
        for fold_idx, score in enumerate(cv_scores):
            mlflow.log_metric(f"cv_fold_{fold_idx}_score", float(score))

        logger.info(
            "Training complete in %.2fs | CV mean=%.4f std=%.4f",
            elapsed_seconds,
            cv_scores.mean(),
            cv_scores.std(),
        )

        model_dir = Path(config.paths.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / config.paths.model_filename
        joblib.dump(model, model_path)
        logger.info("Serialized trained model to '%s'.", model_path)

        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_artifact(str(model_path))

        return model, X_test, y_test, run.info.run_id

    except Exception as exc:
        mlflow.end_run(status="FAILED")
        raise TrainingError(f"Training pipeline failed: {exc}") from exc
