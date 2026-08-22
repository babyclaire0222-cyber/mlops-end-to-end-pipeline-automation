"""
Model registration module.

Responsibilities:
    * Register the trained model (logged under a given MLflow run) to the
      MLflow Model Registry, but ONLY if the evaluation stage's quality
      gates passed.
    * Transition the newly registered model version to the configured
      stage (default: "Staging") using the `MlflowClient` API.
"""

from __future__ import annotations

from typing import Optional

import mlflow
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.config import DotDict
from src.exceptions import ModelRegistrationError
from src.logger import get_logger

logger = get_logger(__name__)


def register_model(
    run_id: str,
    passed_gates: bool,
    config: DotDict,
    artifact_path: str = "model",
) -> Optional[ModelVersion]:
    """
    Register a model from a completed MLflow run to the Model Registry.

    The model is registered under `config.mlflow.registered_model_name` and
    transitioned to `config.mlflow.registry_stage_on_pass` (default
    "Staging") ONLY when `passed_gates` is True. If the quality gates
    failed, registration is skipped entirely and `None` is returned.

    Args:
        run_id: The MLflow run ID that logged the model artifact.
        passed_gates: The `passed_gates` verdict produced by the evaluation
            stage. Registration is gated strictly on this value.
        config: Loaded project configuration.
        artifact_path: The artifact sub-path under the run where the model
            was logged (matches `mlflow.sklearn.log_model` in `train.py`).

    Returns:
        The registered `ModelVersion` if registration occurred, otherwise
        `None` when gates failed.

    Raises:
        ModelRegistrationError: If the registry API calls fail.
    """
    if not passed_gates:
        logger.warning(
            "Skipping model registration: quality gates were not passed for run_id='%s'.",
            run_id,
        )
        return None

    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    client = MlflowClient()
    model_uri = f"runs:/{run_id}/{artifact_path}"
    registered_model_name = config.mlflow.registered_model_name
    target_stage = config.mlflow.registry_stage_on_pass

    try:
        model_version: ModelVersion = mlflow.register_model(
            model_uri=model_uri, name=registered_model_name
        )
        logger.info(
            "Registered model '%s' version %s from run_id='%s'.",
            registered_model_name,
            model_version.version,
            run_id,
        )

        client.transition_model_version_stage(
            name=registered_model_name,
            version=model_version.version,
            stage=target_stage,
            archive_existing_versions=False,
        )
        logger.info(
            "Transitioned model '%s' version %s to stage '%s'.",
            registered_model_name,
            model_version.version,
            target_stage,
        )

        client.update_model_version(
            name=registered_model_name,
            version=model_version.version,
            description=(
                f"Auto-registered by CI/CD pipeline from run_id={run_id}. "
                f"Promoted to '{target_stage}' after passing all quality gates."
            ),
        )

        return model_version

    except MlflowException as exc:
        raise ModelRegistrationError(
            f"Failed to register/transition model '{registered_model_name}': {exc}"
        ) from exc
