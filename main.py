"""
Unified CLI orchestrator for the mlops-end-to-end-pipeline-automation project.

Chains data ingestion -> training -> evaluation -> registration, each of
which can also be run independently. MLflow run lifecycle is owned here:
`train.py` opens the run, `evaluate.py` logs into it, and this module
closes it before registration (registration re-opens the run implicitly
via the MLflow Model Registry API using the stored `run_id`).

Usage:
    python main.py --run-all
    python main.py --ingest
    python main.py --train
    python main.py --evaluate
    python main.py --register
    python main.py --config config/config.yaml --run-all
"""

from __future__ import annotations

import sys
from typing import Optional

import click
import mlflow

from src.config import Config
from src.data_ingestion import run_ingestion_pipeline
from src.evaluate import run_evaluation_pipeline
from src.exceptions import PipelineError
from src.logger import get_logger
from src.register import register_model
from src.train import run_training_pipeline

logger = get_logger(__name__)


def _ingest_stage(config) -> tuple[str, str]:
    logger.info("=== STAGE: Data Ingestion ===")
    train_path, test_path = run_ingestion_pipeline(config)
    logger.info("Ingestion complete. train='%s' test='%s'", train_path, test_path)
    return train_path, test_path


def _train_stage(config, train_path: str, test_path: str):
    logger.info("=== STAGE: Model Training ===")
    model, X_test, y_test, run_id = run_training_pipeline(config, train_path, test_path)
    logger.info("Training complete. Active MLflow run_id='%s'", run_id)
    return model, X_test, y_test, run_id


def _evaluate_stage(config, model, X_test, y_test, run_id: str) -> bool:
    logger.info("=== STAGE: Evaluation ===")
    result = run_evaluation_pipeline(model, X_test, y_test, config)
    mlflow.end_run(status="FINISHED")
    logger.info(
        "Evaluation complete for run_id='%s'. passed_gates=%s metrics=%s",
        run_id,
        result.passed_gates,
        result.metrics,
    )
    if not result.passed_gates:
        logger.warning("Failed checks: %s", result.failed_checks)
    return result.passed_gates


def _register_stage(config, run_id: str, passed_gates: bool) -> None:
    logger.info("=== STAGE: Model Registration ===")
    model_version = register_model(run_id=run_id, passed_gates=passed_gates, config=config)
    if model_version is None:
        logger.warning("Model was NOT registered because quality gates were not passed.")
    else:
        logger.info(
            "Model registered: name='%s' version=%s stage='%s'",
            config.mlflow.registered_model_name,
            model_version.version,
            config.mlflow.registry_stage_on_pass,
        )


@click.command()
@click.option("--config", "config_path", default="config/config.yaml", show_default=True, help="Path to the YAML configuration file.")
@click.option("--run-all", is_flag=True, help="Run ingestion, training, evaluation, and registration sequentially.")
@click.option("--ingest", is_flag=True, help="Run only the data ingestion stage.")
@click.option("--train", "do_train", is_flag=True, help="Run ingestion + training (training needs processed data).")
@click.option("--evaluate", "do_evaluate", is_flag=True, help="Run ingestion + training + evaluation.")
@click.option("--register", "do_register", is_flag=True, help="Run the full pipeline including registration (equivalent to --run-all).")
def cli(config_path: str, run_all: bool, ingest: bool, do_train: bool, do_evaluate: bool, do_register: bool) -> None:
    """Command-line entrypoint chaining the MLOps pipeline stages."""
    if not any([run_all, ingest, do_train, do_evaluate, do_register]):
        click.echo("No stage flag provided. Use --help to see available options.")
        sys.exit(1)

    try:
        config = Config.load(config_path)

        train_path: Optional[str] = None
        test_path: Optional[str] = None
        model = X_test = y_test = None
        run_id: Optional[str] = None
        passed_gates = False

        needs_ingest = run_all or ingest or do_train or do_evaluate or do_register
        needs_train = run_all or do_train or do_evaluate or do_register
        needs_evaluate = run_all or do_evaluate or do_register
        needs_register = run_all or do_register

        if needs_ingest:
            train_path, test_path = _ingest_stage(config)

        if needs_train:
            model, X_test, y_test, run_id = _train_stage(config, train_path, test_path)

        if needs_evaluate:
            passed_gates = _evaluate_stage(config, model, X_test, y_test, run_id)

        if needs_register:
            _register_stage(config, run_id, passed_gates)

        logger.info("Pipeline execution finished successfully.")

    except PipelineError as exc:
        logger.error("Pipeline execution failed: %s", exc)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001 - top-level safety net for the CLI
        logger.exception("Unexpected error during pipeline execution: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    cli()
