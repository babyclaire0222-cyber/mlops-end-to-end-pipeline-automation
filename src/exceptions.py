"""
Custom exception hierarchy for the mlops-end-to-end-pipeline-automation project.

Using specific exception types (rather than bare `Exception`) allows callers
—especially `main.py`—to catch and handle failures at each pipeline stage
distinctly, and produces clearer error messages/logs in production.
"""

from __future__ import annotations


class PipelineError(Exception):
    """Base class for all custom exceptions raised by this pipeline."""


class ConfigurationError(PipelineError):
    """Raised when configuration loading or validation fails."""


class DataIngestionError(PipelineError):
    """Raised when data loading, cleaning, splitting, or S3 I/O fails."""


class S3OperationError(DataIngestionError):
    """Raised specifically when an AWS S3 upload/download operation fails."""


class DataValidationError(DataIngestionError):
    """Raised when raw or processed data fails schema/quality validation checks."""


class TrainingError(PipelineError):
    """Raised when model training or MLflow logging during training fails."""


class EvaluationError(PipelineError):
    """Raised when model evaluation against quality gates fails."""


class ModelRegistrationError(PipelineError):
    """Raised when registering a model to the MLflow Model Registry fails."""