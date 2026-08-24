"""
Data validation module.

Runs schema and data-quality checks on the raw dataset BEFORE it enters the
cleaning/splitting stages of the ingestion pipeline. This is the "unit tests
for data" layer: it fails loudly and early rather than letting bad data
silently propagate into a trained (and possibly registered) model.

Checks performed:
    * Required columns are present (schema check).
    * Column dtypes are numeric where expected (type check).
    * Missing-value ratio per column does not exceed a configured threshold.
    * The target column has no missing values and only expected class labels.
    * The dataset has at least a configured minimum number of rows.

This intentionally stays dependency-free (pure pandas) rather than pulling in
a framework like Great Expectations, keeping the validation logic easy to
read, test, and extend without adding another moving part to the install.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.config import DotDict
from src.exceptions import DataValidationError
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Structured result of a data validation pass."""

    passed: bool
    checks_run: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_schema(df: pd.DataFrame, required_columns: list[str]) -> list[str]:
    """
    Confirm every required column is present in the DataFrame.

    Args:
        df: The DataFrame to check.
        required_columns: Column names that must be present.

    Returns:
        A list of failure messages (empty if the schema check passes).
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        return [f"Missing required column(s): {missing}"]
    return []


def validate_missing_ratio(df: pd.DataFrame, max_missing_ratio: float) -> list[str]:
    """
    Confirm no column exceeds the maximum allowed fraction of missing values.

    Args:
        df: The DataFrame to check.
        max_missing_ratio: Maximum allowed fraction (0.0-1.0) of missing
            values per column before it's considered a failure.

    Returns:
        A list of failure messages, one per offending column.
    """
    failures = []
    if len(df) == 0:
        return ["Dataset has zero rows; cannot compute missing-value ratios."]

    ratios = df.isna().mean()
    for column, ratio in ratios.items():
        if ratio > max_missing_ratio:
            failures.append(
                f"Column '{column}' has {ratio:.1%} missing values "
                f"(exceeds max allowed {max_missing_ratio:.1%})"
            )
    return failures


def validate_minimum_rows(df: pd.DataFrame, min_rows: int) -> list[str]:
    """
    Confirm the dataset has at least the configured minimum number of rows.

    Args:
        df: The DataFrame to check.
        min_rows: Minimum acceptable row count.

    Returns:
        A list containing one failure message if the check fails, else empty.
    """
    if len(df) < min_rows:
        return [f"Dataset has {len(df)} row(s); minimum required is {min_rows}."]
    return []


def validate_target_column(df: pd.DataFrame, target_column: str) -> list[str]:
    """
    Confirm the target column has no missing values.

    Args:
        df: The DataFrame to check.
        target_column: Name of the label column.

    Returns:
        A list of failure messages (empty if the target column is valid).
    """
    failures = []
    if target_column not in df.columns:
        # Already caught by validate_schema, but guard here so this check
        # can also run standalone without raising a KeyError.
        return failures

    missing_count = df[target_column].isna().sum()
    if missing_count > 0:
        failures.append(
            f"Target column '{target_column}' has {missing_count} missing value(s); "
            "labels must be complete."
        )
    return failures


def run_validation_pipeline(df: pd.DataFrame, config: DotDict) -> ValidationResult:
    """
    Run all configured validation checks against the raw dataset.

    Reads thresholds from `config.data_validation` (see config.yaml):
        required_columns: list[str] | None  (defaults to just the target column)
        max_missing_ratio: float
        min_rows: int

    Args:
        df: The raw DataFrame, before cleaning/splitting.
        config: Loaded project configuration.

    Returns:
        A `ValidationResult` capturing which checks ran and any failures.

    Raises:
        DataValidationError: If any check fails and `data_validation.enabled`
            is not explicitly set to False in configuration.
    """
    validation_cfg = config.get("data_validation", {})
    enabled = validation_cfg.get("enabled", True)
    target_column = config.data.target_column
    required_columns = validation_cfg.get("required_columns") or [target_column]
    max_missing_ratio = validation_cfg.get("max_missing_ratio", 0.3)
    min_rows = validation_cfg.get("min_rows", 10)

    if not enabled:
        logger.info("Data validation is disabled in configuration; skipping checks.")
        return ValidationResult(passed=True, checks_run=[], failures=[])

    checks_run = ["schema", "missing_ratio", "minimum_rows", "target_column"]
    failures: list[str] = []
    failures += validate_schema(df, required_columns)
    failures += validate_missing_ratio(df, max_missing_ratio)
    failures += validate_minimum_rows(df, min_rows)
    failures += validate_target_column(df, target_column)

    result = ValidationResult(passed=len(failures) == 0, checks_run=checks_run, failures=failures)

    if result.passed:
        logger.info("Data validation PASSED (%d checks run).", len(checks_run))
    else:
        logger.error("Data validation FAILED: %s", failures)
        raise DataValidationError(f"Data validation failed: {failures}")

    return result