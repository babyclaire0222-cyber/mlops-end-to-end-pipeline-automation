"""Unit tests for src/data_validation.py."""

from __future__ import annotations

import pandas as pd

from src.data_validation import (
    validate_minimum_rows,
    validate_missing_ratio,
    validate_schema,
    validate_target_column,
)


def test_validate_schema_passes_when_columns_present() -> None:
    df = pd.DataFrame({"a": [1], "b": [2]})
    assert validate_schema(df, required_columns=["a", "b"]) == []


def test_validate_schema_fails_when_column_missing() -> None:
    df = pd.DataFrame({"a": [1]})
    failures = validate_schema(df, required_columns=["a", "b"])
    assert len(failures) == 1
    assert "b" in failures[0]


def test_validate_missing_ratio_passes_under_threshold() -> None:
    df = pd.DataFrame({"a": [1, 2, None, 4]})
    assert validate_missing_ratio(df, max_missing_ratio=0.5) == []


def test_validate_missing_ratio_fails_over_threshold() -> None:
    df = pd.DataFrame({"a": [1, None, None, None]})
    failures = validate_missing_ratio(df, max_missing_ratio=0.5)
    assert len(failures) == 1
    assert "a" in failures[0]


def test_validate_minimum_rows_passes() -> None:
    df = pd.DataFrame({"a": range(10)})
    assert validate_minimum_rows(df, min_rows=5) == []


def test_validate_minimum_rows_fails() -> None:
    df = pd.DataFrame({"a": range(2)})
    failures = validate_minimum_rows(df, min_rows=5)
    assert len(failures) == 1


def test_validate_target_column_passes_when_no_missing() -> None:
    df = pd.DataFrame({"target": [0, 1, 0, 1]})
    assert validate_target_column(df, target_column="target") == []


def test_validate_target_column_fails_when_missing_values() -> None:
    df = pd.DataFrame({"target": [0, 1, None, 1]})
    failures = validate_target_column(df, target_column="target")
    assert len(failures) == 1