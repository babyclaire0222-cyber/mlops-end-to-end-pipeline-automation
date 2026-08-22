"""Unit tests for src/data_ingestion.py."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_ingestion import handle_missing_and_scale, load_raw_data, split_dataset
from src.exceptions import DataIngestionError


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0, 8.0],
            "feature_b": [10, 20, 30, 40, 50, 60, 70, 80],
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )


def test_load_raw_data_missing_file_raises() -> None:
    with pytest.raises(DataIngestionError):
        load_raw_data("nonexistent/path/to/file.csv")


def test_load_raw_data_success(tmp_path) -> None:
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(csv_path, index=False)

    df = load_raw_data(str(csv_path))

    assert not df.empty
    assert list(df.columns) == ["a", "b"]


def test_handle_missing_and_scale_imputes_and_scales(sample_df: pd.DataFrame) -> None:
    result = handle_missing_and_scale(
        sample_df, target_column="target", missing_value_strategy="median", scale_numeric_features=True
    )

    assert result["feature_a"].isna().sum() == 0
    assert abs(result["feature_a"].mean()) < 1e-6  # standardized -> ~0 mean


def test_handle_missing_and_scale_missing_target_raises(sample_df: pd.DataFrame) -> None:
    with pytest.raises(DataIngestionError):
        handle_missing_and_scale(sample_df, target_column="not_a_column")


def test_split_dataset_respects_test_size(sample_df: pd.DataFrame) -> None:
    train_df, test_df = split_dataset(sample_df, target_column="target", test_size=0.25, random_state=42)

    assert len(train_df) + len(test_df) == len(sample_df)
    assert len(test_df) == 2
