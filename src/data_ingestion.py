"""
Data ingestion module.

Responsibilities:
    * Load raw CSV data from the local filesystem.
    * Handle missing values and scale numeric features.
    * Split the dataset into train/test sets.
    * Upload raw and processed artifacts to AWS S3 via boto3.
    * Download artifacts back from S3 when needed.
"""

from __future__ import annotations

from pathlib import Path

import boto3
import pandas as pd
from botocore.exceptions import BotoCoreError, ClientError
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import DotDict
from src.data_validation import run_validation_pipeline
from src.exceptions import DataIngestionError, S3OperationError
from src.logger import get_logger

logger = get_logger(__name__)


class S3ArtifactManager:
    """Thin wrapper around boto3 for uploading/downloading pipeline artifacts."""

    def __init__(self, config: DotDict) -> None:
        """
        Args:
            config: Loaded project configuration.
        """
        self._config = config
        self._enabled = bool(config.s3.get("enabled", False))
        self._bucket = config.s3.bucket_name
        self._region = config.s3.get("region", "us-east-1")
        self._client = None

        if self._enabled:
            try:
                self._client = boto3.client("s3", region_name=self._region)
            except (BotoCoreError, ClientError) as exc:
                raise S3OperationError(f"Failed to initialize boto3 S3 client: {exc}") from exc

    @property
    def enabled(self) -> bool:
        """Whether S3 integration is turned on in configuration."""
        return self._enabled

    def upload_file(self, local_path: str, s3_key: str) -> None:
        """
        Upload a local file to the configured S3 bucket.

        Args:
            local_path: Path to the local file to upload.
            s3_key: Destination key within the S3 bucket.

        Raises:
            S3OperationError: If the upload fails or S3 is disabled.
        """
        if not self._enabled or self._client is None:
            logger.warning("S3 is disabled in configuration; skipping upload of '%s'.", local_path)
            return

        try:
            self._client.upload_file(Filename=local_path, Bucket=self._bucket, Key=s3_key)
            logger.info("Uploaded '%s' to s3://%s/%s", local_path, self._bucket, s3_key)
        except (BotoCoreError, ClientError, FileNotFoundError) as exc:
            raise S3OperationError(
                f"Failed to upload '{local_path}' to s3://{self._bucket}/{s3_key}: {exc}"
            ) from exc

    def download_file(self, s3_key: str, local_path: str) -> None:
        """
        Download a file from the configured S3 bucket to a local path.

        Args:
            s3_key: Source key within the S3 bucket.
            local_path: Destination path on the local filesystem.

        Raises:
            S3OperationError: If the download fails or S3 is disabled.
        """
        if not self._enabled or self._client is None:
            logger.warning("S3 is disabled in configuration; skipping download of '%s'.", s3_key)
            return

        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            self._client.download_file(Bucket=self._bucket, Key=s3_key, Filename=local_path)
            logger.info("Downloaded s3://%s/%s to '%s'", self._bucket, s3_key, local_path)
        except (BotoCoreError, ClientError) as exc:
            raise S3OperationError(
                f"Failed to download s3://{self._bucket}/{s3_key} to '{local_path}': {exc}"
            ) from exc


def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Load a raw CSV dataset from disk.

    Args:
        file_path: Path to the CSV file.

    Returns:
        The loaded DataFrame.

    Raises:
        DataIngestionError: If the file is missing or cannot be parsed.
    """
    path = Path(file_path)
    if not path.exists():
        raise DataIngestionError(f"Raw data file not found at: {path.resolve()}")

    try:
        df = pd.read_csv(path)
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DataIngestionError(f"Failed to parse CSV at '{path}': {exc}") from exc

    if df.empty:
        raise DataIngestionError(f"Loaded dataset from '{path}' is empty.")

    logger.info("Loaded raw dataset with shape %s from '%s'.", df.shape, path)
    return df


def handle_missing_and_scale(
    df: pd.DataFrame,
    target_column: str,
    missing_value_strategy: str = "median",
    scale_numeric_features: bool = True,
) -> pd.DataFrame:
    """
    Impute missing values in numeric feature columns and optionally scale them.

    The target column is excluded from imputation/scaling to avoid leaking
    transformation logic into the label.

    Args:
        df: Input DataFrame.
        target_column: Name of the label column to exclude from transforms.
        missing_value_strategy: Strategy passed to `SimpleImputer` ("median",
            "mean", or "most_frequent").
        scale_numeric_features: Whether to standardize numeric features.

    Returns:
        A new DataFrame with cleaned (and optionally scaled) features.

    Raises:
        DataIngestionError: If the target column is missing or no numeric
            feature columns are found.
    """
    if target_column not in df.columns:
        raise DataIngestionError(
            f"Target column '{target_column}' not found in dataset columns: {list(df.columns)}"
        )

    df = df.copy()
    feature_columns = [c for c in df.columns if c != target_column]
    numeric_columns = df[feature_columns].select_dtypes(include=["number"]).columns.tolist()

    if not numeric_columns:
        raise DataIngestionError("No numeric feature columns found to impute/scale.")

    imputer = SimpleImputer(strategy=missing_value_strategy)
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
    logger.info(
        "Imputed missing values in %d numeric column(s) using strategy='%s'.",
        len(numeric_columns),
        missing_value_strategy,
    )

    if scale_numeric_features:
        scaler = StandardScaler()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        logger.info("Scaled %d numeric column(s) using StandardScaler.", len(numeric_columns))

    return df


def split_dataset(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into stratified train and test partitions.

    Args:
        df: Cleaned input DataFrame containing features and target.
        target_column: Name of the label column used for stratification.
        test_size: Fraction of data to allocate to the test set.
        random_state: Seed for reproducibility.

    Returns:
        A tuple of (train_df, test_df).

    Raises:
        DataIngestionError: If the split cannot be performed.
    """
    try:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df[target_column],
        )
    except ValueError as exc:
        raise DataIngestionError(f"Failed to split dataset: {exc}") from exc

    logger.info(
        "Split dataset into train=%s and test=%s (test_size=%.2f).",
        train_df.shape,
        test_df.shape,
        test_size,
    )
    return train_df, test_df


def run_ingestion_pipeline(config: DotDict) -> tuple[str, str]:
    """
    Execute the full ingestion stage: load, clean, split, persist, and upload.

    Args:
        config: Loaded project configuration.

    Returns:
        A tuple of (local_train_path, local_test_path).

    Raises:
        DataIngestionError: If any ingestion step fails.
    """
    raw_path = Path(config.paths.raw_data_dir) / config.paths.raw_data_filename
    processed_dir = Path(config.paths.processed_data_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_path = processed_dir / config.paths.train_filename
    test_path = processed_dir / config.paths.test_filename

    df = load_raw_data(str(raw_path))
    run_validation_pipeline(df, config)
    df_clean = handle_missing_and_scale(
        df,
        target_column=config.data.target_column,
        missing_value_strategy=config.data.missing_value_strategy,
        scale_numeric_features=config.data.scale_numeric_features,
    )
    train_df, test_df = split_dataset(
        df_clean,
        target_column=config.data.target_column,
        test_size=config.data.test_size,
        random_state=config.project.random_state,
    )

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    logger.info("Persisted processed train/test CSVs to '%s'.", processed_dir)

    s3_manager = S3ArtifactManager(config)
    if s3_manager.enabled:
        s3_manager.upload_file(str(raw_path), config.s3.raw_data_key)
        s3_manager.upload_file(str(train_path), config.s3.processed_train_key)
        s3_manager.upload_file(str(test_path), config.s3.processed_test_key)

    return str(train_path), str(test_path)
