"""
Utility script to generate a synthetic binary-classification CSV dataset
so the pipeline can be run end-to-end without external data.

Usage:
    python scripts/generate_sample_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from src.config import Config
from src.logger import get_logger

logger = get_logger(__name__)


def generate_sample_dataset(output_path: str, n_samples: int = 2000, random_state: int = 42) -> None:
    """
    Generate and persist a synthetic classification dataset with a few
    injected missing values, so the ingestion module's imputation logic
    has something to do.

    Args:
        output_path: Destination CSV path.
        n_samples: Number of rows to generate.
        random_state: Seed for reproducibility.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        weights=[0.6, 0.4],
        random_state=random_state,
    )

    columns = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=columns)
    df["target"] = y

    rng = np.random.default_rng(random_state)
    missing_mask = rng.random(size=df.shape[0]) < 0.03
    df.loc[missing_mask, "feature_0"] = np.nan

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Generated synthetic dataset with shape %s at '%s'.", df.shape, path)


if __name__ == "__main__":
    config = Config.load()
    out_path = Path(config.paths.raw_data_dir) / config.paths.raw_data_filename
    generate_sample_dataset(str(out_path))
