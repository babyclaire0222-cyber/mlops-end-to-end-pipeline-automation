"""
Decoupled configuration management.

Loads `config/config.yaml` into a typed, dot-accessible object so that
every pipeline module reads settings the same way instead of parsing YAML
independently. Environment variables (loaded via python-dotenv) take
precedence for secrets such as AWS credentials, which are never stored
in the YAML file itself.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from src.exceptions import ConfigurationError
from src.logger import get_logger

logger = get_logger(__name__)

DEFAULT_CONFIG_PATH = os.getenv("CONFIG_PATH", "config/config.yaml")


class DotDict(dict):
    """A dictionary that also supports attribute-style (dot) access, recursively."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(
                f"Configuration key '{item}' was not found."
            ) from exc
        if isinstance(value, dict) and not isinstance(value, DotDict):
            value = DotDict(value)
            self[item] = value
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


class Config:
    """
    Loads and validates the project's YAML configuration file.

    Usage:
        config = Config.load()
        bucket = config.s3.bucket_name
        min_acc = config.quality_gates.min_accuracy
    """

    _REQUIRED_TOP_LEVEL_KEYS = (
        "project",
        "paths",
        "s3",
        "data",
        "model",
        "mlflow",
        "quality_gates",
    )

    @classmethod
    def load(cls, config_path: str = DEFAULT_CONFIG_PATH) -> DotDict:
        """
        Load environment variables and the YAML configuration file.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            A `DotDict` exposing configuration values via attribute access.

        Raises:
            ConfigurationError: If the file is missing, malformed, or fails
                validation of required top-level sections.
        """
        load_dotenv()  # populate os.environ from a local .env file, if present

        path = Path(config_path)
        if not path.exists():
            raise ConfigurationError(f"Configuration file not found at: {path.resolve()}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                raw_config: dict[str, Any] = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ConfigurationError(f"Failed to parse YAML configuration: {exc}") from exc

        if not raw_config:
            raise ConfigurationError("Configuration file is empty.")

        cls._validate(raw_config)
        logger.info("Configuration successfully loaded from '%s'.", path)
        return DotDict(raw_config)

    @classmethod
    def _validate(cls, raw_config: dict[str, Any]) -> None:
        """Ensure all required top-level configuration sections are present."""
        missing = [key for key in cls._REQUIRED_TOP_LEVEL_KEYS if key not in raw_config]
        if missing:
            raise ConfigurationError(
                f"Configuration is missing required top-level section(s): {missing}"
            )
