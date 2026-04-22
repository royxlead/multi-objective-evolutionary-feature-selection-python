from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
import json
import logging
import platform
import random
import re
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


def set_global_seed(seed: int) -> None:
    """Set global random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)


def ensure_directories(paths: list[Path]) -> None:
    """Create directories if they do not already exist."""

    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def configure_logging(log_file: Path, level: int = logging.INFO) -> logging.Logger:
    """Configure project logger with console and file handlers."""

    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("moefs")
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def slugify(value: str) -> str:
    """Convert names to filesystem-safe slugs."""

    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def save_json(payload: dict[str, Any], output_path: Path) -> None:
    """Persist a dictionary as JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def make_json_serializable(value: Any) -> Any:
    """Recursively convert objects to JSON-serializable values."""

    if is_dataclass(value):
        return make_json_serializable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): make_json_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_serializable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def get_package_versions(packages: list[str]) -> dict[str, str]:
    """Return installed package versions for reproducibility reports."""

    versions: dict[str, str] = {}
    for package_name in packages:
        try:
            versions[package_name] = importlib_metadata.version(package_name)
        except importlib_metadata.PackageNotFoundError:
            versions[package_name] = "not-installed"
    return versions


def collect_environment_metadata() -> dict[str, Any]:
    """Capture Python and OS runtime metadata."""

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def save_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    """Persist a dataframe to CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def dataframe_to_markdown(df: pd.DataFrame, float_precision: int = 4) -> str:
    """Create a markdown table without external dependencies."""

    if df.empty:
        return "| No data |\n|---|\n| - |"

    rounded_df = df.copy()
    for column in rounded_df.select_dtypes(include=["number"]).columns:
        rounded_df[column] = rounded_df[column].map(lambda x: round(float(x), float_precision))

    headers = list(rounded_df.columns)
    lines = [
        "| " + " | ".join(str(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in rounded_df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")

    return "\n".join(lines)
