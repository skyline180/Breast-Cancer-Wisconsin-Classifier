"""Utility helpers for the breast cancer classifier project."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it as Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def project_path(*parts: str | Path) -> Path:
    """Build a path relative to the repository root."""
    return PROJECT_ROOT.joinpath(*parts)


def save_object(obj, path: str | Path) -> Path:
    """Persist a Python object with joblib."""
    path = Path(path)
    ensure_dir(path.parent)
    joblib.dump(obj, path)
    return path


def load_object(path: str | Path):
    """Load a joblib-serialized Python object."""
    return joblib.load(Path(path))


def evaluate_binary_classifier(y_true, y_pred, y_proba=None) -> dict:
    """Compute standard binary classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics["roc_auc"] = None
    return metrics


def binary_label_from_diagnosis(value) -> int:
    """Convert M/B labels to 1/0 for malignant/benign."""
    if pd.isna(value):
        raise ValueError("Diagnosis contains missing values.")
    value = str(value).strip().upper()
    if value == "M":
        return 1
    if value == "B":
        return 0
    raise ValueError(f"Unexpected diagnosis label: {value!r}")


def coerce_numeric_frame(df: pd.DataFrame, exclude: Sequence[str] | None = None) -> pd.DataFrame:
    """Return a numeric-only dataframe after dropping excluded columns."""
    exclude = set(exclude or [])
    numeric_df = df.copy()
    for col in list(numeric_df.columns):
        if col in exclude:
            continue
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors="coerce")
    return numeric_df.drop(columns=[c for c in numeric_df.columns if c in exclude], errors="ignore")


def summarize_dataframe(df: pd.DataFrame) -> dict:
    """A small summary useful for notebooks and reports."""
    return {
        "shape": df.shape,
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": list(df.columns),
    }
