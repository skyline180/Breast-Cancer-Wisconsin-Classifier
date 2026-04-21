"""Data loading and splitting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.helpers import binary_label_from_diagnosis


DEFAULT_TARGET = "diagnosis"
DEFAULT_DROP_COLUMNS = {"id", "Unnamed: 32"}


def load_raw_data(csv_path: str | Path) -> pd.DataFrame:
    """Load the raw Kaggle/UCI CSV file."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def clean_data(df: pd.DataFrame, target_col: str = DEFAULT_TARGET) -> pd.DataFrame:
    """Drop unused columns, convert target to binary, and remove empty columns."""
    df = df.copy()

    # Drop standard nuisance columns if present
    df = df.drop(columns=[c for c in DEFAULT_DROP_COLUMNS if c in df.columns], errors="ignore")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    # Convert target to binary labels
    df[target_col] = df[target_col].apply(binary_label_from_diagnosis)

    # Remove fully empty columns if any
    df = df.dropna(axis=1, how="all")

    # Remove rows with missing target or all-feature missing rows
    df = df.dropna(subset=[target_col])

    return df


def split_features_target(
    df: pd.DataFrame,
    target_col: str = DEFAULT_TARGET,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate X and y."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    return X, y


def train_val_test_split(
    df: pd.DataFrame,
    target_col: str = DEFAULT_TARGET,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
):
    """Create train/validation/test splits with stratification."""
    X, y = split_features_target(df, target_col=target_col)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), random_state=random_state, stratify=y
    )

    relative_val_size = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - relative_val_size), random_state=random_state, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def load_and_prepare_data(csv_path: str | Path, target_col: str = DEFAULT_TARGET) -> pd.DataFrame:
    """Convenience wrapper to load and clean the dataset."""
    return clean_data(load_raw_data(csv_path), target_col=target_col)
