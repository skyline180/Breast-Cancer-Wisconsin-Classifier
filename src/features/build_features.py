"""Feature building and preprocessing pipeline."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_preprocessor(numeric_features: list[str]) -> ColumnTransformer:
    """Create preprocessing for numeric breast-cancer features."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_numeric_feature_names(df) -> list[str]:
    """Return numeric feature columns excluding typical ID/label columns."""
    excluded = {"diagnosis", "id", "Unnamed: 32"}
    numeric_cols = []
    for col in df.columns:
        if col in excluded:
            continue
        if str(df[col].dtype) != "object":
            numeric_cols.append(col)
    return numeric_cols
