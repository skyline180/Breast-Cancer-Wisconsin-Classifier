import pandas as pd
import pytest

from src.data.load_data import clean_data, split_features_target
from src.utils.helpers import binary_label_from_diagnosis


def test_binary_label_from_diagnosis():
    assert binary_label_from_diagnosis("M") == 1
    assert binary_label_from_diagnosis("B") == 0
    assert binary_label_from_diagnosis(" m ") == 1


def test_clean_data_drops_unused_columns_and_converts_target():
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "diagnosis": ["M", "B"],
            "feature_1": [10.0, 20.0],
            "Unnamed: 32": [None, None],
        }
    )
    cleaned = clean_data(df)
    assert "id" not in cleaned.columns
    assert "Unnamed: 32" not in cleaned.columns
    assert set(cleaned["diagnosis"].unique()) == {0, 1}


def test_split_features_target():
    df = pd.DataFrame(
        {
            "diagnosis": [1, 0],
            "feature_1": [10.0, 20.0],
            "feature_2": [3.0, 4.0],
        }
    )
    X, y = split_features_target(df)
    assert list(X.columns) == ["feature_1", "feature_2"]
    assert y.tolist() == [1, 0]
