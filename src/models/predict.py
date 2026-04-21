"""Run inference with a saved breast cancer classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.utils.helpers import load_object


DROP_COLUMNS = {"diagnosis", "id", "Unnamed: 32"}


def prepare_input_data(input_path: str | Path) -> pd.DataFrame:
    """Load and clean new data for prediction."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns], errors="ignore")
    df = df.dropna(axis=1, how="all")
    return df


def predict(model_path: str | Path, input_path: str | Path) -> pd.DataFrame:
    """Load a model and generate predictions."""
    model = load_object(model_path)
    X = prepare_input_data(input_path)

    proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)

    output = X.copy()
    output["predicted_label"] = pred
    output["predicted_class"] = output["predicted_label"].map({1: "malignant", 0: "benign"})
    output["malignant_probability"] = proba
    return output


def parse_args():
    parser = argparse.ArgumentParser(description="Predict breast cancer class for new samples.")
    parser.add_argument("--model-path", required=True, help="Path to the saved model.")
    parser.add_argument("--input", required=True, help="Path to CSV with new samples.")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV path.")
    return parser.parse_args()


def main():
    args = parse_args()
    pred_df = predict(args.model_path, args.input)
    pred_df.to_csv(args.output, index=False)
    print(json.dumps({"output": args.output, "rows": len(pred_df)}, indent=2))


if __name__ == "__main__":
    main()
