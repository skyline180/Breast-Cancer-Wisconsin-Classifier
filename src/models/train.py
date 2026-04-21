"""Train a breast cancer classifier and save the model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.data.load_data import load_and_prepare_data, train_val_test_split
from src.features.build_features import build_preprocessor, get_numeric_feature_names
from src.utils.helpers import evaluate_binary_classifier, ensure_dir, save_object


def build_model_pipeline(feature_names: list[str]) -> Pipeline:
    """Create preprocessing + classifier pipeline."""
    preprocessor = build_preprocessor(feature_names)

    classifier = LogisticRegression(
        max_iter=5000,
        solver="liblinear",
        class_weight="balanced",
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def train_model(
    data_path: str | Path,
    model_path: str | Path = "models/trained_model.pkl",
    report_dir: str | Path = "reports",
    random_state: int = 42,
):
    """Train the model, evaluate it, and save outputs."""
    data_path = Path(data_path)
    model_path = Path(model_path)
    report_dir = Path(report_dir)
    ensure_dir(model_path.parent)
    ensure_dir(report_dir)
    ensure_dir(report_dir / "figures")

    df = load_and_prepare_data(data_path)
    feature_names = get_numeric_feature_names(df)

    if not feature_names:
        raise ValueError("No numeric feature columns were found in the dataset.")

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        df, random_state=random_state
    )

    X_train_full = pd.concat([X_train, X_val], axis=0)
    y_train_full = pd.concat([y_train, y_val], axis=0)

    pipeline = build_model_pipeline(feature_names)

    param_grid = {
        "classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    )
    search.fit(X_train_full, y_train_full)

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    metrics = evaluate_binary_classifier(y_test, y_pred, y_proba)
    metrics["best_params"] = search.best_params_
    metrics["classification_report"] = classification_report(y_test, y_pred, output_dict=True)
    metrics["validation_roc_auc"] = float(search.best_score_)

    save_object(best_model, model_path)

    with open(report_dir / "results.md", "w", encoding="utf-8") as f:
        f.write("# Model Results\n\n")
        f.write(f"- Best parameters: `{search.best_params_}`\n")
        f.write(f"- Validation ROC AUC: `{search.best_score_:.4f}`\n")
        f.write(f"- Test Accuracy: `{metrics['accuracy']:.4f}`\n")
        f.write(f"- Test Precision: `{metrics['precision']:.4f}`\n")
        f.write(f"- Test Recall: `{metrics['recall']:.4f}`\n")
        f.write(f"- Test F1: `{metrics['f1']:.4f}`\n")
        f.write(f"- Test ROC AUC: `{metrics.get('roc_auc'):.4f}`\n")
        f.write("\n## Confusion Matrix\n\n")
        f.write(f"```text\n{metrics['confusion_matrix']}\n```\n")

    plt.figure(figsize=(7, 6))
    RocCurveDisplay.from_estimator(best_model, X_test, y_test)
    plt.title("ROC Curve - Breast Cancer Wisconsin Classifier")
    plt.tight_layout()
    plt.savefig(report_dir / "figures" / "roc_curve.png", dpi=200)
    plt.close()

    metrics_path = report_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "results_path": str(report_dir / "results.md"),
        "metrics": metrics,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train a breast cancer classifier.")
    parser.add_argument("--data", required=True, help="Path to the raw CSV dataset.")
    parser.add_argument("--model-path", default="models/trained_model.pkl", help="Output model path.")
    parser.add_argument("--report-dir", default="reports", help="Directory to save reports.")
    return parser.parse_args()


def main():
    args = parse_args()
    output = train_model(args.data, args.model_path, args.report_dir)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
