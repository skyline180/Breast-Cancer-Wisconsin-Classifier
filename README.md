# Breast Cancer Wisconsin Classifier

A complete, reproducible binary classification project for the **Breast Cancer Wisconsin (Diagnostic)** dataset.

## Dataset
Download from:
https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

Place it inside:
data/raw/data.csv

`data/raw/`

The code expects a CSV with columns similar to the standard Kaggle/UCI version:
- `id`
- `diagnosis` (`M` = malignant, `B` = benign)
- 30 numeric feature columns
- an optional `Unnamed: 32` column, which will be dropped automatically if present

## Project workflow
1. Load raw data
2. Clean and split into train/test
3. Build preprocessing pipeline
4. Train and evaluate a classifier
5. Save the trained model
6. Run inference on new samples

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
python -m src.models.train --data data/raw/breast-cancer-wisconsin.csv --model-path models/trained_model.pkl
```

If your CSV file has another name, pass that file path instead.

## Prediction

```bash
python -m src.models.predict --model-path models/trained_model.pkl --input path/to/new_samples.csv
```

The prediction file should contain the same feature columns used during training. The script automatically ignores `id`, `diagnosis`, and `Unnamed: 32` if present.

## Tests

```bash
pytest -q
```

## Notes
- The repository uses a scikit-learn pipeline with imputation and scaling.
- The default model is logistic regression, which is a strong and interpretable baseline for this dataset.
- Evaluation metrics include accuracy, precision, recall, F1-score, ROC AUC, and a confusion matrix.
