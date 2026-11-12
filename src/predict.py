
"""Run inference using a saved model bundle.

Supports Random Forest and XGBoost model bundles.
Synthetic research data only; predictions do not establish
real-world vehicle diagnostic accuracy.
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main():

    parser = argparse.ArgumentParser(
        description="Predict vehicle condition using a trained model."
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to saved .joblib model bundle"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to CSV containing vehicle sensor readings"
    )

    args = parser.parse_args()

    # Resolve file paths
    model_path = Path(args.model)
    input_path = Path(args.input)

    if not model_path.is_absolute():
        model_path = ROOT / model_path

    if not input_path.is_absolute():
        input_path = ROOT / input_path

    # Check files exist
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    # Load model bundle
    bundle = joblib.load(model_path)

    model = bundle["model"]
    features = bundle["features"]
    target = bundle["target"]
    classes = bundle["classes"]

    # Read input sensor data
    df = pd.read_csv(input_path)

    # Check required sensor columns
    missing = sorted(set(features) - set(df.columns))

    if missing:
        raise ValueError(
            f"Input CSV missing required columns: {missing}"
        )

    X = df[features]

    # Check missing values
    if X.isna().any().any():
        raise ValueError("Input contains missing sensor values.")

    # Generate predictions
    predictions = model.predict(X)

    # Convert XGBoost numerical predictions into text labels.
    # Random Forest already returns text labels.
    if bundle.get("model_name") == "xgboost":

        label_encoder = bundle["label_encoder"]

        pred_labels = label_encoder.inverse_transform(
            np.asarray(predictions, dtype=int)
        )

    else:

        pred_labels = np.asarray(predictions).astype(str)

    # Generate prediction probabilities
    probabilities = model.predict_proba(X)

    if len(classes) != probabilities.shape[1]:
        raise ValueError(
            "Model classes do not match prediction probability columns."
        )

    # Display results
    print("\n" + "=" * 60)
    print("VEHICLE FAILURE MANAGEMENT - PREDICTION RESULTS")
    print("=" * 60)

    print("Model:", bundle.get("model_name", "Unknown"))
    print("Prediction target:", target)
    print("Number of records:", len(df))

    for i, label in enumerate(pred_labels):

        print("\nRecord", i + 1)
        print(f"Predicted {target}: {label}")

        probability_results = {
            name: round(float(probabilities[i, j]), 4)
            for j, name in enumerate(classes)
        }

        print("Prediction probabilities:", probability_results)

    print("\n" + "=" * 60)
    print("PREDICTION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
