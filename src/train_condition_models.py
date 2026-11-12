
"""
Train and evaluate Random Forest and XGBoost on sequence-level condition split.

Synthetic research data only; results do not establish real-world
vehicle diagnostic accuracy.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from xgboost import XGBClassifier

BASE = Path(__file__).resolve().parents[1]
TRAIN_PATH = BASE / 'data' / 'processed' / 'train_condition.csv'
TEST_PATH = BASE / 'data' / 'processed' / 'test_condition.csv'
OUT = BASE / 'artifacts' / 'condition_model_training_20261112'
OUT.mkdir(parents=True, exist_ok=True)
FEATURES = ['engine_coolant_temperature','oil_pressure_kpa','engine_rpm','vehicle_speed','throttle_position','engine_load']
TARGET = 'condition'
GROUP = 'sequence_id'
LABELS = ['normal','warning','critical']

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print("Training dataset shape:", train.shape)
print("Testing dataset shape:", test.shape)


# ============================================================
# 4. VALIDATE DATASETS
# ============================================================

required = FEATURES + [TARGET, GROUP]

for name, df in [("train", train), ("test", test)]:

    missing = sorted(set(required) - set(df.columns))

    if missing:
        raise ValueError(
            f"{name} file missing columns: {missing}"
        )

    if df[required].isna().any().any():
        raise ValueError(
            f"{name} file has missing values in required fields"
        )

    if set(df[TARGET].unique()) != set(LABELS):
        raise ValueError(
            f"{name} labels differ from expected labels: "
            f"{df[TARGET].unique()}"
        )

    if df.empty:
        raise ValueError(f"{name} dataset is empty")

    if not all(
        pd.api.types.is_numeric_dtype(df[col])
        for col in FEATURES
    ):
        raise ValueError(
            f"{name} dataset contains non-numeric features"
        )

    if not np.isfinite(df[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError(
            f"{name} dataset contains infinite feature values"
        )


# ============================================================
# 5. PREVENT SEQUENCE DATA LEAKAGE
# ============================================================

if not set(train[GROUP]).isdisjoint(set(test[GROUP])):

    raise ValueError(
        "Sequence leakage detected: sequence_id overlaps "
        "between train and test"
    )

print("\nSequence-level split validation passed.")

print("Training sequences:", train[GROUP].nunique())
print("Testing sequences:", test[GROUP].nunique())


# ============================================================
# 6. PREPARE FEATURES AND TARGET
# ============================================================

X_train = train[FEATURES]
X_test = test[FEATURES]

y_train = train[TARGET]
y_test = test[TARGET]


# Explicit stable class mapping for multiclass XGBoost
label_to_int = {
    label: i for i, label in enumerate(LABELS)
}

y_train_int = y_train.map(label_to_int)
y_test_int = y_test.map(label_to_int)


# ============================================================
# 7. DEFINE MACHINE LEARNING MODELS
# ============================================================

models = {

    "random_forest": RandomForestClassifier(

        n_estimators=400,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1

    ),

    "xgboost": XGBClassifier(

        objective="multi:softprob",
        num_class=3,
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss",
        tree_method="hist"

    )

}


# ============================================================
# 8. TRAIN AND EVALUATE MODELS
# ============================================================

summary = []
all_reports = {}

for name, model in models.items():

    print(f"\nTraining {name}...", flush=True)

    # Use integer labels for XGBoost
    y_fit = (
        y_train
        if name == "random_forest"
        else y_train_int
    )

    # Train model
    model.fit(X_train, y_fit)

    # Generate predictions
    pred_raw = model.predict(X_test)

    if name == "xgboost":

        pred = (
            pd.Series(pred_raw)
            .map({
                i: label
                for label, i in label_to_int.items()
            })
            .to_numpy()
        )

        proba = model.predict_proba(X_test)
        proba_labels = LABELS

    else:

        pred = pred_raw

        proba = model.predict_proba(X_test)
        proba_labels = list(model.classes_)

    # Reorder probability columns
    proba_df = pd.DataFrame(
        proba,
        columns=proba_labels
    )

    proba_ordered = proba_df[LABELS].to_numpy()

    # Calculate evaluation metrics
    p, r, f, _ = precision_recall_fscore_support(
        y_test,
        pred,
        labels=LABELS,
        average="macro",
        zero_division=0
    )

    # Calculate multiclass ROC-AUC
    try:

        auc_labels = sorted(LABELS)

        auc_probs = proba_df[auc_labels].to_numpy()

        auc = roc_auc_score(
            y_test,
            auc_probs,
            labels=auc_labels,
            multi_class="ovr",
            average="macro"
        )

    except Exception as e:

        warnings.warn(
            f"ROC-AUC unavailable for {name}: {e}"
        )

        auc = None

    # Collect evaluation results
    metrics = {

        "model": name,

        "accuracy": float(
            accuracy_score(y_test, pred)
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, pred)
        ),

        "macro_precision": float(p),

        "macro_recall": float(r),

        "macro_f1": float(f),

        "macro_roc_auc_ovr": (
            float(auc) if auc is not None else None
        ),

        "train_rows": int(len(train)),

        "test_rows": int(len(test)),

        "train_sequences": int(
            train[GROUP].nunique()
        ),

        "test_sequences": int(
            test[GROUP].nunique()
        ),

        "predictor_features": FEATURES,

        "target": TARGET,

        "note": (
            "Synthetic dataset evaluation; not evidence "
            "of real-world Toyota Aqua diagnostic accuracy."
        )

    }

    summary.append(metrics)

    # Classification report
    all_reports[name] = classification_report(
        y_test,
        pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0
    )

    # Save confusion matrix
    cm = confusion_matrix(
        y_test,
        pred,
        labels=LABELS
    )

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{x}" for x in LABELS],
        columns=[f"pred_{x}" for x in LABELS]
    )

    cm_df.to_csv(
        OUT / f"{name}_confusion_matrix.csv"
    )

    # Save test predictions
    pd.DataFrame({

        "y_true": y_test.to_numpy(),

        "y_pred": pred

    }).to_csv(
        OUT / f"{name}_test_predictions.csv",
        index=False
    )

    # Save feature importance
    if hasattr(model, "feature_importances_"):

        importance_df = pd.DataFrame({

            "feature": FEATURES,

            "importance": model.feature_importances_

        }).sort_values(
            "importance",
            ascending=False
        )

        importance_df.to_csv(
            OUT / f"{name}_feature_importance.csv",
            index=False
        )

    # Save trained model in a prediction-compatible bundle

    label_encoder = LabelEncoder()

    # Preserve the same label order used by XGBoost.
    label_encoder.classes_ = np.array(LABELS)

    # Record the probability-column order for each model.
    if name == "xgboost":
        model_classes = LABELS
    else:
        model_classes = list(model.classes_)

    model_bundle = {
        "model": model,
        "target": TARGET,
        "features": FEATURES,
        "classes": model_classes,
        "label_encoder": label_encoder,
        "model_name": name,
        "random_state": 42,
        "data_type": "synthetic"
    }

    joblib.dump(
        model_bundle,
        OUT / f"{name}_condition_model.joblib"
    )

    print(f"Saved compatible model bundle: {name}")

    print(
        json.dumps(metrics, indent=2),
        flush=True
    )


# ============================================================
# 9. SAVE MODEL COMPARISON METRICS
# ============================================================

pd.DataFrame(summary).to_csv(
    OUT / "model_comparison_metrics.csv",
    index=False
)


# ============================================================
# 10. SAVE CLASSIFICATION REPORTS
# ============================================================

with open(
    OUT / "classification_reports.json",
    "w"
) as f:

    json.dump(
        all_reports,
        f,
        indent=2
    )


# ============================================================
# 11. SAVE TRAINING SUMMARY
# ============================================================

training_summary = {

    "target": TARGET,

    "features": FEATURES,

    "labels": LABELS,

    "split": {

        "train_rows": len(train),

        "test_rows": len(test),

        "train_sequences": int(
            train[GROUP].nunique()
        ),

        "test_sequences": int(
            test[GROUP].nunique()
        ),

        "sequence_overlap": len(
            set(train[GROUP]) & set(test[GROUP])
        )

    },

    "metrics": summary,

    "interpretation_warning": (
        "These results evaluate models on synthetic data "
        "generated by rules. Strong metrics may reflect "
        "those rules and must not be presented as "
        "real-world vehicle fault accuracy."
    )

}

with open(
    OUT / "training_summary.json",
    "w"
) as f:

    json.dump(
        training_summary,
        f,
        indent=2
    )


# ============================================================
# 12. TRAINING COMPLETED
# ============================================================

print("\n" + "=" * 60)

print("MODEL TRAINING AND EVALUATION COMPLETED SUCCESSFULLY!")

print("=" * 60)

print("\nSaved outputs to:")

print(OUT)

print("\nModels trained:")

print("- Random Forest")

print("- XGBoost")

print("\nEvaluation reports and trained models saved successfully.")
