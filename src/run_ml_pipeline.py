from __future__ import annotations
from pathlib import Path
from datetime import datetime
import json, logging, sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support, classification_report, confusion_matrix,
    roc_auc_score, ConfusionMatrixDisplay)
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data' / 'FYP_synthetic_vehicle_fault_dataset_50000_checked.csv'
FEATURES = [
    'engine_coolant_temperature', 'oil_pressure_kpa', 'engine_rpm',
    'vehicle_speed', 'throttle_position', 'engine_load'
]
GROUP_COL = 'sequence_id'
TARGETS = ['condition', 'fault_type']
RANDOM_STATE = 42
TEST_SIZE = 0.20


def setup_logging() -> None:
    log_dir = ROOT / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"training_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[logging.FileHandler(log_path, encoding='utf-8'), logging.StreamHandler(sys.stdout)])
    logging.info('Log file: %s', log_path)


def load_and_validate() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f'Dataset not found: {DATA_PATH}\nKeep the CSV at data/{DATA_PATH.name}')
    df = pd.read_csv(DATA_PATH)
    required = FEATURES + [GROUP_COL] + TARGETS
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f'Dataset is missing required columns: {missing}')
    if df[required].isna().any().any():
        raise ValueError('Missing values found in features, targets, or group ID. Investigate before training.')
    if not np.isfinite(df[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError('Non-finite sensor values found (NaN or infinity).')
    if df[GROUP_COL].nunique() < 5:
        raise ValueError('Too few independent sequences for a group-based split.')
    for target in TARGETS:
        if df[target].nunique() < 2:
            raise ValueError(f'Target {target} has fewer than two classes.')
    logging.info('Dataset shape: %s', df.shape)
    logging.info('Unique sequences: %s', df[GROUP_COL].nunique())
    logging.info('Missing values in required columns: %s', int(df[required].isna().sum().sum()))
    logging.info('Exact duplicate rows: %s', int(df.duplicated().sum()))
    return df


def make_models(n_classes: int) -> dict:
    return {
        'random_forest': RandomForestClassifier(
            n_estimators=300, max_features='sqrt', min_samples_leaf=2,
            class_weight='balanced_subsample', random_state=RANDOM_STATE, n_jobs=-1
        ),
        'xgboost': XGBClassifier(
            objective='multi:softprob', num_class=n_classes,
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=RANDOM_STATE, n_jobs=-1, eval_metric='mlogloss',
            tree_method='hist'
        )
    }


def train_one_target(df: pd.DataFrame, target: str) -> list[dict]:
    logging.info('===== TARGET: %s =====', target)
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(df[FEATURES], df[target], groups=df[GROUP_COL]))
    train_df, test_df = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
    train_groups, test_groups = set(train_df[GROUP_COL]), set(test_df[GROUP_COL])
    overlap = train_groups.intersection(test_groups)
    if overlap:
        raise RuntimeError(f'Group leakage detected: {len(overlap)} sequences overlap.')

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_df[target].astype(str))
    known = set(encoder.classes_)
    unseen = set(test_df[target].astype(str)) - known
    if unseen:
        raise ValueError(f'Test split has labels not present in training: {unseen}')
    y_test = encoder.transform(test_df[target].astype(str))
    X_train, X_test = train_df[FEATURES], test_df[FEATURES]
    if set(np.unique(y_train)) != set(range(len(encoder.classes_))):
        raise ValueError(f'Training split does not contain every class for {target}.')
    if set(np.unique(y_test)) != set(range(len(encoder.classes_))):
        raise ValueError(f'Test split does not contain every class for {target}; adjust split strategy.')

    processed_dir = ROOT / 'data' / 'processed'
    model_dir = ROOT / 'models' / target
    report_dir = ROOT / 'reports' / target
    for d in (processed_dir, model_dir, report_dir): d.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(processed_dir / f'{target}_train.csv', index=False)
    test_df.to_csv(processed_dir / f'{target}_test.csv', index=False)
    logging.info('Train: %d rows / %d sequences; test: %d rows / %d sequences; overlap: %d',
        len(train_df), len(train_groups), len(test_df), len(test_groups), len(overlap))
    logging.info('Class labels: %s', list(encoder.classes_))
    logging.info('Train class counts: %s', train_df[target].value_counts().to_dict())
    logging.info('Test class counts: %s', test_df[target].value_counts().to_dict())

    rows = []
    for model_name, model in make_models(len(encoder.classes_)).items():
        logging.info('Training %s for %s ...', model_name, target)
        model.fit(X_train, y_train)
        pred = model.predict(X_test).astype(int)
        proba = model.predict_proba(X_test)
        labels = np.arange(len(encoder.classes_))
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, pred, labels=labels, average='macro', zero_division=0)
        try:
            auc = roc_auc_score(y_test, proba, labels=labels, multi_class='ovr', average='macro')
        except ValueError as exc:
            logging.warning('ROC-AUC unavailable: %s', exc)
            auc = None
        metrics = {
            'target': target, 'model': model_name,
            'accuracy': float(accuracy_score(y_test, pred)),
            'balanced_accuracy': float(balanced_accuracy_score(y_test, pred)),
            'macro_precision': float(precision), 'macro_recall': float(recall),
            'macro_f1': float(f1), 'macro_roc_auc_ovr': None if auc is None else float(auc),
            'train_rows': int(len(train_df)), 'test_rows': int(len(test_df)),
            'train_sequences': int(len(train_groups)), 'test_sequences': int(len(test_groups)),
            'sequence_overlap': int(len(overlap)), 'features': FEATURES,
            'classes': list(encoder.classes_),
            'data_type': 'synthetic',
            'interpretation_warning': 'Synthetic-data evaluation only; not evidence of real-world diagnostic accuracy.'
        }
        rows.append(metrics)
        pred_labels = encoder.inverse_transform(pred)
        true_labels = encoder.inverse_transform(y_test)
        report = classification_report(true_labels, pred_labels,
            labels=list(encoder.classes_), output_dict=True, zero_division=0)
        with open(report_dir / f'{model_name}_classification_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        cm = confusion_matrix(true_labels, pred_labels, labels=list(encoder.classes_))
        pd.DataFrame(cm, index=[f'true_{x}' for x in encoder.classes_],
            columns=[f'pred_{x}' for x in encoder.classes_]).to_csv(report_dir / f'{model_name}_confusion_matrix.csv')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
        fig, ax = plt.subplots(figsize=(7, 6))
        disp.plot(ax=ax, cmap='Blues', colorbar=False, xticks_rotation=30)
        ax.set_title(f'{target} — {model_name} confusion matrix')
        fig.tight_layout()
        fig.savefig(report_dir / f'{model_name}_confusion_matrix.png', dpi=180)
        plt.close(fig)
        pd.DataFrame({'actual': true_labels, 'predicted': pred_labels}).to_csv(
            report_dir / f'{model_name}_test_predictions.csv', index=False)
        if hasattr(model, 'feature_importances_'):
            imp = pd.DataFrame({'feature': FEATURES, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
            imp.to_csv(report_dir / f'{model_name}_feature_importance.csv', index=False)
            fig, ax = plt.subplots(figsize=(8, 5))
            imp.sort_values('importance').plot.barh(x='feature', y='importance', legend=False, ax=ax)
            ax.set_title(f'{target} — {model_name} feature importance')
            ax.set_xlabel('Importance (model-specific)')
            fig.tight_layout()
            fig.savefig(report_dir / f'{model_name}_feature_importance.png', dpi=180)
            plt.close(fig)
        bundle = {
            'model': model, 'target': target, 'features': FEATURES,
            'classes': list(encoder.classes_), 'label_encoder': encoder,
            'model_name': model_name, 'random_state': RANDOM_STATE,
            'data_type': 'synthetic'
        }
        joblib.dump(bundle, model_dir / f'{model_name}.joblib')
        logging.info('%s %s | accuracy %.4f | balanced accuracy %.4f | macro F1 %.4f',
            target, model_name, metrics['accuracy'], metrics['balanced_accuracy'], metrics['macro_f1'])

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(report_dir / 'model_comparison_metrics.csv', index=False)
    with open(report_dir / 'training_summary.json', 'w', encoding='utf-8') as f:
        json.dump({'target': target, 'features': FEATURES, 'classes': list(encoder.classes_),
            'split': {'train_rows': len(train_df), 'test_rows': len(test_df),
                'train_sequences': len(train_groups), 'test_sequences': len(test_groups),
                'sequence_overlap': len(overlap), 'test_size': TEST_SIZE, 'random_state': RANDOM_STATE},
            'metrics': rows,
            'limitations': ['Dataset is synthetic.', 'Labels follow documented simulation rules.',
                'Results do not establish real-world Toyota Aqua diagnostic accuracy.']}, f, indent=2)
    return rows


def main() -> None:
    setup_logging()
    df = load_and_validate()
    all_metrics = []
    for target in TARGETS:
        all_metrics.extend(train_one_target(df, target))
    combined = pd.DataFrame(all_metrics)
    combined.to_csv(ROOT / 'reports' / 'all_model_comparison_metrics.csv', index=False)
    logging.info('All experiments completed. Summary:')
    logging.info('\n%s', combined[['target','model','accuracy','balanced_accuracy','macro_f1']].to_string(index=False))
    logging.info('Models saved under models/<target>/; reports under reports/<target>/.')

if __name__ == '__main__':
    main()
