"""Train and evaluate Random Forest and XGBoost on sequence-level condition split.
Synthetic research data only; results do not establish real-world vehicle diagnostic accuracy.
"""
from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix, roc_auc_score)
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
required = FEATURES + [TARGET, GROUP]
for name, df in [('train',train),('test',test)]:
    missing = sorted(set(required)-set(df.columns))
    if missing: raise ValueError(f'{name} file missing columns: {missing}')
    if df[required].isna().any().any(): raise ValueError(f'{name} file has missing values in required fields')
    if set(df[TARGET].unique()) != set(LABELS): raise ValueError(f'{name} labels differ from expected labels: {df[TARGET].unique()}')
if not set(train[GROUP]).isdisjoint(set(test[GROUP])):
    raise ValueError('Sequence leakage detected: sequence_id overlaps between train and test')

X_train, X_test = train[FEATURES], test[FEATURES]
y_train, y_test = train[TARGET], test[TARGET]
# Explicit stable class mapping for multiclass XGBoost.
label_to_int = {label:i for i,label in enumerate(LABELS)}
y_train_int = y_train.map(label_to_int)
y_test_int = y_test.map(label_to_int)

models = {
    'random_forest': RandomForestClassifier(
        n_estimators=400, max_features='sqrt', min_samples_leaf=2,
        class_weight='balanced_subsample', random_state=42, n_jobs=-1
    ),
    'xgboost': XGBClassifier(
        objective='multi:softprob', num_class=3, n_estimators=400,
        max_depth=6, learning_rate=0.05, subsample=0.9,
        colsample_bytree=0.9, reg_lambda=1.0, random_state=42,
        n_jobs=-1, eval_metric='mlogloss', tree_method='hist'
    )
}
summary=[]
all_reports={}
for name, model in models.items():
    print(f'Training {name}...', flush=True)
    y_fit = y_train if name == 'random_forest' else y_train_int
    model.fit(X_train, y_fit)
    pred_raw = model.predict(X_test)
    if name == 'xgboost':
        pred = pd.Series(pred_raw).map({i:label for label,i in label_to_int.items()}).to_numpy()
        proba = model.predict_proba(X_test)
        proba_labels = LABELS
    else:
        pred = pred_raw
        proba = model.predict_proba(X_test)
        proba_labels = list(model.classes_)
    # Reorder probability columns into declared label order for comparable AUC.
    proba_df = pd.DataFrame(proba, columns=proba_labels)
    proba_ordered = proba_df[LABELS].to_numpy()
    p, r, f, _ = precision_recall_fscore_support(y_test, pred, labels=LABELS, average='macro', zero_division=0)
    try:
        auc_labels = sorted(LABELS)
        auc_probs = proba_df[auc_labels].to_numpy()
        auc = roc_auc_score(y_test, auc_probs, labels=auc_labels, multi_class='ovr', average='macro')
    except Exception as e:
        warnings.warn(f'ROC-AUC unavailable for {name}: {e}'); auc = None
    metrics = {
        'model': name,
        'accuracy': float(accuracy_score(y_test,pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_test,pred)),
        'macro_precision': float(p), 'macro_recall': float(r), 'macro_f1': float(f),
        'macro_roc_auc_ovr': float(auc) if auc is not None else None,
        'train_rows': int(len(train)), 'test_rows': int(len(test)),
        'train_sequences': int(train[GROUP].nunique()), 'test_sequences': int(test[GROUP].nunique()),
        'predictor_features': FEATURES, 'target': TARGET,
        'note': 'Synthetic dataset evaluation; not evidence of real-world Toyota Aqua diagnostic accuracy.'
    }
    summary.append(metrics)
    all_reports[name] = classification_report(y_test,pred,labels=LABELS,output_dict=True,zero_division=0)
    pd.DataFrame(confusion_matrix(y_test,pred,labels=LABELS),index=[f'true_{x}' for x in LABELS],columns=[f'pred_{x}' for x in LABELS]).to_csv(OUT/f'{name}_confusion_matrix.csv')
    pd.DataFrame({'y_true':y_test.to_numpy(),'y_pred':pred}).to_csv(OUT/f'{name}_test_predictions.csv',index=False)
    if hasattr(model,'feature_importances_'):
        pd.DataFrame({'feature':FEATURES,'importance':model.feature_importances_}).sort_values('importance',ascending=False).to_csv(OUT/f'{name}_feature_importance.csv',index=False)
    joblib.dump(model, OUT/f'{name}_condition_model.joblib')
    print(json.dumps(metrics,indent=2),flush=True)

pd.DataFrame(summary).to_csv(OUT/'model_comparison_metrics.csv',index=False)
with open(OUT/'classification_reports.json','w') as f: json.dump(all_reports,f,indent=2)
with open(OUT/'training_summary.json','w') as f: json.dump({
    'target':TARGET,'features':FEATURES,'labels':LABELS,
    'split':{'train_rows':len(train),'test_rows':len(test),'train_sequences':int(train[GROUP].nunique()),'test_sequences':int(test[GROUP].nunique()),'sequence_overlap':len(set(train[GROUP])&set(test[GROUP]))},
    'metrics':summary,
    'interpretation_warning':'These results evaluate models on synthetic data generated by rules. Strong metrics may reflect those rules and must not be presented as real-world vehicle fault accuracy.'
},f,indent=2)
print('Saved outputs to',OUT)
