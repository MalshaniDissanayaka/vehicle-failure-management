"""Create SHAP global explanations and a LIME local explanation.
Example:
 python src/explain_models.py --target condition --model random_forest
 python src/explain_models.py --target fault_type --model xgboost
Run after src/run_ml_pipeline.py.
"""
from pathlib import Path
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'FYP_synthetic_vehicle_fault_dataset_50000_checked.csv'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', choices=['condition','fault_type'], required=True)
    parser.add_argument('--model', choices=['random_forest','xgboost'], default='random_forest')
    parser.add_argument('--sample-count', type=int, default=150)
    args = parser.parse_args()
    bundle_path = ROOT / 'models' / args.target / f'{args.model}.joblib'
    if not bundle_path.exists(): raise FileNotFoundError(f'Model not found: {bundle_path}. Train first.')
    bundle = joblib.load(bundle_path)
    model, features, classes = bundle['model'], bundle['features'], bundle['classes']
    df = pd.read_csv(DATA)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(splitter.split(df[features], df[args.target], groups=df['sequence_id']))
    X_train = df.iloc[train_idx][features]
    X_test = df.iloc[test_idx][features]
    sample = X_test.sample(n=min(args.sample_count, len(X_test)), random_state=42)
    out = ROOT / 'reports' / 'explainable_ai' / args.target / args.model
    out.mkdir(parents=True, exist_ok=True)

    # SHAP: class-specific global feature contribution summaries.
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        explanation = explainer(sample)
        values = np.asarray(explanation.values)
        base_values = np.asarray(explanation.base_values)
        if values.ndim == 3:
            for class_index, class_name in enumerate(classes):
                class_exp = shap.Explanation(
                    values=values[:, :, class_index],
                    base_values=base_values[:, class_index],
                    data=sample.to_numpy(), feature_names=features)
                shap.plots.beeswarm(class_exp, show=False, max_display=len(features))
                import matplotlib.pyplot as plt
                plt.title(f'SHAP summary — {args.target}: {class_name}')
                plt.tight_layout()
                plt.savefig(out / f'shap_beeswarm_{class_name}.png', dpi=180, bbox_inches='tight')
                plt.close()
                shap.plots.waterfall(class_exp[0], show=False, max_display=len(features))
                plt.tight_layout()
                plt.savefig(out / f'shap_waterfall_{class_name}_first_sample.png', dpi=180, bbox_inches='tight')
                plt.close()
        elif values.ndim == 2:
            shap.plots.beeswarm(explanation, show=False, max_display=len(features))
            import matplotlib.pyplot as plt
            plt.tight_layout(); plt.savefig(out/'shap_beeswarm.png', dpi=180, bbox_inches='tight'); plt.close()
        else:
            raise ValueError(f'Unexpected SHAP values shape: {values.shape}')
        print('SHAP plots saved to:', out)
    except ImportError as exc:
        print('SHAP not installed. Run: pip install shap. Details:', exc)

    # LIME: explain one held-out observation locally.
    try:
        from lime.lime_tabular import LimeTabularExplainer
        background = X_train.sample(n=min(5000, len(X_train)), random_state=42).to_numpy()
        lime_explainer = LimeTabularExplainer(
            training_data=background, feature_names=features,
            class_names=[str(c) for c in classes], mode='classification',
            discretize_continuous=True, random_state=42)
        row = X_test.iloc[0]
        def predict_proba(array):
            frame = pd.DataFrame(array, columns=features)
            return model.predict_proba(frame)
        explanation = lime_explainer.explain_instance(
            row.to_numpy(), predict_proba, num_features=len(features), top_labels=1)
        top_label = explanation.top_labels[0]
        explanation.save_to_file(str(out/'lime_first_test_sample.html'))
        readable = explanation.as_list(label=top_label)
        text = [f'Target: {args.target}', f'Model: {args.model}',
                f'Explained sample index in test split: 0',
                f'Predicted class: {classes[top_label]}', '', 'Local feature contributions:']
        text += [f'{rule}: {weight:.6f}' for rule, weight in readable]
        (out/'lime_first_test_sample.txt').write_text('\n'.join(text), encoding='utf-8')
        print('LIME HTML/text explanation saved to:', out)
    except ImportError as exc:
        print('LIME not installed. Run: pip install lime. Details:', exc)

if __name__ == '__main__': main()
