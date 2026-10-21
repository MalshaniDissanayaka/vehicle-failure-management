# Condition Model Training Report

## Experimental setup
- Target: `condition` (`normal`, `warning`, `critical`)
- Predictors: six sensor/vehicle features only
- Training: 40,000 rows / 400 sequences
- Testing: 10,000 rows / 100 sequences
- Train/test sequence overlap: 0
- Models: Random Forest and XGBoost
- Evaluation: held-out synthetic sequences

## Results

| Model | Accuracy | Balanced accuracy | Macro precision | Macro recall | Macro F1 | Macro ROC-AUC (OvR) |
|---|---:|---:|---:|---:|---:|---:|
| random_forest | 0.9998 | 0.9997 | 0.9996 | 0.9997 | 0.9997 | 1.0000 |
| xgboost | 0.9972 | 0.9933 | 0.9917 | 0.9933 | 0.9925 | 1.0000 |

## Interpretation and limitations

Random Forest scored higher on this held-out synthetic test set. The very high metrics may reflect the simulated rules used to assign `condition` from sensor values. These results must not be presented as proof of real-world Toyota Aqua diagnostic accuracy.

This stage covers condition classification only. `fault_type` classification, early fault detection lead time, SHAP/LIME explanations, and 3D visualisation remain separate future stages.
