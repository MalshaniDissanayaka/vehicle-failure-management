# Condition Model Training Report

## Experimental setup

- Target: `condition` (`normal`, `warning`, `critical`)
- Predictors: six sensor/vehicle features only
- Training: 40,000 rows / 400 sequences
- Testing: 10,000 rows / 100 sequences
- Train/test sequence overlap: 0
- Models: Random Forest and XGBoost
- Evaluation: held-out synthetic sequences
- Random state: 42

## Results

| Model | Accuracy | Balanced accuracy | Macro precision | Macro recall | Macro F1 | Macro ROC-AUC (OvR) |
|---|---:|---:|---:|---:|---:|---:|
| random_forest | 0.9998 | 0.999747 | 0.999580 | 0.999747 | 0.999663 | 0.999999882 |
| xgboost | 0.9970 | 0.992212 | 0.991561 | 0.992212 | 0.991886 | 0.999970771 |

## Model comparison

Random Forest achieved the highest accuracy, balanced accuracy, macro precision, macro recall, and macro F1 on this held-out synthetic test set. Both models achieved very high ROC-AUC values.

These results indicate that both models learned to classify the generated condition labels effectively. They do not establish that either model can diagnose real vehicle faults accurately.

## Interpretation and limitations

- The dataset is synthetic, and its condition labels are generated using simulated rules.
- High performance may reflect those underlying rules rather than generalisable real-world vehicle behaviour.
- The evaluation used separate training and testing sequences, with zero sequence overlap.
- The results must not be presented as proof of real-world Toyota Aqua diagnostic accuracy.
- Further validation using suitable real-world vehicle data is required.

## Saved outputs

New training outputs are saved in:

`artifacts/condition_model_training_20261112/`

The folder contains trained models, test predictions, classification reports, confusion matrices, feature importance, comparison metrics, and the training summary.

## Project stage

This stage covers condition classification only. `fault_type` classification, early fault detection lead time, SHAP/LIME explanations, and 3D visualisation remain separate future stages.
