# 🚗 From Black-Box Predictions to Visual Intelligence
### A Synthetic Vehicle Fault Dataset for Explainable AI and Vehicle Failure Management

**Research Project | BSc (Hons) Data Science | Final Year Project**

## 📌 About the Project

Vehicle faults can be difficult to understand when diagnostic systems provide sensor readings or fault predictions without clear explanations. This research project explores how Machine Learning (ML), Explainable AI (XAI), and 3D visualisation can support a more understandable approach to vehicle failure management.

As an initial step, this repository contains a **synthetic vehicle fault dataset** designed to support the development and evaluation of machine learning models for two fault categories:

- 🌡️ **Engine Overheating**
- 🛢️ **Low Engine Oil Pressure**

The dataset is designed around a Toyota Aqua / Toyota Prius c reference context. It is simulated research data, not a collection of real Toyota vehicle fault records.

## 🎯 Project Objectives

- Generate synthetic vehicle sensor data for normal and fault-related operating conditions.
- Develop ML models to identify potential vehicle fault conditions.
- Evaluate model performance using appropriate classification metrics.
- Explore SHAP and LIME to explain model predictions.
- Support the development of an interpretable vehicle failure management prototype with 3D visualisation.

*This repository section focuses on the synthetic dataset. Model training, explainability, and visualisation are subsequent project stages.*

## 📂 Dataset Files

| File | Description |
|---|---|
| `FYP_synthetic_vehicle_fault_dataset_50000_checked.csv` | Main synthetic dataset containing 50,000 records |
| `FYP_synthetic_dataset_checked_methodology.csv` | Documents data sources, assumptions, and generation rules |
| `FYP_synthetic_dataset_checked_validation_report.json` | Records validation checks and baseline comparisons |

## 📊 Dataset Overview

| Property | Description |
|---|---|
| Number of records | 50,000 |
| Sensor/vehicle features | 6 |
| Fault categories | 2 |
| Condition labels | `normal`, `warning`, `critical` |
| Data type | Synthetic |
| Intended use | ML development and research experimentation |

### Sensor and Vehicle Features

The six predictor features are:

1. Engine coolant temperature
2. Engine oil pressure
3. Engine RPM
4. Vehicle speed
5. Throttle position
6. Engine load

Check the dataset and methodology documentation for the exact column names and units before using the features in code.

### Target Variables

Choose the target according to the research question:

- **`fault_type`** — for classifying the fault category.
- **`condition`** — for classifying the operating condition as normal, warning, or critical.

These targets represent different prediction tasks. Select one for a particular experiment and document the choice.

## 🧪 Data Generation and Validation

The dataset is designed using source-data distributions, simulated fault scenarios, and documented modelling assumptions.

The accompanying validation report records **41/41 programmed checks passed**. These checks support the assessment of dataset structure and internal consistency; they do not establish real-world diagnostic accuracy.

Before reporting model results, review both:

- `FYP_synthetic_dataset_checked_methodology.csv`
- `FYP_synthetic_dataset_checked_validation_report.json`

## 🤖 Guidelines for Machine Learning

For a reliable experimental workflow:

- Use only the six sensor/vehicle features as predictors.
- Select either `fault_type` or `condition` as the target.
- Split training and testing data by `sequence_id`, rather than randomly splitting individual rows.
- Exclude `sequence_id`, `timestamp_step`, `scenario_type`, and `fault_active` from the predictor features.
- Fit preprocessing steps using the training data only.
- Evaluate the selected model on held-out sequences.

Recommended classification metrics include accuracy, precision, recall, F1-score, ROC-AUC where appropriate, and a confusion matrix. Early fault detection lead time can also be evaluated if the prediction setup supports it.

## ⚠️ Scientific Limitations

This dataset is intended for **synthetic research and model development**.

- It has not been validated against real Toyota Aqua fault logs.
- The source datasets are not necessarily Toyota-specific.
- Some fault thresholds are modelling assumptions, not official Toyota warning or critical limits.
- Baseline distribution similarity and internal consistency do not guarantee that the simulated faults reproduce real vehicle behaviour.
- High model performance on this dataset alone should not be interpreted as proof of real-world vehicle diagnostic accuracy.

These limitations should be considered when interpreting and presenting research findings.

## 🗂️ Suggested Repository Structure

```text
vehicle-failure-management/
├── data/
│   └── synthetic/
│       ├── FYP_synthetic_vehicle_fault_dataset_50000_checked.csv
│       ├── FYP_synthetic_dataset_checked_methodology.csv
│       └── FYP_synthetic_dataset_checked_validation_report.json
├── notebooks/
│   └── model_training.ipynb
├── src/
│   ├── data_preparation.py
│   └── train_model.py
├── reports/
├── models/
├── README.md
└── .gitignore
```

*This is a suggested structure. Add the training scripts, notebooks, and saved models as they are developed and verified.*

## 🚀 Getting Started

1. Clone or download this repository.
2. Install the required Python libraries for your chosen workflow.
3. Review the methodology and validation report.
4. Load the synthetic CSV dataset.
5. Prepare the predictor features and selected target.
6. Split the data by independent sequences and begin model development.

## 🔬 Research Integrity

This project distinguishes between synthetic-data validation and real-world vehicle validation. All modelling assumptions, evaluation procedures, and limitations should be documented transparently.

The aim is to investigate the potential of ML, Explainable AI, and 3D visualisation for vehicle failure management—not to claim a production-ready diagnostic system without appropriate real-world evidence.

---

**Project:** From Black-Box Predictions to Visual Intelligence: An Explainable AI and 3D Visualization Framework for Vehicle Failure Management

**Dataset status:** Synthetic dataset generated; documented internal validation checks passed.

**Next stage:** Machine learning model development and evaluation.
