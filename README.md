FYP SYNTHETIC VEHICLE FAULT DATASET

Files
- FYP_synthetic_vehicle_fault_dataset_50000_checked.csv: 50,000 rows.
- FYP_synthetic_dataset_checked_methodology.csv: source, assumptions, and rules.
- FYP_synthetic_dataset_checked_validation_report.json: validation checks and comparisons.

Scope
Two fault categories: engine overheating and low engine oil pressure. Labels: normal, warning, critical.

Use
Use the six sensor/vehicle features as predictors. Do not use sequence_id, timestamp_step, scenario_type, or fault_active as predictors. Split train/test by sequence_id. Choose either fault_type or condition as target.

Important scientific note
This is synthetic research data. It is checked for internal consistency and baseline similarity, but has not been validated against real Toyota Aqua fault logs. Some thresholds are explicit modelling assumptions rather than official Toyota warning/critical limits. Read the methodology CSV and JSON validation report before reporting results.

Validation checks passed: 41/41.
