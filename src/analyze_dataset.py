from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'FYP_synthetic_vehicle_fault_dataset_50000_checked.csv'
OUT = ROOT / 'reports' / 'dataset_analysis'
FEATURES = ['engine_coolant_temperature','oil_pressure_kpa','engine_rpm','vehicle_speed','throttle_position','engine_load']


def main():
    if not DATA.exists():
        raise FileNotFoundError(f'Dataset not found: {DATA}')
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    required = FEATURES + ['sequence_id','condition','fault_type']
    missing = sorted(set(required)-set(df.columns))
    if missing: raise ValueError(f'Missing required columns: {missing}')
    summary = {
        'shape': list(df.shape),
        'columns': df.columns.tolist(),
        'missing_values_by_column': df.isna().sum().to_dict(),
        'exact_duplicate_rows': int(df.duplicated().sum()),
        'unique_sequences': int(df.sequence_id.nunique()),
        'rows_per_sequence_min': int(df.groupby('sequence_id').size().min()),
        'rows_per_sequence_max': int(df.groupby('sequence_id').size().max()),
        'condition_counts': df.condition.value_counts().to_dict(),
        'fault_type_counts': df.fault_type.value_counts().to_dict(),
        'numeric_describe': df[FEATURES].describe().to_dict(),
        'note': 'Synthetic research data; descriptive checks do not establish real-world validity.'
    }
    (OUT/'dataset_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    df[FEATURES].describe().to_csv(OUT/'sensor_descriptive_statistics.csv')
    for col in ['condition','fault_type']:
        counts = df[col].value_counts().rename_axis('class').reset_index(name='records')
        counts.to_csv(OUT/f'{col}_class_counts.csv', index=False)
        fig, ax = plt.subplots(figsize=(7,5))
        ax.bar(counts['class'].astype(str), counts['records'])
        ax.set_title(f'Dataset class distribution: {col}')
        ax.tick_params(axis='x', rotation=20)
        fig.tight_layout(); fig.savefig(OUT/f'{col}_class_distribution.png', dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9,7))
    corr = df[FEATURES].corr(numeric_only=True)
    image = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap='coolwarm')
    ax.set_xticks(range(len(FEATURES)), FEATURES, rotation=45, ha='right')
    ax.set_yticks(range(len(FEATURES)), FEATURES)
    for i in range(len(FEATURES)):
        for j in range(len(FEATURES)):
            ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center', fontsize=8)
    fig.colorbar(image, ax=ax, label='Correlation')
    ax.set_title('Sensor feature correlation (descriptive only)')
    fig.tight_layout(); fig.savefig(OUT/'sensor_correlation_heatmap.png', dpi=180); plt.close(fig)
    for col in FEATURES:
        fig, ax = plt.subplots(figsize=(7,4))
        ax.hist(df[col], bins=30, edgecolor='black')
        ax.set_title(f'Distribution: {col}')
        fig.tight_layout(); fig.savefig(OUT/f'{col}_distribution.png', dpi=180); plt.close(fig)
    print('EDA complete. Results saved to:', OUT)
    print('Rows/columns:', df.shape)
    print('Sequences:', df.sequence_id.nunique())
    print('Missing values:', int(df.isna().sum().sum()))
    print('Duplicates:', int(df.duplicated().sum()))
    print('Condition counts:', df.condition.value_counts().to_dict())
    print('Fault counts:', df.fault_type.value_counts().to_dict())

if __name__ == '__main__': main()
