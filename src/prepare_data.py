"""
Phase 2: Data Preparation
=========================
Loads ETF + investor data, runs the rule-based allocation pipeline to generate
ground-truth labels, engineers features, splits train/test, and saves all
artifacts to data/training/.

Run from project root:
    venv/bin/python src/prepare_data.py
"""

import os
import sys
import glob
import warnings
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.allocations import run_allocations_pipeline

OUTPUT_DIR = 'data/training'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Load raw data
# ---------------------------------------------------------------------------
print("=" * 60)
print("PHASE 2: DATA PREPARATION")
print("=" * 60)

etf_file = 'data/raw/combined_etf_with_morning_star.csv'
if not os.path.exists(etf_file):
    raise FileNotFoundError(f"ETF file not found: {etf_file}. Run pipeline.py first.")

investor_files = sorted(glob.glob('data/processed/synthetic_investor_data_*.csv'))
if not investor_files:
    raise FileNotFoundError("No investor data found in data/processed/. Run pipeline.py first.")
investor_file = investor_files[-1]  # Use the most recent

print(f"\nLoading ETF data from:      {etf_file}")
print(f"Loading investor data from: {investor_file}")

etf_df = pd.read_csv(etf_file)
investors_df = pd.read_csv(investor_file)

print(f"\nLoaded {len(etf_df)} ETFs and {len(investors_df)} investors")


# ---------------------------------------------------------------------------
# 2. Generate ground-truth labels via rule-based allocation pipeline
# ---------------------------------------------------------------------------
print("\nRunning allocation pipeline to generate labels...")
result_df = run_allocations_pipeline(etf_df, investors_df)
print(f"Pipeline complete. Result shape: {result_df.shape}")


# ---------------------------------------------------------------------------
# 3. Extract features (X) — 12 raw columns → 19 after encoding
# ---------------------------------------------------------------------------
feature_columns = [
    'age', 'Gender', 'Education', 'MaritalStatus', 'HouseholdSize',
    'income', 'InvestmentGoal', 'horizon', 'InvestmentCapital',
    'risk_tolerance', 'FinancialInvolvement', 'experience'
]
X = result_df[feature_columns].copy()

categorical_features = ['Gender', 'Education', 'MaritalStatus', 'InvestmentGoal',
                         'risk_tolerance', 'FinancialInvolvement']
numerical_features   = ['age', 'HouseholdSize', 'income', 'horizon',
                         'InvestmentCapital', 'experience']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(),                                          numerical_features),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features),
])

X_processed = preprocessor.fit_transform(X)
cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
all_feature_names = numerical_features + list(cat_feature_names)
X_processed_df = pd.DataFrame(X_processed, columns=all_feature_names)

print(f"\nFeature engineering: {X.shape[1]} raw → {X_processed_df.shape[1]} processed features")


# ---------------------------------------------------------------------------
# 4. Extract targets
# ---------------------------------------------------------------------------
# Target 1 — Risk Profile (Classification)
y1_risk_profile = result_df['risk_profile'].copy()

# Target 2 — Allocation % (Multi-output Regression)
def extract_allocation(alloc):
    if isinstance(alloc, dict):
        return alloc.get('equity', 0.0), alloc.get('bond', 0.0)
    import ast
    try:
        d = ast.literal_eval(str(alloc))
        return d.get('equity', 0.0), d.get('bond', 0.0)
    except Exception:
        return 0.0, 0.0

y2_allocation = pd.DataFrame(
    result_df['allocation'].apply(extract_allocation).tolist(),
    columns=['equity_pct', 'bond_pct']
)

# Target 3 — Total ETF count (Regression)
y3_total_etfs = result_df['total_etfs'].copy()


# ---------------------------------------------------------------------------
# 5. Train / test split  (80 / 20, stratified on risk profile)
# ---------------------------------------------------------------------------
X_train, X_test, y1_train, y1_test, y2_train, y2_test, y3_train, y3_test = train_test_split(
    X_processed_df, y1_risk_profile, y2_allocation, y3_total_etfs,
    test_size=0.2, random_state=42, stratify=y1_risk_profile
)

print(f"\nTrain/Test split (80/20, stratified):")
print(f"  X_train: {X_train.shape}   X_test: {X_test.shape}")


# ---------------------------------------------------------------------------
# 6. Save artifacts
# ---------------------------------------------------------------------------
X_train.to_csv(f'{OUTPUT_DIR}/X_train.csv',              index=False)
X_test.to_csv(f'{OUTPUT_DIR}/X_test.csv',                index=False)
y1_train.to_csv(f'{OUTPUT_DIR}/y_train_profile.csv',     index=False, header=['risk_profile'])
y1_test.to_csv(f'{OUTPUT_DIR}/y_test_profile.csv',       index=False, header=['risk_profile'])
y2_train.to_csv(f'{OUTPUT_DIR}/y_train_allocation.csv',  index=False)
y2_test.to_csv(f'{OUTPUT_DIR}/y_test_allocation.csv',    index=False)
y3_train.to_csv(f'{OUTPUT_DIR}/y_train_etfs.csv',        index=False, header=['total_etfs'])
y3_test.to_csv(f'{OUTPUT_DIR}/y_test_etfs.csv',          index=False, header=['total_etfs'])

joblib.dump(preprocessor, f'{OUTPUT_DIR}/preprocessor.pkl')

print(f"\nSaved all artifacts to {OUTPUT_DIR}/")
print("\nRisk profile distribution (train):")
print(y1_train.value_counts(normalize=True).mul(100).round(1).to_string())

print("\n" + "=" * 60)
print("PHASE 2 COMPLETE — ready for src/train_models.py")
print("=" * 60)
