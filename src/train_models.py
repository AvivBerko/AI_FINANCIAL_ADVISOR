"""
Phase 3: Production Model Training (V3 Winners)
================================================
Trains and saves the three production models with the hyperparameters
selected by the V3 evaluation (see model_winners.md).
Algorithm comparison / hyperparameter search live in the
`notebooks/phase3*_*_experiments*.ipynb` notebooks — this script only
fits the chosen winners so the production pipeline is reproducible.

Winners (from V3 evaluation, data generated with LABEL_TEMP=0.14, ETF_COUNT_NOISE_P=0.25):
  Stage 1 — Risk Profile (3-class):
        Best:        TabPFN          (Test Acc 0.8000, Macro F1 0.8004)
        Fallback:    Random Forest   (Test Acc 0.7990, Macro F1 0.7976,
                     params: max_depth=5, min_samples_split=2, n_estimators=200)
        TabPFN requires a one-time license/API token. Set TABPFN_TOKEN in
        your environment (see https://ux.priorlabs.ai/account) to enable it;
        otherwise the script falls back to the Random Forest winner.
  Stage 2 — Allocation (multi-output regression): Random Forest
             (n_estimators=1000, max_depth=10, min_samples_split=20; MAE 0.0760)
  Stage 3 — ETF Count (regression): Random Forest
             (n_estimators=1000, max_depth=5, min_samples_split=2; MAE 0.5729)

Run from project root:
    venv/bin/python src/train_models.py
"""
from __future__ import annotations

import json
import os
import warnings

from dotenv import load_dotenv
load_dotenv()

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor)
from sklearn.metrics import (accuracy_score, mean_absolute_error,
                              precision_recall_fscore_support, r2_score)
from sklearn.multioutput import MultiOutputRegressor

warnings.filterwarnings('ignore')


def _build_stage1_classifier():
    """Return (name, model) for Stage 1.

    Prefers TabPFN when `TABPFN_TOKEN` is set (the V2 winner); otherwise
    falls back to the Random Forest configuration that was the
    second-best entry in the V2 evaluation.
    """
    if os.getenv('TABPFN_TOKEN'):
        try:
            from tabpfn import TabPFNClassifier
            return 'TabPFN', TabPFNClassifier(random_state=42)
        except Exception as e:
            print(f"  ⚠️  TabPFN unavailable ({e}); falling back to Random Forest.")
    else:
        print("  ⚠️  TABPFN_TOKEN not set — using Random Forest fallback.")
        print("      To enable TabPFN: register at https://ux.priorlabs.ai,")
        print("      accept the license, then export TABPFN_TOKEN=<api-key>.")
    return (
        'Random Forest',
        RandomForestClassifier(
            max_depth=5, min_samples_split=2, n_estimators=200,
            random_state=42, n_jobs=-1,
        ),
    )

DATA_DIR   = 'data/training'
MODEL_DIR  = 'models'
REPORT_DIR = 'evaluation_reports'
for d in (MODEL_DIR, REPORT_DIR):
    os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Load preprocessed data (produced by prepare_data.py)
# ---------------------------------------------------------------------------
X_train = pd.read_csv(f'{DATA_DIR}/X_train.csv')
X_test  = pd.read_csv(f'{DATA_DIR}/X_test.csv')

y1_train = pd.read_csv(f'{DATA_DIR}/y_train_profile.csv')['risk_profile']
y1_test  = pd.read_csv(f'{DATA_DIR}/y_test_profile.csv')['risk_profile']

y2_train = pd.read_csv(f'{DATA_DIR}/y_train_allocation.csv')
y2_test  = pd.read_csv(f'{DATA_DIR}/y_test_allocation.csv')

y3_train = pd.read_csv(f'{DATA_DIR}/y_train_etfs.csv')['total_etfs']
y3_test  = pd.read_csv(f'{DATA_DIR}/y_test_etfs.csv')['total_etfs']

print(f"Loaded: {len(X_train)} train / {len(X_test)} test, {X_train.shape[1]} features")


# ===========================================================================
# STAGE 1 — Risk Profile Classification
# ===========================================================================
print("\n" + "=" * 60)
print("STAGE 1: Risk Profile Classification")
print("=" * 60)

stage1_name, stage1 = _build_stage1_classifier()
print(f"  Algorithm: {stage1_name}")

# Label encoding: TabPFN uses integer-encoded labels (matching the
# XGBoost/MLP convention), Random Forest stays on string labels.
# Inference handles both via label_info['winner'].
label_map = {'Aggressive': 0, 'Conservative': 1, 'Moderate': 2}
inv_label_map = {v: k for k, v in label_map.items()}

if stage1_name == 'TabPFN':
    y_train_fit = y1_train.map(label_map).astype(int).values
    y_test_eval = y1_test.map(label_map).astype(int).values
    stage1.fit(X_train.values, y_train_fit)
    s1_train_pred = stage1.predict(X_train.values)
    s1_test_pred  = stage1.predict(X_test.values)
else:
    y_train_fit = y1_train.values
    y_test_eval = y1_test.values
    stage1.fit(X_train, y1_train)
    s1_train_pred = stage1.predict(X_train)
    s1_test_pred  = stage1.predict(X_test)

s1_train_acc = accuracy_score(y_train_fit, s1_train_pred)
s1_test_acc  = accuracy_score(y_test_eval, s1_test_pred)
_, _, s1_f1, _ = precision_recall_fscore_support(
    y_test_eval, s1_test_pred, average='macro', zero_division=0)
print(f"  Train Accuracy: {s1_train_acc*100:.2f}%")
print(f"  Test  Accuracy: {s1_test_acc*100:.2f}%")
print(f"  Macro F1:       {s1_f1:.4f}")

joblib.dump(stage1, f'{MODEL_DIR}/risk_profile_classifier.pkl')
with open(f'{MODEL_DIR}/label_map.json', 'w') as f:
    json.dump(
        {'label_map':     {k: v for k, v in label_map.items()},
         'inv_label_map': {str(k): v for k, v in inv_label_map.items()},
         'winner':        stage1_name},
        f, indent=2,
    )
print(f"  Saved → {MODEL_DIR}/risk_profile_classifier.pkl")


# ===========================================================================
# STAGE 2 — Allocation Regression (Random Forest, V3 hyperparams)
# ===========================================================================
print("\n" + "=" * 60)
print("STAGE 2: Allocation Regression (Random Forest)")
print("=" * 60)
print("  Hyperparameters: n_estimators=1000, max_depth=10, min_samples_split=20")

stage2 = MultiOutputRegressor(
    RandomForestRegressor(
        n_estimators=1000,
        max_depth=10,
        min_samples_split=20,
        random_state=42,
        n_jobs=-1,
    )
)
stage2.fit(X_train, y2_train)
s2_pred = stage2.predict(X_test)
s2_mae_eq = mean_absolute_error(y2_test['equity_pct'], s2_pred[:, 0])
s2_mae_bd = mean_absolute_error(y2_test['bond_pct'],   s2_pred[:, 1])
s2_r2_eq  = r2_score(y2_test['equity_pct'], s2_pred[:, 0])
s2_r2_bd  = r2_score(y2_test['bond_pct'],   s2_pred[:, 1])
s2_avg_mae = (s2_mae_eq + s2_mae_bd) / 2
print(f"  Equity MAE: {s2_mae_eq:.4f}  Bond MAE: {s2_mae_bd:.4f}  Avg MAE: {s2_avg_mae:.4f}")
print(f"  Equity R²:  {s2_r2_eq:.4f}  Bond R²:  {s2_r2_bd:.4f}")

joblib.dump(stage2, f'{MODEL_DIR}/allocation_regressor.pkl')
print(f"  Saved → {MODEL_DIR}/allocation_regressor.pkl")


# ===========================================================================
# STAGE 3 — ETF Count Regression (Random Forest, V3 hyperparams)
# ===========================================================================
print("\n" + "=" * 60)
print("STAGE 3: ETF Count Regression (Random Forest)")
print("=" * 60)
print("  Hyperparameters: n_estimators=1000, max_depth=5, min_samples_split=2")

stage3 = RandomForestRegressor(
    n_estimators=1000,
    max_depth=5,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1,
)
stage3.fit(X_train, y3_train)
s3_pred = stage3.predict(X_test)
s3_pred_rnd = np.clip(np.round(s3_pred), 7, 13).astype(int)
s3_mae      = mean_absolute_error(y3_test, s3_pred)
s3_mae_rnd  = mean_absolute_error(y3_test, s3_pred_rnd)
s3_r2       = r2_score(y3_test, s3_pred)
s3_exact    = (y3_test == s3_pred_rnd).mean()
print(f"  MAE (raw):     {s3_mae:.4f}")
print(f"  MAE (rounded): {s3_mae_rnd:.4f}")
print(f"  R²:            {s3_r2:.4f}")
print(f"  Exact match:   {s3_exact*100:.2f}%")

joblib.dump(stage3, f'{MODEL_DIR}/etf_count_regressor.pkl')
print(f"  Saved → {MODEL_DIR}/etf_count_regressor.pkl")


# ===========================================================================
# Write metrics summary
# ===========================================================================
metrics = {
    'stage_1_risk_profile': {
        'winner':         stage1_name,
        'train_accuracy': round(float(s1_train_acc), 4),
        'test_accuracy':  round(float(s1_test_acc),  4),
        'macro_f1':       round(float(s1_f1),        4),
    },
    'stage_2_allocation': {
        'winner':          'Random Forest',
        'hyperparameters': {'n_estimators': 1000, 'max_depth': 10, 'min_samples_split': 20},
        'equity_mae':      round(float(s2_mae_eq),   4),
        'bond_mae':        round(float(s2_mae_bd),   4),
        'avg_mae':         round(float(s2_avg_mae),  4),
        'equity_r2':       round(float(s2_r2_eq),    4),
        'bond_r2':         round(float(s2_r2_bd),    4),
    },
    'stage_3_etf_count': {
        'winner':              'Random Forest',
        'hyperparameters':     {'n_estimators': 1000, 'max_depth': 5, 'min_samples_split': 2},
        'mae':                 round(float(s3_mae),     4),
        'mae_rounded':         round(float(s3_mae_rnd), 4),
        'r2':                  round(float(s3_r2),      4),
        'exact_match':         round(float(s3_exact),   4),
    },
}

with open(f'{REPORT_DIR}/model_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("\n" + "=" * 60)
print("Training complete.  All artifacts written to ./models/")
print(f"Metrics → {REPORT_DIR}/model_metrics.json")
print("=" * 60)
