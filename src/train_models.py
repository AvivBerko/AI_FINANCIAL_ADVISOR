"""
Phase 3: Model Training
=======================
Trains and compares multiple algorithms per stage as committed in the mid-term
report (section 4.2.2):

  Stage 1 — Risk Profile Classification (3-class):
      KNN, Logistic Regression, Random Forest (+ GridSearchCV), XGBoost, MLP
      → best model saved as models/risk_profile_classifier.pkl

  Stage 2 — Asset Allocation Regression (multi-output):
      Random Forest, GradientBoosting (via MultiOutput wrapper)
      → best model saved as models/allocation_regressor.pkl

  Stage 3 — Total ETF Count Regression:
      Random Forest (sufficient for cardinality prediction)
      → saved as models/etf_count_regressor.pkl

Run from project root:
    venv/bin/python src/train_models.py
"""

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                               GradientBoostingRegressor)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              confusion_matrix, mean_absolute_error, r2_score)
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

DATA_DIR   = 'data/training'
MODEL_DIR  = 'models'
REPORT_DIR = 'evaluation_reports'
for d in [MODEL_DIR, REPORT_DIR]:
    os.makedirs(d, exist_ok=True)

print("=" * 60)
print("PHASE 3: MODEL TRAINING & COMPARISON")
print("=" * 60)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
X_train = pd.read_csv(f'{DATA_DIR}/X_train.csv')
X_test  = pd.read_csv(f'{DATA_DIR}/X_test.csv')

y1_train = pd.read_csv(f'{DATA_DIR}/y_train_profile.csv')['risk_profile']
y1_test  = pd.read_csv(f'{DATA_DIR}/y_test_profile.csv')['risk_profile']

y2_train = pd.read_csv(f'{DATA_DIR}/y_train_allocation.csv')
y2_test  = pd.read_csv(f'{DATA_DIR}/y_test_allocation.csv')

y3_train = pd.read_csv(f'{DATA_DIR}/y_train_etfs.csv')['total_etfs']
y3_test  = pd.read_csv(f'{DATA_DIR}/y_test_etfs.csv')['total_etfs']

print(f"\nLoaded: {len(X_train)} train / {len(X_test)} test, {X_train.shape[1]} features")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def eval_classifier(model, X_tr, y_tr, X_te, y_te):
    """Fit model and return dict of key metrics."""
    model.fit(X_tr, y_tr)
    train_acc = accuracy_score(y_tr, model.predict(X_tr))
    test_acc  = accuracy_score(y_te, model.predict(X_te))
    _, _, macro_f1, _ = precision_recall_fscore_support(
        y_te, model.predict(X_te), average='macro', zero_division=0)
    return {'train_acc': train_acc, 'test_acc': test_acc, 'macro_f1': macro_f1,
            'model': model}


# ===========================================================================
# STAGE 1 — RISK PROFILE CLASSIFICATION
# ===========================================================================
print("\n" + "=" * 60)
print("STAGE 1: RISK PROFILE CLASSIFICATION")
print("=" * 60)

# Label-encode for XGBoost
label_map = {'Aggressive': 0, 'Conservative': 1, 'Moderate': 2}
inv_label_map = {v: k for k, v in label_map.items()}
y1_train_enc = y1_train.map(label_map)
y1_test_enc  = y1_test.map(label_map)

# --- 1a. KNN (benchmark) ---
print("\n[1/5] KNN — fitting with k=5,11,21 via cross-val...")
knn_params = {'n_neighbors': [5, 11, 21], 'weights': ['uniform', 'distance']}
knn_gs = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5,
                      scoring='accuracy', n_jobs=-1)
knn_gs.fit(X_train, y1_train)
knn_best = knn_gs.best_estimator_
knn_res = eval_classifier(knn_best, X_train, y1_train, X_test, y1_test)
print(f"     Best params: {knn_gs.best_params_}")
print(f"     Test Accuracy: {knn_res['test_acc']*100:.2f}%  Macro F1: {knn_res['macro_f1']:.3f}")

# --- 1b. Logistic Regression ---
print("\n[2/5] Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
lr_res = eval_classifier(lr, X_train, y1_train, X_test, y1_test)
print(f"     Test Accuracy: {lr_res['test_acc']*100:.2f}%  Macro F1: {lr_res['macro_f1']:.3f}")

# --- 1c. Random Forest + GridSearchCV ---
print("\n[3/5] Random Forest + GridSearchCV...")
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth':    [8, 12, None],
    'min_samples_split': [5, 10],
}
rf_gs = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0
)
rf_gs.fit(X_train, y1_train)
rf_best = rf_gs.best_estimator_
rf_res = eval_classifier(rf_best, X_train, y1_train, X_test, y1_test)
print(f"     Best params: {rf_gs.best_params_}")
print(f"     Test Accuracy: {rf_res['test_acc']*100:.2f}%  Macro F1: {rf_res['macro_f1']:.3f}")

# --- 1d. XGBoost ---
print("\n[4/5] XGBoost...")
xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                    random_state=42, n_jobs=-1,
                    eval_metric='mlogloss', verbosity=0)
xgb.fit(X_train, y1_train_enc)
xgb_train_acc = accuracy_score(y1_train_enc, xgb.predict(X_train))
xgb_test_preds_enc = xgb.predict(X_test)
xgb_test_preds = pd.Series(xgb_test_preds_enc).map(inv_label_map)
xgb_test_acc = accuracy_score(y1_test, xgb_test_preds)
_, _, xgb_f1, _ = precision_recall_fscore_support(
    y1_test, xgb_test_preds, average='macro', zero_division=0)
xgb_res = {'train_acc': xgb_train_acc, 'test_acc': xgb_test_acc,
           'macro_f1': xgb_f1, 'model': xgb}
print(f"     Test Accuracy: {xgb_test_acc*100:.2f}%  Macro F1: {xgb_f1:.3f}")

# --- 1e. MLP ---
# Note: early_stopping on string labels hits a sklearn/Python 3.14 bug,
# so we use integer-encoded labels for MLP only.
print("\n[5/5] MLP Neural Network...")
mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu',
                    max_iter=500, random_state=42)
mlp.fit(X_train, y1_train_enc)
mlp_train_acc = accuracy_score(y1_train_enc, mlp.predict(X_train))
mlp_test_preds_enc = mlp.predict(X_test)
mlp_test_preds = pd.Series(mlp_test_preds_enc).map(inv_label_map)
mlp_test_acc = accuracy_score(y1_test, mlp_test_preds)
_, _, mlp_f1, _ = precision_recall_fscore_support(
    y1_test, mlp_test_preds, average='macro', zero_division=0)
mlp_res = {'train_acc': mlp_train_acc, 'test_acc': mlp_test_acc,
           'macro_f1': mlp_f1, 'model': mlp}
print(f"     Test Accuracy: {mlp_res['test_acc']*100:.2f}%  Macro F1: {mlp_res['macro_f1']:.3f}")

# --- Comparison table ---
stage1_results = {
    'KNN':                 knn_res,
    'Logistic Regression': lr_res,
    'Random Forest':       rf_res,
    'XGBoost':             xgb_res,
    'MLP':                 mlp_res,
}

print("\n--- Stage 1 Comparison (Test Set) ---")
print(f"{'Algorithm':<22} {'Train Acc':>10} {'Test Acc':>10} {'Macro F1':>10}")
print("-" * 55)
for name, r in stage1_results.items():
    print(f"{name:<22} {r['train_acc']*100:>9.2f}% {r['test_acc']*100:>9.2f}% {r['macro_f1']:>10.3f}")

# Select winner by macro F1
best_name_s1 = max(stage1_results, key=lambda k: stage1_results[k]['macro_f1'])
best_model_s1 = stage1_results[best_name_s1]['model']
print(f"\nWinner: {best_name_s1}  (Macro F1 = {stage1_results[best_name_s1]['macro_f1']:.3f})")

# Confusion matrix for winner
if best_name_s1 in ('XGBoost', 'MLP'):
    y1_test_pred_best = pd.Series(best_model_s1.predict(X_test)).map(inv_label_map)
else:
    y1_test_pred_best = best_model_s1.predict(X_test)
classes = sorted(y1_test.unique())
cm = confusion_matrix(y1_test, y1_test_pred_best, labels=classes)
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes, ax=ax)
ax.set_title(f'Confusion Matrix — {best_name_s1} (Best Stage 1 Model)')
ax.set_ylabel('True')
ax.set_xlabel('Predicted')
fig.tight_layout()
fig.savefig(f'{REPORT_DIR}/confusion_matrix_risk_profile.png', dpi=150)
plt.close(fig)

# Feature importance (RF only)
fi_plot_model = rf_best  # Always plot RF feature importance
fi = pd.DataFrame({'feature': X_train.columns,
                   'importance': fi_plot_model.feature_importances_}
                  ).sort_values('importance', ascending=False)
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(fi.head(10)['feature'][::-1], fi.head(10)['importance'][::-1])
ax.set_xlabel('Importance')
ax.set_title('Top 10 Feature Importances — Random Forest')
fig.tight_layout()
fig.savefig(f'{REPORT_DIR}/feature_importance_risk_profile.png', dpi=150)
plt.close(fig)

# Algorithm comparison bar chart
fig, ax = plt.subplots(figsize=(9, 5))
names  = list(stage1_results.keys())
f1s    = [stage1_results[n]['macro_f1'] for n in names]
accs   = [stage1_results[n]['test_acc'] for n in names]
x = np.arange(len(names))
bars1 = ax.bar(x - 0.2, accs,  0.38, label='Test Accuracy', color='steelblue')
bars2 = ax.bar(x + 0.2, f1s,   0.38, label='Macro F1',      color='coral')
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=15, ha='right')
ax.set_ylim(0, 1)
ax.set_ylabel('Score')
ax.set_title('Stage 1 — Algorithm Comparison (Risk Profile Classification)')
ax.legend()
ax.axhline(0.333, ls='--', color='grey', lw=1, label='Random baseline')
fig.tight_layout()
fig.savefig(f'{REPORT_DIR}/stage1_algorithm_comparison.png', dpi=150)
plt.close(fig)
print(f"Plots saved to {REPORT_DIR}/")

# Save winner
joblib.dump(best_model_s1, f'{MODEL_DIR}/risk_profile_classifier.pkl')
# Always save RF too (needed for feature importance in inference)
joblib.dump(rf_best, f'{MODEL_DIR}/risk_profile_rf.pkl')
# Save label map for XGBoost decoding
with open(f'{MODEL_DIR}/label_map.json', 'w') as f:
    json.dump({'label_map': label_map, 'inv_label_map': inv_label_map,
               'winner': best_name_s1}, f, indent=2)
print(f"Winner model saved: {MODEL_DIR}/risk_profile_classifier.pkl")


# ===========================================================================
# STAGE 2 — ALLOCATION REGRESSION
# ===========================================================================
print("\n" + "=" * 60)
print("STAGE 2: ALLOCATION REGRESSION (Multi-Output)")
print("=" * 60)

def eval_regressor_mo(model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    mae_eq  = mean_absolute_error(y_te['equity_pct'], pred[:, 0])
    mae_bd  = mean_absolute_error(y_te['bond_pct'],   pred[:, 1])
    r2_eq   = r2_score(y_te['equity_pct'], pred[:, 0])
    r2_bd   = r2_score(y_te['bond_pct'],   pred[:, 1])
    avg_mae = (mae_eq + mae_bd) / 2
    return {'mae_eq': mae_eq, 'mae_bd': mae_bd, 'r2_eq': r2_eq,
            'r2_bd': r2_bd, 'avg_mae': avg_mae, 'model': model,
            'pred': pred}

print("\n[1/2] Multi-Output Random Forest...")
mo_rf = MultiOutputRegressor(
    RandomForestRegressor(n_estimators=100, max_depth=15,
                          min_samples_split=10, min_samples_leaf=5,
                          random_state=42, n_jobs=-1))
mo_rf_res = eval_regressor_mo(mo_rf, X_train, y2_train, X_test, y2_test)
print(f"     Equity MAE: {mo_rf_res['mae_eq']*100:.2f}%  Bond MAE: {mo_rf_res['mae_bd']*100:.2f}%  Avg MAE: {mo_rf_res['avg_mae']*100:.2f}%")

print("\n[2/2] Multi-Output Gradient Boosting...")
mo_gb = MultiOutputRegressor(
    GradientBoostingRegressor(n_estimators=100, max_depth=5,
                              learning_rate=0.1, random_state=42))
mo_gb_res = eval_regressor_mo(mo_gb, X_train, y2_train, X_test, y2_test)
print(f"     Equity MAE: {mo_gb_res['mae_eq']*100:.2f}%  Bond MAE: {mo_gb_res['mae_bd']*100:.2f}%  Avg MAE: {mo_gb_res['avg_mae']*100:.2f}%")

stage2_results = {'Random Forest': mo_rf_res, 'Gradient Boosting': mo_gb_res}

print("\n--- Stage 2 Comparison ---")
print(f"{'Algorithm':<22} {'Equity MAE':>12} {'Bond MAE':>10} {'Avg MAE':>10}")
print("-" * 57)
for name, r in stage2_results.items():
    print(f"{name:<22} {r['mae_eq']*100:>11.2f}% {r['mae_bd']*100:>9.2f}% {r['avg_mae']*100:>9.2f}%")

best_name_s2 = min(stage2_results, key=lambda k: stage2_results[k]['avg_mae'])
best_model_s2 = stage2_results[best_name_s2]['model']
best_pred_s2  = stage2_results[best_name_s2]['pred']
print(f"\nWinner: {best_name_s2}  (Avg MAE = {stage2_results[best_name_s2]['avg_mae']*100:.2f}%)")

# Scatter plots
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for i, col in enumerate(['equity_pct', 'bond_pct']):
    axes[i].scatter(y2_test[col], best_pred_s2[:, i], alpha=0.4, s=8)
    mn, mx = y2_test[col].min(), y2_test[col].max()
    axes[i].plot([mn, mx], [mn, mx], 'r--', lw=1.5, label='Perfect')
    axes[i].set_xlabel('Actual')
    axes[i].set_ylabel('Predicted')
    axes[i].set_title(col.replace('_', ' ').title())
    axes[i].legend()
fig.suptitle(f'Allocation Regressor ({best_name_s2}) — Predicted vs Actual')
fig.tight_layout()
fig.savefig(f'{REPORT_DIR}/allocation_predictions.png', dpi=150)
plt.close(fig)

joblib.dump(best_model_s2, f'{MODEL_DIR}/allocation_regressor.pkl')
print(f"Winner model saved: {MODEL_DIR}/allocation_regressor.pkl")


# ===========================================================================
# STAGE 3 — ETF COUNT REGRESSION
# ===========================================================================
print("\n" + "=" * 60)
print("STAGE 3: ETF COUNT REGRESSION")
print("=" * 60)

etf_regressor = RandomForestRegressor(
    n_estimators=100, max_depth=10,
    min_samples_split=10, min_samples_leaf=5,
    random_state=42, n_jobs=-1)
etf_regressor.fit(X_train, y3_train)

y3_pred = etf_regressor.predict(X_test)
y3_pred_rounded = np.clip(np.round(y3_pred), 7, 13).astype(int)

test_mae     = mean_absolute_error(y3_test, y3_pred)
test_mae_rnd = mean_absolute_error(y3_test, y3_pred_rounded)
test_r2      = r2_score(y3_test, y3_pred)
test_exact   = (y3_test == y3_pred_rounded).mean()

print(f"\n  MAE (raw):     {test_mae:.3f} ETFs")
print(f"  MAE (rounded): {test_mae_rnd:.3f} ETFs")
print(f"  R²:            {test_r2:.3f}")
print(f"  Exact match:   {test_exact*100:.2f}%")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].hist(y3_test, bins=np.arange(6.5, 14.5, 1), alpha=0.7, edgecolor='black')
axes[0].set_title('Actual Distribution (Test)')
axes[1].hist(y3_pred_rounded, bins=np.arange(6.5, 14.5, 1), alpha=0.7,
             color='orange', edgecolor='black')
axes[1].set_title('Predicted Distribution (Test)')
for ax in axes:
    ax.set_xlabel('Total ETFs')
    ax.set_ylabel('Count')
fig.tight_layout()
fig.savefig(f'{REPORT_DIR}/etf_count_distribution.png', dpi=150)
plt.close(fig)

joblib.dump(etf_regressor, f'{MODEL_DIR}/etf_count_regressor.pkl')
print(f"Model saved: {MODEL_DIR}/etf_count_regressor.pkl")


# ===========================================================================
# Save all metrics
# ===========================================================================
metrics = {
    'stage_1_risk_profile': {
        'winner': best_name_s1,
        'all_models': {
            name: {
                'train_accuracy': round(r['train_acc'], 4),
                'test_accuracy':  round(r['test_acc'],  4),
                'macro_f1':       round(r['macro_f1'],  4),
            }
            for name, r in stage1_results.items()
        }
    },
    'stage_2_allocation': {
        'winner': best_name_s2,
        'all_models': {
            name: {
                'equity_test_mae': round(r['mae_eq'],  4),
                'bond_test_mae':   round(r['mae_bd'],  4),
                'avg_mae':         round(r['avg_mae'], 4),
                'equity_r2':       round(r['r2_eq'],   4),
                'bond_r2':         round(r['r2_bd'],   4),
            }
            for name, r in stage2_results.items()
        }
    },
    'stage_3_etf_count': {
        'test_mae':             round(float(test_mae),     4),
        'test_mae_rounded':     round(float(test_mae_rnd), 4),
        'test_r2':              round(float(test_r2),      4),
        'exact_match_accuracy': round(float(test_exact),   4),
    },
}

with open(f'{REPORT_DIR}/model_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"\nAll metrics saved to {REPORT_DIR}/model_metrics.json")

print("\n" + "=" * 60)
print("PHASE 3 COMPLETE")
print(f"  Stage 1 winner: {best_name_s1}")
print(f"  Stage 2 winner: {best_name_s2}")
print(f"  Stage 3: Random Forest  (MAE={test_mae_rnd:.3f})")
print("=" * 60)
