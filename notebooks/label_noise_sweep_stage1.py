"""
Stage 1 (Risk Profile) — label-noise sensitivity / robustness analysis.

Motivation (mentor feedback): choosing the label-noise level *to hit a target
accuracy* is circular ("cooking the results"). The principled alternative is to
treat the noise level as the INDEPENDENT variable and measure how learnability
depends on it. This script sweeps the label-generation noise across its full
range and reports accuracy as a FUNCTION of injected noise.

Design:
  - The label is generated from get_profile_probabilities() (the domain model)
    via temperature-sharpened sampling: label ~ sample(p ** (1/T)).
  - T -> 0  : argmax (deterministic, no injected noise)  -> leakage regime
    T -> 1  : honest sampling from the domain probabilities -> Bayes-ceiling regime
  - We report a model-independent noise measure: FLIP RATE = fraction of labels
    that differ from the argmax label. That is the actual amount of label noise
    injected, independent of any model's score.
  - Model (RandomForest) and train/test split are HELD FIXED across all points,
    so the only thing changing is the label noise. The split is stratified on the
    stable argmax label so it is identical for every T.

Output: a table + a plot (accuracy & train-test gap vs injected noise) saved to
evaluation_reports/.

    python notebooks/label_noise_sweep_stage1.py
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.allocations import get_profile_probabilities  # noqa: E402

CLASSES = ['Aggressive', 'Moderate', 'Conservative']
OUT_DIR = os.path.join(ROOT, 'evaluation_reports')

# --- 1. Load raw investors and align column names to the domain model ---------
inv_file = sorted(glob.glob(os.path.join(ROOT, 'data', 'processed',
                                          'synthetic_investor_data_*.csv')))[-1]
inv = pd.read_csv(inv_file).rename(columns={
    'Age': 'age', 'InvestmentHorizon': 'horizon', 'RiskTolerance': 'risk_tolerance',
    'Income': 'income', 'ExperienceYears': 'experience',
})
print(f'Investors: {len(inv)} from {os.path.basename(inv_file)}')

# --- 2. Compute the domain probabilities ONCE (independent of noise level) -----
prob_dicts = inv.apply(get_profile_probabilities, axis=1)
P = np.array([[d[c] for c in CLASSES] for d in prob_dicts])   # shape (N, 3)
argmax_label = np.array(CLASSES)[P.argmax(axis=1)]            # stable, T-independent

# --- 3. Build features exactly as prepare_data.py does -------------------------
feature_columns = ['age', 'Gender', 'Education', 'MaritalStatus', 'HouseholdSize',
                   'income', 'InvestmentGoal', 'horizon', 'InvestmentCapital',
                   'risk_tolerance', 'FinancialInvolvement', 'experience']
categorical = ['Gender', 'Education', 'MaritalStatus', 'InvestmentGoal',
               'risk_tolerance', 'FinancialInvolvement']
numerical = ['age', 'HouseholdSize', 'income', 'horizon', 'InvestmentCapital', 'experience']
X = inv[feature_columns].copy()

# --- 4. ONE fixed split (stratified on the stable argmax label) ----------------
idx = np.arange(len(inv))
tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=42,
                                  stratify=argmax_label)
pre = ColumnTransformer([
    ('num', StandardScaler(), numerical),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical),
])
X_tr = pre.fit_transform(X.iloc[tr_idx])   # fit on train only (no leakage)
X_te = pre.transform(X.iloc[te_idx])


def sample_labels(P, T, seed=42):
    """Temperature-sharpened sampling from the domain probabilities."""
    rng = np.random.default_rng(seed)
    sharp = P ** (1.0 / T)
    sharp = sharp / sharp.sum(axis=1, keepdims=True)
    picks = np.array([rng.choice(3, p=sharp[i]) for i in range(len(P))])
    return np.array(CLASSES)[picks]


# --- 5. Sweep the noise level --------------------------------------------------
TEMPS = [0.01, 0.05, 0.10, 0.14, 0.20, 0.30, 0.50, 0.70, 1.00]
rows = []
print(f"\n{'T':>5} {'flip%':>6} {'train':>7} {'test':>7} {'gap':>7} {'F1':>7}")
for T in TEMPS:
    y = sample_labels(P, T)
    flip_rate = float(np.mean(y != argmax_label))    # injected noise (model-free)
    y_tr, y_te = y[tr_idx], y[te_idx]

    clf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                 random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    train_acc = accuracy_score(y_tr, clf.predict(X_tr))
    test_pred = clf.predict(X_te)
    test_acc = accuracy_score(y_te, test_pred)
    _, _, f1, _ = precision_recall_fscore_support(y_te, test_pred,
                                                  average='macro', zero_division=0)
    rows.append({'LABEL_TEMP': T, 'flip_rate': flip_rate, 'train_acc': train_acc,
                 'test_acc': test_acc, 'gap': train_acc - test_acc, 'macro_f1': f1})
    print(f"{T:>5.2f} {flip_rate*100:>5.1f}% {train_acc:>7.3f} {test_acc:>7.3f} "
          f"{train_acc-test_acc:>7.3f} {f1:>7.3f}")

df = pd.DataFrame(rows)
csv_path = os.path.join(OUT_DIR, 'stage1_label_noise_sweep.csv')
df.to_csv(csv_path, index=False)
print(f"\nSaved table -> {csv_path}")

# --- 6. Plot accuracy as a function of INJECTED NOISE (flip rate) --------------
fig, ax = plt.subplots(figsize=(9, 5.5))
x = df['flip_rate'] * 100
ax.plot(x, df['train_acc'], 'o-', color='coral', label='Train accuracy')
ax.plot(x, df['test_acc'], 'o-', color='steelblue', label='Test accuracy')
ax.axhline(1/3, ls='--', color='grey', lw=1, label='Random baseline (33%)')
ax.set_xlabel('Injected label noise — % of labels flipped from argmax')
ax.set_ylabel('Accuracy')
ax.set_title('Stage 1: Risk-Profile learnability vs injected label noise\n'
             '(model & split held fixed; only label noise varies)')
ax.set_ylim(0.25, 1.02)
ax.legend(loc='upper right')
for _, r in df.iterrows():
    ax.annotate(f"T={r['LABEL_TEMP']:g}", (r['flip_rate']*100, r['test_acc']),
                textcoords='offset points', xytext=(0, -14), fontsize=7,
                ha='center', color='steelblue')
plt.tight_layout()
png_path = os.path.join(OUT_DIR, 'stage1_label_noise_sweep.png')
plt.savefig(png_path, dpi=130)
print(f"Saved plot  -> {png_path}")
