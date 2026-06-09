"""
Standalone TabPFN evaluation for Stage 1 (Risk Profile classification).

Kept OUT of phase3A_classification_experiments.ipynb on purpose: TabPFN predicts
with a transformer forward-pass over the full training context, so scoring the
4000 train + 1000 test rows is slow and bogs down the interactive notebook.
Run this separately to get TabPFN's row for the model-comparison table.

    python notebooks/tabpfn_stage1_eval.py

Reads the same prepared splits as the notebook (data/training/) and the
TABPFN_TOKEN from .env, mirroring src/train_models.py exactly.
"""
import os
import time
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix)

# Resolve repo root regardless of where the script is launched from
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, '.env'))

DATA_DIR = os.path.join(ROOT, 'data', 'training')
LABEL_MAP = {'Aggressive': 0, 'Conservative': 1, 'Moderate': 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
CLASSES = ['Aggressive', 'Moderate', 'Conservative']

X_train = pd.read_csv(f'{DATA_DIR}/X_train.csv')
X_test = pd.read_csv(f'{DATA_DIR}/X_test.csv')
y_train = pd.read_csv(f'{DATA_DIR}/y_train_profile.csv')['risk_profile']
y_test = pd.read_csv(f'{DATA_DIR}/y_test_profile.csv')['risk_profile']

# TabPFN wants integer labels (same convention as XGBoost/MLP in the notebook)
y_train_enc = y_train.map(LABEL_MAP).astype(int).values
y_test_enc = y_test.map(LABEL_MAP).astype(int).values

print(f'TABPFN_TOKEN loaded: {bool(os.getenv("TABPFN_TOKEN"))}')
print(f'Train: {len(X_train)} rows | Test: {len(X_test)} rows | Features: {X_train.shape[1]}')

from tabpfn import TabPFNClassifier  # noqa: E402

t0 = time.time()
clf = TabPFNClassifier(random_state=42)
clf.fit(X_train.values, y_train_enc)
train_pred = pd.Series(clf.predict(X_train.values)).map(INV_LABEL_MAP).values
test_pred = pd.Series(clf.predict(X_test.values)).map(INV_LABEL_MAP).values
elapsed = time.time() - t0

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)
prec, rec, f1, _ = precision_recall_fscore_support(
    y_test, test_pred, average='macro', zero_division=0)
cm = confusion_matrix(y_test, test_pred, labels=CLASSES)

print('\n' + '=' * 60)
print('TabPFN — Stage 1 Risk Profile')
print('=' * 60)
print(f'  Best Params     : (pretrained — no grid search)')
print(f'  Train Accuracy  : {train_acc:.4f}')
print(f'  Test Accuracy   : {test_acc:.4f}')
print(f'  Macro Precision : {prec:.4f}')
print(f'  Macro Recall    : {rec:.4f}')
print(f'  Macro F1        : {f1:.4f}')
print(f'  Train-Test gap  : {train_acc - test_acc:.4f}')
print(f'  Wall time       : {elapsed:.1f}s')
print('\n  Confusion matrix (rows=true, cols=pred):')
print('  ' + pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_string().replace('\n', '\n  '))

# One-line row matching the notebook comparison columns
print('\nComparison-table row:')
print(f'TabPFN | train={train_acc:.4f} test={test_acc:.4f} '
      f'P={prec:.4f} R={rec:.4f} F1={f1:.4f}')
