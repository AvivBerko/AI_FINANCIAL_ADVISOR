"""
Core Recommendation Engine — Inference Pipeline
================================================
Takes a structured investor profile (dict) and returns a complete
portfolio recommendation by running through all 4 stages:

  Stage 1 (ML)          → Risk Profile   : Aggressive / Moderate / Conservative
  Stage 2 (ML)          → Allocation %   : equity_pct, bond_pct, alt_pct
  Stage 3 (ML)          → ETF Count      : how many ETFs per asset class
  Stage 4 (Rule-Based)  → ETF Selection  : specific tickers (conflict-free)

Usage:
    from src.inference import recommend
    result = recommend({
        "age": 30, "Gender": "Male", "Education": "Bachelor",
        "MaritalStatus": "Single", "HouseholdSize": 1,
        "income": 80000, "InvestmentGoal": "Retirement",
        "horizon": 30, "InvestmentCapital": 50000,
        "risk_tolerance": "Medium", "FinancialInvolvement": "Medium",
        "experience": 3
    })

Or run directly:
    venv/bin/python src/inference.py
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.allocations import get_num_etfs, assign_portfolio_numeric_aum, get_category_mapping

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_DIR    = 'models'
TRAINING_DIR = 'data/training'
ETF_FILE     = 'data/raw/combined_etf_with_morning_star.csv'

# Expected input feature columns (must match prepare_data.py)
FEATURE_COLUMNS = [
    'age', 'Gender', 'Education', 'MaritalStatus', 'HouseholdSize',
    'income', 'InvestmentGoal', 'horizon', 'InvestmentCapital',
    'risk_tolerance', 'FinancialInvolvement', 'experience'
]

# Valid values for each categorical field
VALID_VALUES = {
    'Gender':               ['Male', 'Female'],
    'Education':            ['High school', 'Bachelor', 'Master'],
    'MaritalStatus':        ['Single', 'Married', 'Divorced', 'Widowed'],
    'InvestmentGoal':       ['Retirement', 'Home purchase', 'Child education', 'Other'],
    'risk_tolerance':       ['Low', 'Medium', 'High'],
    'FinancialInvolvement': ['Low', 'Medium', 'High'],
}


# ---------------------------------------------------------------------------
# Load artifacts once (module-level, cached after first import)
# ---------------------------------------------------------------------------
def _load_artifacts():
    preprocessor      = joblib.load(f'{TRAINING_DIR}/preprocessor.pkl')
    classifier        = joblib.load(f'{MODEL_DIR}/risk_profile_classifier.pkl')
    alloc_regressor   = joblib.load(f'{MODEL_DIR}/allocation_regressor.pkl')
    count_regressor   = joblib.load(f'{MODEL_DIR}/etf_count_regressor.pkl')

    with open(f'{MODEL_DIR}/label_map.json') as f:
        label_info = json.load(f)

    etf_df = pd.read_csv(ETF_FILE)

    # Prepare ETF lookup by asset class (same logic as allocations.py)
    etf_features = etf_df.rename(columns={
        'Fund Symbol':                    'symbol',
        'Category':                       'category',
        'Assets Under Management (AUM)':  'AUM',
        'custom_star_rating':             'star_rating',
        'Volatility (Annual STD)':        'volatility',
    })
    etf_features['AUM']         = pd.to_numeric(etf_features['AUM'],         errors='coerce').fillna(1e9)
    etf_features['star_rating'] = pd.to_numeric(etf_features['star_rating'],  errors='coerce').fillna(3)
    etf_features['volatility']  = pd.to_numeric(etf_features['volatility'],   errors='coerce').fillna(0.15)
    etf_features['asset_class'] = etf_features['category'].apply(get_category_mapping)

    etf_by_class = {
        cls: etf_features[etf_features['asset_class'] == cls].copy()
        for cls in ['equity', 'bond', 'alternative']
    }

    return preprocessor, classifier, alloc_regressor, count_regressor, label_info, etf_by_class

_ARTIFACTS = None

def _get_artifacts():
    global _ARTIFACTS
    if _ARTIFACTS is None:
        _ARTIFACTS = _load_artifacts()
    return _ARTIFACTS


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def validate_input(user: dict) -> None:
    missing = [f for f in FEATURE_COLUMNS if f not in user]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    for field, valid in VALID_VALUES.items():
        val = user.get(field)
        if val not in valid:
            raise ValueError(
                f"Invalid value '{val}' for '{field}'. Must be one of: {valid}"
            )

    numeric_checks = {
        'age':               (18, 100),
        'HouseholdSize':     (1, 10),
        'income':            (0, 10_000_000),
        'horizon':           (1, 50),
        'InvestmentCapital': (0, 100_000_000),
        'experience':        (0, 80),
    }
    for field, (lo, hi) in numeric_checks.items():
        val = user.get(field)
        if not (lo <= val <= hi):
            raise ValueError(f"'{field}' = {val} is out of range [{lo}, {hi}]")


# ---------------------------------------------------------------------------
# Main recommendation function
# ---------------------------------------------------------------------------
def recommend(user: dict) -> dict:
    """
    Run the full 4-stage pipeline for a single investor profile.

    Parameters
    ----------
    user : dict
        Investor profile with keys matching FEATURE_COLUMNS.

    Returns
    -------
    dict with keys:
        risk_profile   : str   — Aggressive / Moderate / Conservative
        allocation     : dict  — {equity_pct, bond_pct, alt_pct}  (sum = 1.0)
        total_etfs     : int   — total ETFs in portfolio
        etf_counts     : dict  — {equity, bond, alternative} counts
        portfolio      : dict  — {equity: [...], bond: [...], alternative: [...]}
    """
    validate_input(user)

    preprocessor, classifier, alloc_regressor, count_regressor, label_info, etf_by_class = \
        _get_artifacts()

    # -- Build input DataFrame -----------------------------------------------
    X_raw = pd.DataFrame([{col: user[col] for col in FEATURE_COLUMNS}])

    # -- Preprocess (scale + one-hot encode) ---------------------------------
    X_processed = preprocessor.transform(X_raw)

    # -- Stage 1: Risk Profile Classification --------------------------------
    winner = label_info['winner']
    inv_label_map = label_info['inv_label_map']

    raw_pred = classifier.predict(X_processed)

    # XGBoost and MLP were trained on integer-encoded labels
    if winner in ('XGBoost', 'MLP'):
        risk_profile = inv_label_map[str(raw_pred[0])]
    else:
        risk_profile = raw_pred[0]

    # -- Stage 2: Allocation Regression --------------------------------------
    alloc_pred = alloc_regressor.predict(X_processed)[0]
    equity_raw = float(alloc_pred[0])
    bond_raw   = float(alloc_pred[1])

    # Apply risk-profile-aware bounds before normalizing.
    # The regressor has ~15% MAE so we enforce soft guardrails per profile.
    PROFILE_BOUNDS = {
        'Aggressive':   {'equity': (0.55, 0.90), 'bond': (0.05, 0.30)},
        'Moderate':     {'equity': (0.40, 0.70), 'bond': (0.20, 0.45)},
        'Conservative': {'equity': (0.20, 0.45), 'bond': (0.35, 0.65)},
    }
    bounds = PROFILE_BOUNDS.get(risk_profile, {'equity': (0.01, 0.98), 'bond': (0.01, 0.98)})
    equity_raw = max(bounds['equity'][0], min(bounds['equity'][1], equity_raw))
    bond_raw   = max(bounds['bond'][0],   min(bounds['bond'][1],   bond_raw))
    alt_raw    = max(0.01, 1.0 - equity_raw - bond_raw)

    total = equity_raw + bond_raw + alt_raw
    allocation = {
        'equity_pct': round(equity_raw / total, 4),
        'bond_pct':   round(bond_raw   / total, 4),
        'alt_pct':    round(alt_raw    / total, 4),
    }

    # -- Stage 3: ETF Count --------------------------------------------------
    # Derive per-class ETF counts using the same rule as training data generation.
    # Stage 3 ML prediction gives us the total; we use get_num_etfs() for
    # per-class distribution (consistent with how labels were created).
    total_pred = count_regressor.predict(X_processed)[0]
    total_etfs = int(np.clip(round(total_pred), 7, 13))

    num_equity = get_num_etfs('Equity',      risk_profile)
    num_bond   = get_num_etfs('Bond',        risk_profile)
    num_alt    = get_num_etfs('Alternative', risk_profile)

    etf_counts = {
        'equity':      num_equity,
        'bond':        num_bond,
        'alternative': num_alt,
    }

    # -- Stage 4: ETF Selection (rule-based, conflict-free) ------------------
    investor_row = {
        'num_equity_etfs': num_equity,
        'num_bond_etfs':   num_bond,
        'num_alt_etfs':    num_alt,
    }
    portfolio = assign_portfolio_numeric_aum(investor_row, etf_by_class)

    return {
        'risk_profile': risk_profile,
        'allocation':   allocation,
        'total_etfs':   total_etfs,
        'etf_counts':   etf_counts,
        'portfolio':    portfolio,
    }


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------
def print_recommendation(result: dict) -> None:
    print("\n" + "=" * 55)
    print("  PORTFOLIO RECOMMENDATION")
    print("=" * 55)
    print(f"  Risk Profile   : {result['risk_profile']}")
    alloc = result['allocation']
    print(f"  Allocation     : Equity {alloc['equity_pct']*100:.1f}%  |  "
          f"Bonds {alloc['bond_pct']*100:.1f}%  |  "
          f"Alternatives {alloc['alt_pct']*100:.1f}%")
    print(f"  Total ETFs     : {result['total_etfs']}")
    print()
    for asset_class, tickers in result['portfolio'].items():
        count = result['etf_counts'].get(asset_class, len(tickers))
        pct   = alloc.get(f'{asset_class}_pct' if asset_class != 'alternative' else 'alt_pct', 0)
        label = asset_class.capitalize()
        print(f"  {label} ({pct*100:.1f}%):  {', '.join(tickers)}")
    print("=" * 55)


# ---------------------------------------------------------------------------
# Demo — run directly to test
# ---------------------------------------------------------------------------
if __name__ == '__main__':

    test_cases = [
        {
            "name": "Young aggressive investor",
            "age": 25, "Gender": "Male", "Education": "Bachelor",
            "MaritalStatus": "Single", "HouseholdSize": 1,
            "income": 90000, "InvestmentGoal": "Retirement",
            "horizon": 35, "InvestmentCapital": 20000,
            "risk_tolerance": "High", "FinancialInvolvement": "High",
            "experience": 2
        },
        {
            "name": "Middle-aged moderate investor",
            "age": 45, "Gender": "Female", "Education": "Master",
            "MaritalStatus": "Married", "HouseholdSize": 3,
            "income": 130000, "InvestmentGoal": "Child education",
            "horizon": 10, "InvestmentCapital": 150000,
            "risk_tolerance": "Medium", "FinancialInvolvement": "Medium",
            "experience": 15
        },
        {
            "name": "Near-retirement conservative investor",
            "age": 62, "Gender": "Female", "Education": "Bachelor",
            "MaritalStatus": "Married", "HouseholdSize": 2,
            "income": 75000, "InvestmentGoal": "Retirement",
            "horizon": 5, "InvestmentCapital": 400000,
            "risk_tolerance": "Low", "FinancialInvolvement": "Low",
            "experience": 30
        },
    ]

    print("Loading models...")
    for case in test_cases:
        name = case.pop('name')
        print(f"\n--- {name} ---")
        result = recommend(case)
        print_recommendation(result)
