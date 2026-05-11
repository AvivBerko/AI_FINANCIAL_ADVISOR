"""
Shared validation rules for the training pipeline (data_validation.py)
and live inference (inference.py).

Column names use the canonical CSV form produced by data_generation.py
(CapitalCase). Inference.py maps its lowercase keys to these.
"""

# ---------------------------------------------------------------------------
# Investor contract
# ---------------------------------------------------------------------------
INVESTOR_FEATURE_COLUMNS = [
    'Age', 'Gender', 'Education', 'MaritalStatus', 'HouseholdSize',
    'Income', 'InvestmentGoal', 'InvestmentHorizon', 'InvestmentCapital',
    'RiskTolerance', 'FinancialInvolvement', 'ExperienceYears',
]

VALID_VALUES = {
    'Gender':               ['Male', 'Female'],
    'Education':            ['High school', 'Bachelor', 'Master'],
    'MaritalStatus':        ['Married', 'Single', 'Divorced', 'Widowed'],
    'InvestmentGoal':       ['Retirement', 'Home purchase', 'Child education', 'Other'],
    'RiskTolerance':        ['Low', 'Medium', 'High'],
    'FinancialInvolvement': ['Low', 'Medium', 'High'],
}

NUMERIC_RANGES = {
    'Age':               (18, 100),
    'HouseholdSize':     (1, 10),
    'Income':            (0, 10_000_000),
    'InvestmentHorizon': (1, 50),
    'InvestmentCapital': (0, 100_000_000),
    'ExperienceYears':   (0, 80),
}

# ---------------------------------------------------------------------------
# ETF contract
# ---------------------------------------------------------------------------
ETF_REQUIRED_COLUMNS = [
    'Fund Symbol', 'Category',
    'Assets Under Management (AUM)',
    'custom_star_rating',
    'Volatility (Annual STD)',
]

ETF_DROP_IF_MISSING = ['Fund Symbol', 'Category']

# Column -> global fallback (used when per-class median unavailable).
# Matches the existing hardcoded defaults in allocations.py / inference.py
# so behavior is unchanged when per-class median can't be computed.
ETF_IMPUTE_COLUMNS = {
    'Assets Under Management (AUM)': 1e9,
    'custom_star_rating':            3,
    'Volatility (Annual STD)':       0.15,
}

ASSET_CLASS_FLOORS = {'equity': 5, 'bond': 5, 'alternative': 1}

# Keywords recognised by allocations.get_category_mapping(). Mirrored here
# so the ETF validator can detect "unknown category" rows that would silently
# fall through to the 'equity' default.
KNOWN_CATEGORY_KEYWORDS = [
    'large', 'mid', 'small', 'growth', 'value', 'blend',
    'bond', 'corporate', 'treasury', 'government',
    'real estate', 'reit', 'commodities', 'gold',
]
