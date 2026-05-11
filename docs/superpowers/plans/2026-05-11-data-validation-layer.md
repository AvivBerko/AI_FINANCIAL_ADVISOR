# Data Validation Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a validation layer to the training pipeline that drops bad investor rows, imputes bad ETF cells, raises on schema/floor failures, and produces a markdown report.

**Architecture:** Two new files (`src/validation_rules.py` for shared constants, `src/data_validation.py` for two validators + a report dataclass), one refactor (`src/inference.py` imports rules from the new constants module), one integration (`src/prepare_data.py` inserts Step 1.5). Pytest suite covers each validator behavior.

**Tech Stack:** Python 3.x, pandas, numpy, pytest, dataclasses (stdlib).

**Spec reference:** [docs/superpowers/specs/2026-05-11-data-validation-layer-design.md](../specs/2026-05-11-data-validation-layer-design.md)

---

## Task 0: Setup

**Files:**
- Create: `tests/__init__.py` (empty)
- Modify: `requirements.txt` (add pytest)

- [ ] **Step 1: Install pytest**

Run: `venv/bin/pip install pytest`

Expected: pytest installed (version ≥7).

- [ ] **Step 2: Add pytest to requirements.txt**

Append `pytest>=7.0` to `requirements.txt`.

- [ ] **Step 3: Create tests directory + init**

```bash
mkdir -p tests
touch tests/__init__.py
```

- [ ] **Step 4: Verify pytest discovers tests**

Run: `venv/bin/pytest tests/ -v --collect-only`
Expected: `collected 0 items` (no tests yet, but pytest can find the dir).

- [ ] **Step 5: Commit**

```bash
git add tests/__init__.py requirements.txt
git commit -m "chore: add pytest dependency and tests/ directory"
```

---

## Task 1: Validation rules constants module

**Files:**
- Create: `src/validation_rules.py`

- [ ] **Step 1: Create the constants module**

Write `src/validation_rules.py`:

```python
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
```

- [ ] **Step 2: Smoke-import**

Run: `venv/bin/python -c "from src.validation_rules import INVESTOR_FEATURE_COLUMNS, VALID_VALUES, NUMERIC_RANGES, ETF_REQUIRED_COLUMNS, ETF_IMPUTE_COLUMNS, ASSET_CLASS_FLOORS, KNOWN_CATEGORY_KEYWORDS; print(len(INVESTOR_FEATURE_COLUMNS))"`

Expected: `12`

- [ ] **Step 3: Commit**

```bash
git add src/validation_rules.py
git commit -m "feat(validation): add shared validation rules constants module"
```

---

## Task 2: ValidationReport dataclass

**Files:**
- Create: `src/data_validation.py`
- Test: `tests/test_data_validation.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_data_validation.py`:

```python
"""Tests for src/data_validation.py"""
import pytest
import pandas as pd
import numpy as np

from src.data_validation import ValidationReport


def test_validation_report_basic_construction():
    rpt = ValidationReport(name="investor", n_input=100, n_output=98, n_dropped=2)
    rpt.dropped_reasons['nan_in_feature'] = 2
    rpt.warnings.append("2 duplicates detected")
    assert rpt.n_input == 100
    assert rpt.n_output == 98
    assert rpt.dropped_reasons['nan_in_feature'] == 2
    assert "duplicates" in rpt.warnings[0]


def test_validation_report_to_markdown_contains_summary():
    rpt = ValidationReport(name="investor", n_input=100, n_output=98, n_dropped=2)
    rpt.dropped_reasons['nan_in_feature'] = 2
    md = rpt.to_markdown()
    assert "Investor" in md or "investor" in md
    assert "100" in md  # input row count
    assert "98" in md   # output row count
    assert "nan_in_feature" in md
```

- [ ] **Step 2: Verify failure**

Run: `venv/bin/pytest tests/test_data_validation.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'src.data_validation'`.

- [ ] **Step 3: Create data_validation.py with dataclass + to_markdown**

Create `src/data_validation.py`:

```python
"""
Data validation layer for the training pipeline.

Exposes:
  - ValidationReport     : dataclass for human-readable validation results
  - validate_investor_data(df) -> (clean_df, ValidationReport)
  - validate_etf_data(df)      -> (clean_df, ValidationReport)
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from src.validation_rules import (
    INVESTOR_FEATURE_COLUMNS, VALID_VALUES, NUMERIC_RANGES,
    ETF_REQUIRED_COLUMNS, ETF_DROP_IF_MISSING, ETF_IMPUTE_COLUMNS,
    ASSET_CLASS_FLOORS, KNOWN_CATEGORY_KEYWORDS,
)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
@dataclass
class ValidationReport:
    name: str
    n_input: int
    n_output: int
    n_dropped: int
    dropped_reasons:    dict        = field(default_factory=dict)
    imputed:            dict        = field(default_factory=dict)   # ETF only — counts
    imputation_detail:  list        = field(default_factory=list)   # ETF only — per-cell rows
    warnings:           list        = field(default_factory=list)
    critical_issues:    list        = field(default_factory=list)
    asset_class_counts: dict        = field(default_factory=dict)   # ETF only

    def to_markdown(self) -> str:
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        title = f"# {self.name.capitalize()} Data Validation Report"
        pct = (self.n_dropped / self.n_input * 100) if self.n_input else 0.0

        lines = [
            title,
            f"Generated: {ts}",
            "",
            "## Summary",
            f"- Input rows:    {self.n_input}",
            f"- Output rows:   {self.n_output}",
            f"- Dropped:       {self.n_dropped} ({pct:.2f}%)",
            "",
        ]

        if self.dropped_reasons:
            lines.append("## Drop reasons")
            for reason, count in sorted(self.dropped_reasons.items()):
                lines.append(f"- {reason}: {count}")
            lines.append("")

        if self.asset_class_counts:
            lines.append("## Asset class breakdown (post-validation)")
            for cls, count in self.asset_class_counts.items():
                floor = ASSET_CLASS_FLOORS.get(cls, 0)
                status = "OK" if count >= floor else "BELOW FLOOR"
                lines.append(f"- {cls}: {count}  (floor: {floor})  {status}")
            lines.append("")

        if self.imputed:
            lines.append("## Imputations (counts)")
            for col, count in sorted(self.imputed.items()):
                lines.append(f"- {col}: {count}")
            lines.append("")

        if self.imputation_detail:
            lines.append("## Imputation detail")
            for row in self.imputation_detail:
                lines.append(
                    f"- {row['ticker']} | {row['column']} | "
                    f"{row['original']} -> {row['imputed']:.4f} ({row['source']})"
                )
            lines.append("")

        if self.warnings:
            lines.append("## Warnings")
            for w in self.warnings:
                lines.append(f"- {w}")
            lines.append("")

        if self.critical_issues:
            lines.append("## Critical issues")
            for c in self.critical_issues:
                lines.append(f"- {c}")
            lines.append("")

        return "\n".join(lines)
```

- [ ] **Step 4: Run tests, verify pass**

Run: `venv/bin/pytest tests/test_data_validation.py -v`

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/data_validation.py tests/test_data_validation.py
git commit -m "feat(validation): add ValidationReport dataclass with markdown formatter"
```

---

## Task 3: Investor validator — schema check + happy path

**Files:**
- Modify: `src/data_validation.py`
- Modify: `tests/test_data_validation.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/test_data_validation.py`:

```python
# ---------------------------------------------------------------------------
# Fixtures for investor tests
# ---------------------------------------------------------------------------
def _make_good_investor_row(**overrides):
    row = {
        'Age': 30, 'Gender': 'Male', 'Education': 'Bachelor',
        'MaritalStatus': 'Single', 'HouseholdSize': 1,
        'Income': 80000, 'InvestmentGoal': 'Retirement',
        'InvestmentHorizon': 30, 'InvestmentCapital': 50000,
        'RiskTolerance': 'Medium', 'FinancialInvolvement': 'Medium',
        'ExperienceYears': 5,
    }
    row.update(overrides)
    return row


def _make_good_investor_df(n=10):
    return pd.DataFrame([_make_good_investor_row() for _ in range(n)])


# ---------------------------------------------------------------------------
# validate_investor_data
# ---------------------------------------------------------------------------
def test_investor_happy_path_keeps_all_rows():
    from src.data_validation import validate_investor_data
    df = _make_good_investor_df(10)
    clean, rpt = validate_investor_data(df)
    assert rpt.n_input == 10
    assert rpt.n_output == 10
    assert rpt.n_dropped == 0
    assert rpt.dropped_reasons == {}
    assert len(clean) == 10


def test_investor_schema_missing_column_raises():
    from src.data_validation import validate_investor_data
    df = _make_good_investor_df(5).drop(columns=['Age'])
    with pytest.raises(ValueError, match="Age"):
        validate_investor_data(df)
```

- [ ] **Step 2: Verify failure**

Run: `venv/bin/pytest tests/test_data_validation.py -v`
Expected: 2 new tests FAIL with `ImportError` (function not yet defined).

- [ ] **Step 3: Implement schema check + minimal validator**

Append to `src/data_validation.py`:

```python
# ---------------------------------------------------------------------------
# Investor validator
# ---------------------------------------------------------------------------
def validate_investor_data(df: pd.DataFrame) -> tuple[pd.DataFrame, ValidationReport]:
    """
    Returns (cleaned_df, report). Bad rows are DROPPED with reason logged.
    Missing required columns RAISES ValueError.
    """
    rpt = ValidationReport(name="investor", n_input=len(df), n_output=0, n_dropped=0)

    # Schema check
    missing = [c for c in INVESTOR_FEATURE_COLUMNS if c not in df.columns]
    if missing:
        rpt.critical_issues.append(f"Missing columns: {missing}")
        raise ValueError(f"Investor data missing required columns: {missing}")

    # No row-level validation yet — placeholder, completed in Task 4.
    clean = df.copy()
    rpt.n_output = len(clean)
    rpt.n_dropped = rpt.n_input - rpt.n_output
    return clean, rpt
```

- [ ] **Step 4: Run tests, verify pass**

Run: `venv/bin/pytest tests/test_data_validation.py -v`
Expected: 4 passed (2 from Task 2 + 2 new).

- [ ] **Step 5: Commit**

```bash
git add src/data_validation.py tests/test_data_validation.py
git commit -m "feat(validation): add investor schema check and happy-path validator"
```

---

## Task 4: Investor validator — per-row drops

**Files:**
- Modify: `src/data_validation.py`
- Modify: `tests/test_data_validation.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/test_data_validation.py`:

```python
def test_investor_drops_row_with_nan_in_feature():
    from src.data_validation import validate_investor_data
    df = _make_good_investor_df(5)
    df.loc[2, 'Age'] = np.nan
    clean, rpt = validate_investor_data(df)
    assert rpt.n_input == 5
    assert rpt.n_output == 4
    assert rpt.n_dropped == 1
    assert rpt.dropped_reasons.get('nan_in_feature') == 1


def test_investor_drops_row_with_invalid_categorical():
    from src.data_validation import validate_investor_data
    df = _make_good_investor_df(5)
    df.loc[1, 'Gender'] = 'Other'
    clean, rpt = validate_investor_data(df)
    assert rpt.n_dropped == 1
    assert rpt.dropped_reasons.get('invalid_categorical_Gender') == 1


def test_investor_drops_row_with_out_of_range_numeric():
    from src.data_validation import validate_investor_data
    df = _make_good_investor_df(5)
    df.loc[0, 'Age'] = 5  # below range (18, 100)
    df.loc[3, 'Income'] = -1000  # below range (0, 10M)
    clean, rpt = validate_investor_data(df)
    assert rpt.n_dropped == 2
    assert rpt.dropped_reasons.get('out_of_range_Age') == 1
    assert rpt.dropped_reasons.get('out_of_range_Income') == 1


def test_investor_drops_row_with_consistency_violation():
    from src.data_validation import validate_investor_data
    df = _make_good_investor_df(5)
    df.loc[2, 'Age'] = 25
    df.loc[2, 'ExperienceYears'] = 20  # 25 - 18 = 7, so 20 violates
    clean, rpt = validate_investor_data(df)
    assert rpt.n_dropped == 1
    assert rpt.dropped_reasons.get('consistency_experience_vs_age') == 1


def test_investor_row_with_multiple_violations_dropped_once_counts_each():
    from src.data_validation import validate_investor_data
    df = _make_good_investor_df(3)
    df.loc[1, 'Age'] = 5             # out of range
    df.loc[1, 'Gender'] = 'Other'    # invalid categorical
    clean, rpt = validate_investor_data(df)
    assert rpt.n_dropped == 1                              # dropped once
    assert rpt.dropped_reasons.get('out_of_range_Age') == 1
    assert rpt.dropped_reasons.get('invalid_categorical_Gender') == 1
```

- [ ] **Step 2: Verify failure**

Run: `venv/bin/pytest tests/test_data_validation.py -v`
Expected: 5 new tests FAIL.

- [ ] **Step 3: Implement per-row drops**

Replace the placeholder body of `validate_investor_data` in `src/data_validation.py` with:

```python
def validate_investor_data(df: pd.DataFrame) -> tuple[pd.DataFrame, ValidationReport]:
    """
    Returns (cleaned_df, report). Bad rows are DROPPED with reason logged.
    Missing required columns RAISES ValueError.
    """
    rpt = ValidationReport(name="investor", n_input=len(df), n_output=0, n_dropped=0)

    # Schema check
    missing = [c for c in INVESTOR_FEATURE_COLUMNS if c not in df.columns]
    if missing:
        rpt.critical_issues.append(f"Missing columns: {missing}")
        raise ValueError(f"Investor data missing required columns: {missing}")

    # Build a boolean mask of "rows to drop". Increment reason counts per check.
    drop_mask = pd.Series(False, index=df.index)

    # NaN in any feature column
    nan_mask = df[INVESTOR_FEATURE_COLUMNS].isna().any(axis=1)
    n_nan = int(nan_mask.sum())
    if n_nan:
        rpt.dropped_reasons['nan_in_feature'] = n_nan
    drop_mask |= nan_mask

    # Categorical out of allowed set
    for col, allowed in VALID_VALUES.items():
        bad = ~df[col].isin(allowed) & df[col].notna()
        n_bad = int(bad.sum())
        if n_bad:
            rpt.dropped_reasons[f'invalid_categorical_{col}'] = n_bad
        drop_mask |= bad

    # Numeric out of range
    for col, (lo, hi) in NUMERIC_RANGES.items():
        as_num = pd.to_numeric(df[col], errors='coerce')
        bad = as_num.notna() & ((as_num < lo) | (as_num > hi))
        n_bad = int(bad.sum())
        if n_bad:
            rpt.dropped_reasons[f'out_of_range_{col}'] = n_bad
        drop_mask |= bad

    # Consistency: ExperienceYears <= Age - 18
    age_num = pd.to_numeric(df['Age'], errors='coerce')
    exp_num = pd.to_numeric(df['ExperienceYears'], errors='coerce')
    consistency_bad = age_num.notna() & exp_num.notna() & (exp_num > (age_num - 18))
    n_cb = int(consistency_bad.sum())
    if n_cb:
        rpt.dropped_reasons['consistency_experience_vs_age'] = n_cb
    drop_mask |= consistency_bad

    clean = df.loc[~drop_mask].copy()
    rpt.n_output = len(clean)
    rpt.n_dropped = rpt.n_input - rpt.n_output
    return clean, rpt
```

- [ ] **Step 4: Run tests, verify pass**

Run: `venv/bin/pytest tests/test_data_validation.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/data_validation.py tests/test_data_validation.py
git commit -m "feat(validation): drop investor rows with NaN, invalid categorical, out-of-range or consistency issues"
```

---

## Task 5: Investor validator — duplicates + class balance warnings

**Files:**
- Modify: `src/data_validation.py`
- Modify: `tests/test_data_validation.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/test_data_validation.py`:

```python
def test_investor_duplicates_warn_only_not_dropped():
    from src.data_validation import validate_investor_data
    df = pd.concat([_make_good_investor_df(5), _make_good_investor_df(2)], ignore_index=True)
    # rows 0-6 are all identical fixtures so there will be duplicates
    clean, rpt = validate_investor_data(df)
    assert rpt.n_dropped == 0
    assert any("duplicate" in w.lower() for w in rpt.warnings)


def test_investor_class_balance_warns_when_skewed():
    from src.data_validation import validate_investor_data
    # 80% High, 10% Medium, 10% Low — heavily skewed
    df = pd.concat([
        pd.DataFrame([_make_good_investor_row(RiskTolerance='High') for _ in range(8)]),
        pd.DataFrame([_make_good_investor_row(RiskTolerance='Medium') for _ in range(1)]),
        pd.DataFrame([_make_good_investor_row(RiskTolerance='Low') for _ in range(1)]),
    ], ignore_index=True)
    clean, rpt = validate_investor_data(df)
    assert any("class balance" in w.lower() or "imbalance" in w.lower() for w in rpt.warnings)
```

- [ ] **Step 2: Verify failure**

Run: `venv/bin/pytest tests/test_data_validation.py -v`
Expected: 2 new tests FAIL.

- [ ] **Step 3: Append warning checks**

In `src/data_validation.py`, replace the tail of `validate_investor_data` — the existing lines starting at `clean = df.loc[~drop_mask].copy()` through the end of the function — with:

```python
    clean = df.loc[~drop_mask].copy()

    # Duplicates (warn only, don't drop)
    n_dupes = int(df.duplicated().sum())
    if n_dupes:
        rpt.warnings.append(f"{n_dupes} duplicate rows detected (kept)")

    # Class balance (warn only) — uses cleaned df
    if 'RiskTolerance' in clean.columns and len(clean):
        dist = clean['RiskTolerance'].value_counts(normalize=True) * 100
        for cls in ['Low', 'Medium', 'High']:
            pct = float(dist.get(cls, 0.0))
            if pct < 25 or pct > 40:
                rpt.warnings.append(
                    f"Class balance off: RiskTolerance={cls} is {pct:.1f}% (target ~33%)"
                )

    rpt.n_output = len(clean)
    rpt.n_dropped = rpt.n_input - rpt.n_output
    return clean, rpt
```

(Remove the old trailing three lines `clean = df.loc[~drop_mask].copy()` / `rpt.n_output` / `rpt.n_dropped` / `return` since they're now inside this block.)

- [ ] **Step 4: Run tests, verify pass**

Run: `venv/bin/pytest tests/test_data_validation.py -v`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add src/data_validation.py tests/test_data_validation.py
git commit -m "feat(validation): warn on investor duplicates and class imbalance"
```

---

## Task 6: ETF validator — schema check + drop unusable rows

**Files:**
- Modify: `src/data_validation.py`
- Modify: `tests/test_data_validation.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/test_data_validation.py`:

```python
# ---------------------------------------------------------------------------
# Fixtures for ETF tests
# ---------------------------------------------------------------------------
def _make_good_etf_row(**overrides):
    row = {
        'Fund Symbol': 'VOO',
        'Category': 'Large Blend',
        'Assets Under Management (AUM)': 1_000_000_000.0,
        'custom_star_rating': 4.0,
        'Volatility (Annual STD)': 0.18,
    }
    row.update(overrides)
    return row


def _make_good_etf_df():
    """Returns 12 ETFs: 6 equity, 5 bond, 1 alternative — passes all floors."""
    rows = (
        [_make_good_etf_row(**{'Fund Symbol': f'E{i}', 'Category': 'Large Blend'})        for i in range(6)] +
        [_make_good_etf_row(**{'Fund Symbol': f'B{i}', 'Category': 'Corporate Bond'})     for i in range(5)] +
        [_make_good_etf_row(**{'Fund Symbol': 'GLD',   'Category': 'Gold'})]
    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# validate_etf_data
# ---------------------------------------------------------------------------
def test_etf_happy_path_keeps_all_rows():
    from src.data_validation import validate_etf_data
    df = _make_good_etf_df()
    clean, rpt = validate_etf_data(df)
    assert rpt.n_input == 12
    assert rpt.n_output == 12
    assert rpt.n_dropped == 0


def test_etf_schema_missing_column_raises():
    from src.data_validation import validate_etf_data
    df = _make_good_etf_df().drop(columns=['Category'])
    with pytest.raises(ValueError, match="Category"):
        validate_etf_data(df)


def test_etf_drops_missing_fund_symbol():
    from src.data_validation import validate_etf_data
    df = _make_good_etf_df()
    df.loc[0, 'Fund Symbol'] = None
    clean, rpt = validate_etf_data(df)
    assert rpt.n_dropped == 1
    assert rpt.dropped_reasons.get('missing_fund_symbol') == 1


def test_etf_drops_missing_category():
    from src.data_validation import validate_etf_data
    df = _make_good_etf_df()
    df.loc[0, 'Category'] = ''   # empty
    df.loc[1, 'Category'] = None
    clean, rpt = validate_etf_data(df)
    assert rpt.n_dropped == 2
    assert rpt.dropped_reasons.get('missing_category') == 2
```

- [ ] **Step 2: Verify failure**

Run: `venv/bin/pytest tests/test_data_validation.py -v`
Expected: 4 new tests FAIL.

- [ ] **Step 3: Implement ETF validator skeleton**

Append to `src/data_validation.py`:

```python
# ---------------------------------------------------------------------------
# ETF validator helpers
# ---------------------------------------------------------------------------
def _is_known_category(category) -> bool:
    if not isinstance(category, str):
        return False
    cat = category.lower()
    return any(k in cat for k in KNOWN_CATEGORY_KEYWORDS)


def _map_to_asset_class(category) -> str:
    """Same logic as allocations.get_category_mapping (kept local to avoid
    circular concerns during validation)."""
    cat = str(category).lower()
    if any(x in cat for x in ['large', 'mid', 'small', 'growth', 'value', 'blend']):
        return 'equity'
    if any(x in cat for x in ['bond', 'corporate', 'treasury', 'government']):
        return 'bond'
    if any(x in cat for x in ['real estate', 'reit', 'commodities', 'gold']):
        return 'alternative'
    return 'equity'  # default


def validate_etf_data(df: pd.DataFrame) -> tuple[pd.DataFrame, ValidationReport]:
    """
    Returns (cleaned_df, report). Rows missing Fund Symbol or Category are
    dropped. Bad/missing AUM/star_rating/volatility are imputed with per-
    asset-class median (or global default). Asset-class floor violations RAISE.
    """
    rpt = ValidationReport(name="etf", n_input=len(df), n_output=0, n_dropped=0)

    # Schema check
    missing = [c for c in ETF_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        rpt.critical_issues.append(f"Missing columns: {missing}")
        raise ValueError(f"ETF data missing required columns: {missing}")

    work = df.copy()

    # Drop missing Fund Symbol
    sym_bad = work['Fund Symbol'].isna() | (work['Fund Symbol'].astype(str).str.strip() == '')
    n_sym_bad = int(sym_bad.sum())
    if n_sym_bad:
        rpt.dropped_reasons['missing_fund_symbol'] = n_sym_bad
        work = work.loc[~sym_bad].copy()

    # Drop missing Category
    cat_bad = work['Category'].isna() | (work['Category'].astype(str).str.strip() == '')
    n_cat_bad = int(cat_bad.sum())
    if n_cat_bad:
        rpt.dropped_reasons['missing_category'] = n_cat_bad
        work = work.loc[~cat_bad].copy()

    # Imputation + asset-class floor will be added in subsequent tasks.

    rpt.n_output = len(work)
    rpt.n_dropped = rpt.n_input - rpt.n_output
    return work, rpt
```

- [ ] **Step 4: Run tests, verify pass**

Run: `venv/bin/pytest tests/test_data_validation.py -v`
Expected: 15 passed.

- [ ] **Step 5: Commit**

```bash
git add src/data_validation.py tests/test_data_validation.py
git commit -m "feat(validation): add ETF schema check and drop rows with missing Fund Symbol or Category"
```

---

## Task 7: ETF validator — per-class median imputation

**Files:**
- Modify: `src/data_validation.py`
- Modify: `tests/test_data_validation.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/test_data_validation.py`:

```python
def test_etf_imputes_nan_aum_with_class_median():
    from src.data_validation import validate_etf_data
    df = _make_good_etf_df()
    # Equity class has 6 ETFs, AUM all 1e9. Make 1 NaN.
    df.loc[0, 'Assets Under Management (AUM)'] = np.nan
    clean, rpt = validate_etf_data(df)
    assert rpt.n_dropped == 0
    assert rpt.imputed.get('Assets Under Management (AUM)') == 1
    imputed_value = clean.iloc[0]['Assets Under Management (AUM)']
    assert imputed_value == 1e9  # class median
    detail = [d for d in rpt.imputation_detail
              if d['column'] == 'Assets Under Management (AUM)'][0]
    assert detail['source'].startswith('class_median')


def test_etf_imputes_negative_volatility_treated_as_bad():
    from src.data_validation import validate_etf_data
    df = _make_good_etf_df()
    df.loc[0, 'Volatility (Annual STD)'] = -0.05
    clean, rpt = validate_etf_data(df)
    assert rpt.imputed.get('Volatility (Annual STD)') == 1
    assert clean.iloc[0]['Volatility (Annual STD)'] > 0


def test_etf_imputes_with_global_default_when_class_has_too_few_samples():
    from src.data_validation import validate_etf_data
    # Alternative class has only 1 ETF (GLD). If its AUM is bad, can't
    # compute median (< 3 valid samples) — must use global default 1e9.
    df = _make_good_etf_df()
    gld_idx = df[df['Fund Symbol'] == 'GLD'].index[0]
    df.loc[gld_idx, 'Assets Under Management (AUM)'] = np.nan
    clean, rpt = validate_etf_data(df)
    detail = [d for d in rpt.imputation_detail
              if d['ticker'] == 'GLD'
              and d['column'] == 'Assets Under Management (AUM)'][0]
    assert detail['source'] == 'global_default'
    assert detail['imputed'] == 1e9
```

- [ ] **Step 2: Verify failure**

Run: `venv/bin/pytest tests/test_data_validation.py -v`
Expected: 3 new tests FAIL.

- [ ] **Step 3: Implement imputation**

In `src/data_validation.py`, modify `validate_etf_data` by inserting the following block AFTER the "Drop missing Category" block and BEFORE `rpt.n_output = len(work)`:

```python
    # Assign asset_class
    work['_asset_class'] = work['Category'].apply(_map_to_asset_class)

    # Detect unknown-category rows that silently fell to 'equity' default
    unknown_mask = ~work['Category'].apply(_is_known_category)
    if unknown_mask.any():
        unknown_tickers = work.loc[unknown_mask, 'Fund Symbol'].tolist()
        rpt.warnings.append(
            f"{len(unknown_tickers)} ETF(s) with unknown category mapped to 'equity' by default: {unknown_tickers}"
        )

    # Per-class medians for impute columns. Compute over rows where value
    # is valid (notna, positive, finite). If class has < 3 valid values,
    # the per-class median is unreliable -> fall back to global default.
    MIN_VALID_FOR_MEDIAN = 3
    class_medians = {}  # {column: {class: median or None}}
    for col in ETF_IMPUTE_COLUMNS:
        as_num = pd.to_numeric(work[col], errors='coerce')
        valid = as_num.notna() & np.isfinite(as_num) & (as_num > 0)
        class_medians[col] = {}
        for cls in work['_asset_class'].unique():
            cls_mask = work['_asset_class'] == cls
            valid_vals = as_num.loc[cls_mask & valid]
            if len(valid_vals) >= MIN_VALID_FOR_MEDIAN:
                class_medians[col][cls] = float(valid_vals.median())
            else:
                class_medians[col][cls] = None  # use global default

    # Impute bad cells
    for col, global_default in ETF_IMPUTE_COLUMNS.items():
        as_num = pd.to_numeric(work[col], errors='coerce')
        bad = ~(as_num.notna() & np.isfinite(as_num) & (as_num > 0))
        if not bad.any():
            continue
        rpt.imputed[col] = int(bad.sum())
        for idx in work.index[bad]:
            cls = work.at[idx, '_asset_class']
            class_med = class_medians[col].get(cls)
            if class_med is not None:
                imputed_val = class_med
                source = f"class_median:{cls}"
            else:
                imputed_val = global_default
                source = "global_default"
            original = work.at[idx, col]
            work.at[idx, col] = imputed_val
            rpt.imputation_detail.append({
                'ticker': work.at[idx, 'Fund Symbol'],
                'column': col,
                'original': None if pd.isna(original) else float(original),
                'imputed': float(imputed_val),
                'source': source,
            })

    # Drop the helper column before returning
    work = work.drop(columns=['_asset_class'])
```

- [ ] **Step 4: Run tests, verify pass**

Run: `venv/bin/pytest tests/test_data_validation.py -v`
Expected: 18 passed.

- [ ] **Step 5: Commit**

```bash
git add src/data_validation.py tests/test_data_validation.py
git commit -m "feat(validation): impute bad/missing ETF AUM, star_rating, volatility with per-class median"
```

---

## Task 8: ETF validator — asset class floor check

**Files:**
- Modify: `src/data_validation.py`
- Modify: `tests/test_data_validation.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/test_data_validation.py`:

```python
def test_etf_asset_class_counts_reported():
    from src.data_validation import validate_etf_data
    df = _make_good_etf_df()
    clean, rpt = validate_etf_data(df)
    assert rpt.asset_class_counts.get('equity') == 6
    assert rpt.asset_class_counts.get('bond') == 5
    assert rpt.asset_class_counts.get('alternative') == 1


def test_etf_floor_violation_raises():
    from src.data_validation import validate_etf_data
    # 6 equity, 5 bond, 0 alternative -> alternative floor (1) violated
    rows = (
        [_make_good_etf_row(**{'Fund Symbol': f'E{i}', 'Category': 'Large Blend'})    for i in range(6)] +
        [_make_good_etf_row(**{'Fund Symbol': f'B{i}', 'Category': 'Corporate Bond'}) for i in range(5)]
    )
    df = pd.DataFrame(rows)
    with pytest.raises(ValueError, match="alternative"):
        validate_etf_data(df)


def test_etf_floor_violation_multiple_classes():
    from src.data_validation import validate_etf_data
    # 2 equity (below 5), 5 bond, 1 alternative
    rows = (
        [_make_good_etf_row(**{'Fund Symbol': f'E{i}', 'Category': 'Large Blend'})    for i in range(2)] +
        [_make_good_etf_row(**{'Fund Symbol': f'B{i}', 'Category': 'Corporate Bond'}) for i in range(5)] +
        [_make_good_etf_row(**{'Fund Symbol': 'GLD',   'Category': 'Gold'})]
    )
    df = pd.DataFrame(rows)
    with pytest.raises(ValueError, match="equity"):
        validate_etf_data(df)
```

- [ ] **Step 2: Verify failure**

Run: `venv/bin/pytest tests/test_data_validation.py -v`
Expected: 3 new tests FAIL.

- [ ] **Step 3: Implement floor check**

In `src/data_validation.py`, modify `validate_etf_data`. Replace the line `work = work.drop(columns=['_asset_class'])` with the following (so the floor check runs BEFORE we drop the helper column):

```python
    # Asset class floor (raise if any class below its floor)
    asset_counts = work['_asset_class'].value_counts().to_dict()
    rpt.asset_class_counts = {
        cls: int(asset_counts.get(cls, 0))
        for cls in ['equity', 'bond', 'alternative']
    }

    deficient = [
        f"{cls} has {rpt.asset_class_counts[cls]} ETFs (floor: {floor})"
        for cls, floor in ASSET_CLASS_FLOORS.items()
        if rpt.asset_class_counts[cls] < floor
    ]
    if deficient:
        msg = "Asset class floor violation: " + "; ".join(deficient)
        rpt.critical_issues.append(msg)
        raise ValueError(msg)

    # Drop the helper column before returning
    work = work.drop(columns=['_asset_class'])
```

- [ ] **Step 4: Run tests, verify pass**

Run: `venv/bin/pytest tests/test_data_validation.py -v`
Expected: 21 passed.

- [ ] **Step 5: Commit**

```bash
git add src/data_validation.py tests/test_data_validation.py
git commit -m "feat(validation): raise on ETF asset-class floor violation (equity/bond/alt)"
```

---

## Task 9: Refactor inference.py to use validation_rules

**Files:**
- Modify: `src/inference.py:46-61, 110-133`
- Test: smoke test only (inference.py has no existing tests; we run its `__main__` demo path is heavy. We'll just import-check + call validate_input).

- [ ] **Step 1: Append failing tests**

Append to `tests/test_data_validation.py`:

```python
# ---------------------------------------------------------------------------
# inference.validate_input refactor smoke
# ---------------------------------------------------------------------------
def test_inference_validate_input_imports_from_validation_rules():
    """After refactor, inference.py should not hold its own VALID_VALUES dict —
    it must reference src.validation_rules.VALID_VALUES."""
    import src.inference as inf
    from src import validation_rules
    # The constants live in validation_rules now
    assert hasattr(validation_rules, 'VALID_VALUES')
    assert hasattr(validation_rules, 'NUMERIC_RANGES')
    # inference.py either imports them, or aliases them
    src_text = open(inf.__file__).read()
    assert 'from src.validation_rules' in src_text or 'from .validation_rules' in src_text


def test_inference_validate_input_still_works():
    """Behaviour-preserving refactor — same valid input passes, same invalid
    input raises."""
    from src.inference import validate_input
    good = {
        "age": 30, "Gender": "Male", "Education": "Bachelor",
        "MaritalStatus": "Single", "HouseholdSize": 1,
        "income": 80000, "InvestmentGoal": "Retirement",
        "horizon": 30, "InvestmentCapital": 50000,
        "risk_tolerance": "Medium", "FinancialInvolvement": "Medium",
        "experience": 5,
    }
    validate_input(good)  # should not raise

    bad_gender = dict(good); bad_gender['Gender'] = 'Other'
    with pytest.raises(ValueError, match="Gender"):
        validate_input(bad_gender)

    bad_age = dict(good); bad_age['age'] = 5
    with pytest.raises(ValueError, match="age"):
        validate_input(bad_age)
```

- [ ] **Step 2: Verify failure**

Run: `venv/bin/pytest tests/test_data_validation.py -v -k inference`
Expected: 2 new tests FAIL (first one fails because the import line doesn't exist yet).

- [ ] **Step 3: Replace inference.py's inline rules with imports**

In `src/inference.py`, locate lines 46-61 (the existing `FEATURE_COLUMNS` and `VALID_VALUES` blocks). Replace the entire block:

```python
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
```

…with:

```python
from src.validation_rules import (
    VALID_VALUES as _CSV_VALID_VALUES,
    NUMERIC_RANGES as _CSV_NUMERIC_RANGES,
)

# Inference dict uses lowercase keys for some numeric fields; map to CSV-canonical
# names used by validation_rules.
FEATURE_COLUMNS = [
    'age', 'Gender', 'Education', 'MaritalStatus', 'HouseholdSize',
    'income', 'InvestmentGoal', 'horizon', 'InvestmentCapital',
    'risk_tolerance', 'FinancialInvolvement', 'experience'
]
_INFERENCE_TO_CSV = {
    'age': 'Age',
    'income': 'Income',
    'horizon': 'InvestmentHorizon',
    'risk_tolerance': 'RiskTolerance',
    'experience': 'ExperienceYears',
}
def _to_csv(k): return _INFERENCE_TO_CSV.get(k, k)

# Build inference-keyed lookups from the canonical CSV ones
VALID_VALUES   = {k: _CSV_VALID_VALUES[_to_csv(k)]   for k in FEATURE_COLUMNS if _to_csv(k) in _CSV_VALID_VALUES}
NUMERIC_RANGES = {k: _CSV_NUMERIC_RANGES[_to_csv(k)] for k in FEATURE_COLUMNS if _to_csv(k) in _CSV_NUMERIC_RANGES}
```

- [ ] **Step 4: Replace the inline numeric_checks dict in `validate_input`**

In `src/inference.py`, locate the `validate_input` function (around line 110-133). Replace its body:

```python
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
```

…with:

```python
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

    for field, (lo, hi) in NUMERIC_RANGES.items():
        val = user.get(field)
        if not (lo <= val <= hi):
            raise ValueError(f"'{field}' = {val} is out of range [{lo}, {hi}]")
```

- [ ] **Step 5: Run tests, verify pass**

Run: `venv/bin/pytest tests/test_data_validation.py -v`
Expected: 23 passed.

- [ ] **Step 6: Commit**

```bash
git add src/inference.py tests/test_data_validation.py
git commit -m "refactor(inference): import VALID_VALUES and NUMERIC_RANGES from validation_rules"
```

---

## Task 10: Integrate validation into prepare_data.py

**Files:**
- Modify: `src/prepare_data.py` (insert Step 1.5 between Step 1 and Step 2)

- [ ] **Step 1: Insert validation block**

In `src/prepare_data.py`, locate the section ending at line 53 (the print `Loaded {len(etf_df)} ETFs and {len(investors_df)} investors`). Immediately after that line, insert:

```python


# ---------------------------------------------------------------------------
# 1.5. Validate data
# ---------------------------------------------------------------------------
from datetime import datetime as _dt
from src.data_validation import validate_etf_data, validate_investor_data

print("\nRunning data validation...")
etf_df,       etf_report = validate_etf_data(etf_df)
investors_df, inv_report = validate_investor_data(investors_df)

print(etf_report.to_markdown())
print(inv_report.to_markdown())

# Save reports for the academic record
_ts = _dt.now().strftime('%Y%m%d%H%M%S')
os.makedirs('data/validation_reports', exist_ok=True)
with open(f'data/validation_reports/etf_{_ts}.md',      'w') as f: f.write(etf_report.to_markdown())
with open(f'data/validation_reports/investor_{_ts}.md', 'w') as f: f.write(inv_report.to_markdown())

print(f"\nValidation passed. After validation: {len(etf_df)} ETFs and {len(investors_df)} investors.")
```

- [ ] **Step 2: Smoke-run prepare_data.py if artifacts exist**

Check if input artifacts exist:

```bash
ls data/raw/combined_etf_with_morning_star.csv data/processed/synthetic_investor_data_*.csv 2>/dev/null
```

If both exist, run:

```bash
venv/bin/python src/prepare_data.py
```

Expected: Validation block prints two markdown reports; `data/validation_reports/etf_<ts>.md` and `data/validation_reports/investor_<ts>.md` are created; the rest of the script completes normally.

If the input CSVs do NOT exist on disk, skip the end-to-end run (per project memory artifacts are gitignored and not present). Continue to Step 3.

- [ ] **Step 3: Verify integration unit test still passes**

Run: `venv/bin/pytest tests/test_data_validation.py -v`
Expected: 23 passed.

- [ ] **Step 4: Commit**

```bash
git add src/prepare_data.py
git commit -m "feat(prepare_data): run data validation between load and allocations pipeline"
```

---

## Task 11: Update gitignore for validation reports

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Add validation_reports to gitignore**

Append to `.gitignore`:

```
# Validation reports (regenerated per run)
data/validation_reports/
```

- [ ] **Step 2: Verify no validation reports tracked**

Run: `git status --short data/validation_reports/ 2>/dev/null; git ls-files data/validation_reports/ 2>/dev/null`
Expected: no output (untracked, gitignored).

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: gitignore data/validation_reports/"
```

---

## Final verification

- [ ] **Step 1: Full test suite**

Run: `venv/bin/pytest tests/ -v`
Expected: 23 passed (or more if Tasks added tests).

- [ ] **Step 2: Confirm git log**

Run: `git log --oneline -15`
Expected: a clean sequence of commits from Task 0 through Task 11.

- [ ] **Step 3: Confirm files exist**

```bash
ls src/validation_rules.py src/data_validation.py tests/test_data_validation.py
```
Expected: all three listed.

---

## Self-review notes

**Spec coverage:** Every section of the spec maps to a task: rules module → Task 1; report dataclass → Task 2; investor schema → Task 3; investor row drops → Task 4; investor warnings → Task 5; ETF schema + row drops → Task 6; ETF imputation → Task 7; ETF floor → Task 8; inference refactor → Task 9; prepare_data integration → Task 10; gitignore → Task 11.

**Type consistency:** `ValidationReport` field names (`dropped_reasons`, `imputed`, `imputation_detail`, `warnings`, `critical_issues`, `asset_class_counts`) are used identically across the implementation and tests. `_map_to_asset_class` mirrors `allocations.get_category_mapping` exactly.

**Imputation detail row shape:** Defined in Task 7 as `{ticker, column, original, imputed, source}` — matches the spec.

**Out-of-scope items honored:** No outlier detection, no type coercion, no `data/training/` validation, no `extract_allocation` fix.
