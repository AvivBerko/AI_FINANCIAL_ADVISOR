# Data Validation Layer — Design Spec

**Date:** 2026-05-11
**Status:** Approved for implementation
**Scope:** AI Financial Advisor project, training pipeline

## Problem

The training pipeline ([src/prepare_data.py](../../../src/prepare_data.py)) currently trusts its input CSVs completely. There is no schema enforcement, no NaN audit, no range checking, no consistency verification, and no duplicate detection on the investor data feeding the ML features. The ETF side has silent imputation of missing AUM/star_rating/volatility scattered across [src/allocations.py:267-269](../../../src/allocations.py#L267-L269) and [src/inference.py:86-88](../../../src/inference.py#L86-L88) with no visibility into which rows were defaulted. The validation logic that exists ([src/inference.py:110-133](../../../src/inference.py#L110-L133)) only runs at live inference time, never at training time, and the rules are duplicated between locations with no shared source of truth.

This produces three failure modes:
1. **Silent training corruption** — NaN or out-of-range values flow into `StandardScaler` + `OneHotEncoder` and produce model-poisoning features without warning.
2. **Schema drift breaks pipeline with cryptic errors** — a renamed column in `data_generation.py` would fail deep inside `allocations.py` instead of at the obvious point.
3. **No data-quality record** — for the academic report, there is no defensible "data engineering" artifact showing what was validated, dropped, or imputed.

## Goals

1. **Enforce + report.** Critical issues raise a clear error; minor issues are logged and counted in a report saved to disk.
2. **One source of truth** for validation rules (allowed categorical values, numeric ranges) shared between train-time and inference-time.
3. **Tolerant of synthetic data quirks** — bad investor rows are dropped + logged, not raised. The 5000-row dataset can absorb losses.
4. **Tolerant of small ETF universe** — the 95-row ETF set cannot afford to drop rows over imputable issues. Formalize the existing implicit imputation as transparent, logged behavior.
5. **Aggregate safety net** — catch the known "0 alternative-class ETFs" failure mode before it reaches the rule-based ETF selection.

## Architecture

Two new files + two modified files.

```
src/
├── validation_rules.py    ← NEW. Pure constants. No logic.
├── data_validation.py     ← NEW. Two validators + shared report dataclass.
├── inference.py           ← MODIFIED. Imports rules from validation_rules.py.
└── prepare_data.py        ← MODIFIED. Calls validators between load and allocations pipeline.
```

### `src/validation_rules.py`

Pure constants module. Imported by `data_validation.py` (train-time) and `inference.py` (live-time).

```python
# Investor input contract
INVESTOR_FEATURE_COLUMNS = [
    'Age', 'Gender', 'Education', 'MaritalStatus', 'HouseholdSize',
    'Income', 'InvestmentGoal', 'InvestmentHorizon', 'InvestmentCapital',
    'RiskTolerance', 'FinancialInvolvement', 'ExperienceYears'
]
# Note: these are the data_generation.py column names. Renames to lowercase
# happen inside run_allocations_pipeline (allocations.py). Validation runs
# BEFORE the rename, so we use the original names.

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

# ETF contract
ETF_REQUIRED_COLUMNS = [
    'Fund Symbol', 'Category',
    'Assets Under Management (AUM)',
    'custom_star_rating',
    'Volatility (Annual STD)',
]
ETF_DROP_IF_MISSING = ['Fund Symbol', 'Category']  # truly unusable
ETF_IMPUTE_COLUMNS  = {
    'Assets Under Management (AUM)': 1e9,
    'custom_star_rating':            3,
    'Volatility (Annual STD)':       0.15,
}  # column → global fallback (used when per-class median unavailable)
ASSET_CLASS_FLOORS = {'equity': 5, 'bond': 5, 'alternative': 1}
```

`inference.py` will import `VALID_VALUES` and `NUMERIC_RANGES` from this module. The inference-time mapping (lowercase keys: `age`, `income`, etc.) will be derived by mapping the canonical names → inference-time names inside `inference.validate_input()`, keeping the constants module canonical.

### `src/data_validation.py`

```python
from dataclasses import dataclass, field

@dataclass
class ValidationReport:
    name: str                                # "investor" or "etf"
    n_input: int
    n_output: int
    n_dropped: int
    dropped_reasons: dict[str, int]   = field(default_factory=dict)
    imputed: dict[str, int]           = field(default_factory=dict)   # ETF only — counts
    imputation_detail: list[dict]     = field(default_factory=list)   # ETF only — per-cell rows
    warnings: list[str]               = field(default_factory=list)
    critical_issues: list[str]        = field(default_factory=list)   # populated before raise

    def to_markdown(self) -> str: ...
    def to_dict(self) -> dict: ...                                    # for JSON if needed

# imputation_detail rows have shape:
#   {ticker: str, column: str, original: float|None, imputed: float, source: str}
#   source is one of: "class_median:<asset_class>" or "global_default"


def validate_investor_data(df) -> tuple[pd.DataFrame, ValidationReport]:
    """
    Returns (cleaned_df, report). Bad rows are DROPPED with reason logged.
    Schema-level issues (missing required columns) RAISE ValueError.
    """

def validate_etf_data(df) -> tuple[pd.DataFrame, ValidationReport]:
    """
    Returns (cleaned_df, report). Rows with missing Fund Symbol or Category
    are dropped. Bad/missing AUM/star_rating/volatility are IMPUTED with
    per-asset-class median (fallback: global default). Asset-class floor
    violations RAISE.
    """
```

## Investor validation flow

1. **Schema check (RAISES):** All columns in `INVESTOR_FEATURE_COLUMNS` must exist in df. Missing columns indicate code drift and should crash loudly. Raise `ValueError` with the list of missing columns.

2. **Per-row drop + log:** Iterate (vectorized where possible). A row is dropped if **any** of:
   - Any feature column value is NaN/null
   - A categorical value is not in `VALID_VALUES[column]`
   - A numeric value falls outside `NUMERIC_RANGES[column]` (inclusive bounds)
   - Consistency violation: `ExperienceYears > Age - 18`, or `HouseholdSize < 1`, or `InvestmentCapital < 0` (the last two are also caught by range checks; explicit for clarity)

   Each drop reason is incremented in `report.dropped_reasons`. A single row can have multiple violations — count each one it triggered, but drop the row once.

3. **Duplicate detection (WARN ONLY):** `df.duplicated().sum()` reported in warnings. Duplicates are not dropped (synthetic data with `seed=42` should produce 0 — any duplicate indicates an upstream change worth surfacing, but doesn't block training).

4. **Class balance check (WARN ONLY):** Compute `RiskTolerance` class distribution. If any class is below 25% or above 40%, add a warning. The 33/33/33 target is from `data_generation.py:217-222`.

5. **Return** the filtered df + report.

## ETF validation flow

1. **Schema check (RAISES):** All columns in `ETF_REQUIRED_COLUMNS` must exist. Missing → raise.

2. **Drop unusable rows:**
   - `Fund Symbol` NaN/empty → drop, log as `missing_fund_symbol`
   - `Category` NaN/empty → drop, log as `missing_category`
   - Note: we keep the existing tolerant `get_category_mapping` behavior (which defaults unknown categories to `equity`). A separate warning is added per ETF that hit the default path, but they are NOT dropped.

3. **Assign `asset_class`** to remaining rows using `from src.allocations import get_category_mapping`.

4. **Compute per-class medians** for `Assets Under Management (AUM)`, `custom_star_rating`, `Volatility (Annual STD)`. Only over rows where the value is present AND valid (positive, finite). If a class has < 3 valid values for a column, the per-class median is considered unreliable and the global default from `ETF_IMPUTE_COLUMNS` is used instead.

5. **Impute bad/missing values per row** for the three impute columns. "Bad" = NaN, infinite, negative, or zero. For each imputed cell:
   - Use the asset-class median if available (≥3 valid samples)
   - Else use the global default
   - Increment `report.imputed[column_name]`
   - Append a line to a per-imputation detail log (kept in the report for transparency: ticker → column → original → imputed → source)

6. **Asset class floor (RAISES):** After imputation, count ETFs per asset class. If `equity < 5` or `bond < 5` or `alternative < 1`, raise `ValueError` listing the deficient class(es). This is the safety net for the known "0 alternative ETFs" failure mode.

7. **Return** the cleaned + imputed df + report.

## Report format (markdown, saved to disk)

Each report is saved to `data/validation_reports/<name>_<timestamp>.md`.

Example investor report:

```
# Investor Data Validation Report
Generated: 2026-05-11 14:23:01

## Summary
- Input rows:    5000
- Output rows:   4983
- Dropped:       17 (0.34%)

## Drop reasons
- NaN in feature column:        8
- Categorical out of set:       5
- Numeric out of range:         3
- Consistency violation (exp):  1

## Warnings
- 0 duplicate rows detected (kept)
- Class balance: Aggressive 31.2%, Moderate 35.1%, Conservative 33.7% (OK)
```

Example ETF report:

```
# ETF Data Validation Report
Generated: 2026-05-11 14:23:01

## Summary
- Input rows:    95
- Output rows:   94
- Dropped:       1 (1.05%)

## Drop reasons
- missing_fund_symbol: 0
- missing_category:    1

## Asset class breakdown (post-validation)
- equity:      62  (floor: 5)  ✓
- bond:        28  (floor: 5)  ✓
- alternative:  4  (floor: 1)  ✓

## Imputations
- Assets Under Management (AUM):   2
- custom_star_rating:               4
- Volatility (Annual STD):          1

## Imputation detail
- SPYM | custom_star_rating | NaN → 3.0 (global default; class median unavailable)
- IBIT | Volatility (Annual STD) | -0.05 → 0.18 (class median: equity)
- ...

## Warnings
- 3 ETFs with unknown category mapped to 'equity' by default: [...]
```

## Integration into `prepare_data.py`

A new Step 1.5 inserted between Step 1 (load) and Step 2 (allocations pipeline):

```python
from datetime import datetime
from src.data_validation import validate_investor_data, validate_etf_data

# Step 1.5 — Data Validation
print("\nRunning data validation...")
etf_df,        etf_report = validate_etf_data(etf_df)
investors_df,  inv_report = validate_investor_data(investors_df)

print(etf_report.to_markdown())
print(inv_report.to_markdown())

# Save reports for the academic record
ts = datetime.now().strftime('%Y%m%d%H%M%S')
os.makedirs('data/validation_reports', exist_ok=True)
with open(f'data/validation_reports/etf_{ts}.md',      'w') as f: f.write(etf_report.to_markdown())
with open(f'data/validation_reports/investor_{ts}.md', 'w') as f: f.write(inv_report.to_markdown())
```

Steps 2-6 of `prepare_data.py` are unchanged.

## Changes to `inference.py`

Replace the inline `FEATURE_COLUMNS`, `VALID_VALUES`, and the numeric checks dict inside `validate_input()` with imports from `validation_rules.py`. The inference-time feature names use lowercase keys (`age`, `income`, etc.), but the constants module uses the canonical CSV names (`Age`, `Income`, etc.). `inference.validate_input()` will own a small CSV-name → inference-name mapping so the constants module stays canonical and untouched.

This is a pure refactor — same behavior, single source of truth.

## Error handling

- All RAISE paths emit `ValueError` with a clear human-readable message identifying the file/column.
- Before raising, the `ValidationReport.critical_issues` list is populated and the report is still produced (so the user can see what was found).
- Warnings never raise; they accumulate in `report.warnings`.

## Testing

Minimal pytest suite covering the validators (no CI exists in this project; tests are for confidence during refactor and demonstrable correctness for the report):

`tests/test_data_validation.py`:

1. **Investor — happy path:** clean 100-row synthetic df → 100 in, 100 out, no drops, no warnings.
2. **Investor — schema missing column:** raises ValueError with column name.
3. **Investor — NaN in feature:** that row is dropped, reason logged.
4. **Investor — invalid categorical:** dropped, reason logged.
5. **Investor — numeric out of range:** dropped, reason logged.
6. **Investor — consistency violation (`ExperienceYears > Age - 18`):** dropped, reason logged.
7. **ETF — happy path:** all rows survive, imputed counts zero, floors pass.
8. **ETF — missing Fund Symbol:** row dropped.
9. **ETF — NaN AUM with sufficient class samples:** imputed to per-class median, count incremented.
10. **ETF — negative volatility:** treated as bad, imputed.
11. **ETF — class floor failure:** raises ValueError when alternative has 0 rows.

No tests for the markdown formatter — manual inspection is sufficient.

## Out of scope (YAGNI)

- Outlier detection beyond range checks (synthetic data is bounded by `data_generation.py`).
- Type coercion (pandas `read_csv` handles the known column types correctly).
- Validation of `data/training/` artifacts (we produce them; no external consumer).
- Fixing the silent `(0.0, 0.0)` fallback in `extract_allocation` at [prepare_data.py:99-107](../../../src/prepare_data.py#L99-L107) — flagged as a separate small follow-up task.
- ETF validation of sparse columns (`geo_*`, `sector_weightings_*`, `expense_ratio`, `asset_classes_*`) — they are not used downstream today.
- CI integration / pre-commit hooks.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Per-asset-class median computed on tiny classes (e.g. 4 alternatives) is unstable | Fall back to global default when class has < 3 valid samples |
| ETF behavior subtly changes due to imputed values differing from current `1e9 / 3 / 0.15` defaults | When global fallback is used, the value is identical to current code. When per-class median is used, the change is logged in the imputation detail. The user can audit the report to confirm impact. |
| Asset-class floor failure blocks training when it used to silently produce a degenerate model | This is desired behavior — fail loud is better than train silently bad. Workaround documented: widen `get_category_mapping` keywords if alternative is empty. |
| Adding a new feature column in `data_generation.py` requires updating `INVESTOR_FEATURE_COLUMNS` in `validation_rules.py` | Schema check raises with the missing column name — drift is immediately visible. |

## File-by-file change summary

| File | Change | Lines (approx) |
|---|---|---|
| `src/validation_rules.py` | NEW — constants | 40 |
| `src/data_validation.py` | NEW — 2 functions + dataclass | 250 |
| `src/inference.py` | Replace inline rules with imports | -20 / +5 |
| `src/prepare_data.py` | Insert Step 1.5 | +15 |
| `tests/test_data_validation.py` | NEW — 11 tests | 200 |

Total new code: ~500 lines including tests. Net change to existing code: ~20 lines.
