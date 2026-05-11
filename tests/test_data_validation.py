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
