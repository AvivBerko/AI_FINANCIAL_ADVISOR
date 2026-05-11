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
