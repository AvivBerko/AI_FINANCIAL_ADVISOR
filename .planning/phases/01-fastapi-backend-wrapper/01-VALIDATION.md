---
phase: 1
slug: fastapi-backend-wrapper
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-19
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution. Derived from `01-RESEARCH.md` § Validation Architecture.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | `pytest` 9.0.3 (already installed) + `fastapi.testclient.TestClient` (uses installed `httpx` 0.28.1) |
| **Config file** | None — Wave 0 may add `pytest.ini` if needed (pytest auto-discovers `tests/test_*.py`) |
| **Quick run command** | `venv/bin/pytest tests/test_api_*.py -x -q` |
| **Full suite command** | `venv/bin/pytest -x -q` |
| **Estimated runtime** | ~10-30 seconds (assuming yfinance calls are mocked or fixtures pre-cached) |

---

## Sampling Rate

- **After every task commit:** Run `venv/bin/pytest tests/test_api_*.py -x -q` (API tests touched by the change)
- **After every plan wave:** Run `venv/bin/pytest -x -q` (entire suite including pre-existing `tests/test_data_validation.py`)
- **Before `/gsd:verify-work`:** Full suite green AND manual smoke of `python pipeline.py --skip-fetch --skip-training` AND `venv/bin/python src/inference.py`
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| TBD-W0-01 | W0 setup | 0 | (infra) | — | N/A | install | `venv/bin/pip install fastapi 'uvicorn[standard]'` | ❌ W0 | ⬜ pending |
| TBD-W0-02 | W0 setup | 0 | (infra) | — | N/A | fixtures | `venv/bin/pytest tests/conftest.py --collect-only -q` | ❌ W0 | ⬜ pending |
| TBD-01-01 | 01 main | 1 | BACKEND-01 | — | Server boots; 4 routes registered | smoke | `venv/bin/pytest tests/test_api_smoke.py -x` | ❌ W0 | ⬜ pending |
| TBD-02-01 | 02 recommend | 1 | BACKEND-02 | — | Valid 12-field payload → 200 + weights sum to 1.0 (1e-6) | endpoint contract | `venv/bin/pytest tests/test_api_recommend.py::test_recommend_returns_normalized_allocation -x` | ❌ W0 | ⬜ pending |
| TBD-02-02 | 02 recommend | 1 | BACKEND-02 | — | Invalid payload (missing field or bad enum) → 422 | endpoint contract | `venv/bin/pytest tests/test_api_recommend.py::test_recommend_rejects_invalid_payload -x` | ❌ W0 | ⬜ pending |
| TBD-02-03 | 02 recommend | 1 | BACKEND-02 | — | `seed=42` deterministic across two calls | endpoint contract | `venv/bin/pytest tests/test_api_recommend.py::test_recommend_with_seed_is_deterministic -x` | ❌ W0 | ⬜ pending |
| TBD-03-01 | 03 prices | 1 | BACKEND-03 | — | Known tickers → live prices returned | endpoint contract | `venv/bin/pytest tests/test_api_prices.py::test_prices_for_known_tickers -x` | ❌ W0 | ⬜ pending |
| TBD-03-02 | 03 prices | 1 | BACKEND-03 | — | Unknown ticker → per-ticker error field, batch still 200 | endpoint contract | `venv/bin/pytest tests/test_api_prices.py::test_prices_partial_failure -x` | ❌ W0 | ⬜ pending |
| TBD-04-01 | 04 boundary | 1 | BACKEND-04 | — | `yfinance` import contained to `api/services/` only | structural | `grep -RIl "import yfinance\\|from yfinance" api/ \| wc -l` returns 1 | (structural check) | ⬜ pending |
| TBD-05-01 | 05 explain | 1 | BACKEND-05 | — | `/api/explain` returns 3 named sections (risk_class, equity_bond_split, per_ticker) | endpoint contract | `venv/bin/pytest tests/test_api_explain.py::test_explain_has_three_sections -x` | ❌ W0 | ⬜ pending |
| TBD-05-02 | 05 explain | 1 | BACKEND-05 | — | risk_class section includes top feature contributions | endpoint contract | `venv/bin/pytest tests/test_api_explain.py::test_explain_includes_feature_contributions -x` | ❌ W0 | ⬜ pending |
| TBD-05-03 | 05 explain | 1 | BACKEND-05 | — | per_ticker section includes `aum_rank` + `asset_class` | endpoint contract | `venv/bin/pytest tests/test_api_explain.py::test_explain_per_ticker_metadata -x` | ❌ W0 | ⬜ pending |
| TBD-06-01 | 06 etfs | 1 | BACKEND-06 | — | `GET /api/etfs/SPY` → 200 + metadata object | endpoint contract | `venv/bin/pytest tests/test_api_etfs.py::test_etfs_known_ticker -x` | ❌ W0 | ⬜ pending |
| TBD-06-02 | 06 etfs | 1 | BACKEND-06 | — | `GET /api/etfs/ZZZZ` → 404 + locked error message | endpoint contract | `venv/bin/pytest tests/test_api_etfs.py::test_etfs_unknown_ticker_returns_404 -x` | ❌ W0 | ⬜ pending |
| TBD-ADD-01 | additivity | 2 | BACKEND-01..06 | — | `python pipeline.py --skip-fetch` exits 0 (additivity) | manual smoke | `python pipeline.py --skip-fetch` | (existing) | ⬜ pending |
| TBD-ADD-02 | additivity | 2 | BACKEND-01..06 | — | `python src/inference.py` exits 0 (additivity) | manual smoke | `venv/bin/python src/inference.py` | (existing) | ⬜ pending |

*Task IDs use `TBD-*` placeholders until the planner finalizes plan IDs; the planner MUST update this table after writing plans.*

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] Install `fastapi` + `uvicorn[standard]` and pin them in `requirements.txt`
- [ ] `tests/conftest.py` — shared fixtures (`client`, `sample_profile`, `mock_models`)
- [ ] `tests/test_api_smoke.py` — covers BACKEND-01
- [ ] `tests/test_api_recommend.py` — covers BACKEND-02 (3 tests)
- [ ] `tests/test_api_prices.py` — covers BACKEND-03 (2 tests; mock yfinance with `respx` or `unittest.mock.patch`)
- [ ] `tests/test_api_explain.py` — covers BACKEND-05 (3 tests)
- [ ] `tests/test_api_etfs.py` — covers BACKEND-06 (2 tests)

**Test fixture pattern for "models may be missing in CI":**

```python
# tests/conftest.py
import pytest
from pathlib import Path

ARTIFACTS_PRESENT = all([
    Path("models/risk_profile_classifier.pkl").exists(),
    Path("data/training/preprocessor.pkl").exists(),
    Path("data/raw/combined_etf_with_morning_star.csv").exists(),
])

@pytest.fixture(scope="session")
def client():
    if not ARTIFACTS_PRESENT:
        pytest.skip("ML artifacts missing — run `python pipeline.py` to regenerate")
    from fastapi.testclient import TestClient
    from api.main import app
    with TestClient(app) as c:
        yield c

@pytest.fixture
def sample_profile():
    return {
        "Age": 30, "Gender": "Male", "Education": "Bachelor",
        "MaritalStatus": "Single", "HouseholdSize": 1,
        "Income": 80000, "InvestmentGoal": "Retirement",
        "InvestmentHorizon": 30, "InvestmentCapital": 50000,
        "RiskTolerance": "Medium", "FinancialInvolvement": "Medium",
        "ExperienceYears": 3,
    }
```

This pattern means CI without artifacts gracefully skips integration tests; developer regenerates artifacts and reruns for full coverage.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Streamlit prototype still launches | BACKEND-01..06 (additivity) | Visual UI smoke; not in CI | `venv/bin/streamlit run app.py` — visual confirmation that the page loads |
| `/docs` Swagger UI usable from a browser | BACKEND-01 | Manual UX check; FastAPI auto-renders | `uvicorn api.main:app --port 8000` → open http://localhost:8000/docs |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
