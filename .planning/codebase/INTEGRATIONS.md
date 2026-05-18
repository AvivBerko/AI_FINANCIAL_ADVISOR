# External Integrations

**Analysis Date:** 2026-05-18

## APIs & External Services

**Market Data — Historical:**
- yfinance — 10-year historical price history, fund info (AUM, PE, PB, dividendYield, fundFamily), geographic holdings, sector weightings, expense ratios
  - SDK/Client: `yfinance` (`import yfinance as yf`)
  - Auth: No API key required (unofficial Yahoo Finance scraper)
  - Usage: `src/etf_fetcher.py` — `YFinanceETFDataHandler.fetch_etf_data()`, `fetch_fund_operations()`, `fetch_geographic_allocation()`
  - Called for 95 ETF tickers during Step 1 of the training pipeline

**Market Data — Real-time Quotes:**
- Finnhub — Current price, previous close, day high/low, open
  - SDK/Client: `finnhub-python` (`import finnhub`)
  - Auth: `FINNHUB_API_KEY` env var (loaded from `.env`)
  - Usage: `src/etf_fetcher.py` — `FinnhubETFDataHandler.get_finnhub_daily_prices()`
  - Rate limit handled: `time.sleep(1)` between each ticker call
  - Aggregated with yfinance data in `ETFDataAggregator.fetch_complete_etf_data()`

**LLM — Chat & Tool-calling:**
- OpenAI `gpt-4o-mini` — Portfolio explanation, ongoing chat, tool-calling for portfolio adjustments
  - SDK/Client: `openai` (`from openai import OpenAI`)
  - Auth: `OPENAI_API_KEY` env var
  - Client instantiated at module level in `src/llm.py`: `client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))`
  - Two defined tools: `adjust_allocation` (weight deltas) and `update_investor_profile` (re-runs full ML pipeline)
  - Three call sites: `get_initial_explanation()` (max_tokens=500), `chat()` (max_tokens=600, tool_choice="auto"), `get_tool_followup()` (max_tokens=400)

## Data Storage

**Databases:**
- None — no SQL, NoSQL, or ORM detected

**File Storage (local CSV):**
- `data/raw/combined_etf_data.csv` — Raw ETF data from yfinance + Finnhub (gitignored)
- `data/raw/combined_etf_with_morning_star.csv` — ETF data enriched with Morningstar-style star ratings (gitignored)
- `data/processed/synthetic_investor_data_*.csv` — 5,000 synthetic investor profiles (gitignored)
- `data/training/X_train.csv`, `X_test.csv`, `y_train_*.csv`, `y_test_*.csv` — Training/test splits (gitignored)
- `data/training/preprocessor.pkl` — Fitted sklearn ColumnTransformer (gitignored)
- `data/validation_reports/etf_<ts>.md`, `investor_<ts>.md` — Per-run validation markdown reports (gitignored)

**Model Artifacts (local .pkl):**
- `models/risk_profile_classifier.pkl` — Best Stage 1 classifier (gitignored)
- `models/risk_profile_rf.pkl` — RandomForest Stage 1 model always saved for feature importance (gitignored)
- `models/allocation_regressor.pkl` — Best Stage 2 multi-output regressor (gitignored)
- `models/etf_count_regressor.pkl` — Stage 3 RandomForest regressor (gitignored)
- `models/label_map.json` — Records winning algorithm name and label encoding for Stage 1 (gitignored)

**Evaluation Artifacts:**
- `evaluation_reports/confusion_matrix_risk_profile.png` — Confusion matrix plot (gitignored)
- `evaluation_reports/feature_importance_risk_profile.png` — Feature importance bar chart (gitignored)
- `evaluation_reports/stage1_algorithm_comparison.png` — Algorithm comparison bar chart (gitignored)
- `evaluation_reports/allocation_predictions.png` — Scatter plots of actual vs predicted allocation (gitignored)
- `evaluation_reports/etf_count_distribution.png` — ETF count distribution histogram (gitignored)
- `evaluation_reports/model_metrics.json` — All metrics for all 3 stages (gitignored)

**Caching:**
- None — no Redis or in-memory cache; `src/inference.py` uses module-level `_ARTIFACTS` global to cache loaded model artifacts across Streamlit re-renders

## Authentication & Identity

**Auth Provider:**
- None — no user authentication system
- The Streamlit app has no login; all users share one session context per browser tab
- API keys are the only credentials; stored in `.env` (gitignored)

## Monitoring & Observability

**Error Tracking:**
- None — no Sentry, Datadog, or equivalent

**Logs:**
- `print()` statements throughout pipeline scripts; no structured logging framework
- Streamlit `st.error()` / `st.spinner()` for user-facing feedback in `app.py`

## CI/CD & Deployment

**Hosting:**
- Not configured — no Dockerfile, Procfile, `app.yaml`, or cloud deployment config found

**CI Pipeline:**
- None — no GitHub Actions, CircleCI, or equivalent config detected

## Environment Configuration

**Required env vars:**
- `FINNHUB_API_KEY` — Required for Step 1 ETF data fetch (`src/etf_fetcher.py`); without it, `main()` returns early
- `OPENAI_API_KEY` — Required for the LLM advisory layer (`src/llm.py`); `OpenAI()` client fails at import time if absent

**Secrets location:**
- `.env` in project root (gitignored via `.gitignore` entry for `.env` and `.env.local`)
- No secrets manager or vault integration

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None — all external calls are synchronous request/response (yfinance, Finnhub, OpenAI)

## Morningstar Ratings (Internal, Not External)

**Note:** Despite the name, `src/morning_star_rating.py` is a **custom internal implementation** — it computes a proprietary star rating (1–5) from ETF performance metrics (returns, volatility, drawdown, AUM). It does NOT call the Morningstar API. Output column `custom_star_rating` is appended to `data/raw/combined_etf_with_morning_star.csv`.

---

*Integration audit: 2026-05-18*
