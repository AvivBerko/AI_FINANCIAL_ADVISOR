╔═══════════════════════════════════════════════════════════════════════════════╗
║                          LAYER 1 — DATA ACQUISITION                           ║
║                    (run once, expensive due to external APIs)                 ║
╚═══════════════════════════════════════════════════════════════════════════════╝

   ┌──────────────────────────┐                ┌──────────────────────────────┐
   │ yfinance API             │                │ Finnhub API                  │
   │ (metadata, history)      │                │ (real-time quotes)           │
   └────────────┬─────────────┘                └──────────────┬───────────────┘
                │                                             │
                └──────────────┬──────────────────────────────┘
                               ▼
                  ┌──────────────────────────┐
                  │ src/etf_fetcher.py       │
                  │   95 hardcoded tickers   │
                  │   → ETF features         │
                  └────────────┬─────────────┘
                               ▼
              ┌────────────────────────────────────┐
              │ data/raw/combined_etf_data.csv     │
              └────────────────┬───────────────────┘
                               ▼
                  ┌──────────────────────────┐
                  │ src/morning_star_rating  │
                  │   .py (calc star ratings)│
                  └────────────┬─────────────┘
                               ▼
   ┌────────────────────────────────────────────────────┐
   │ data/raw/combined_etf_with_morning_star.csv        │   ◄── ETF SIDE
   └────────────────────────────────────────────────────┘

                  ┌──────────────────────────┐
                  │ src/data_generation.py   │
                  │   N=5000, seed=42        │
                  │   12 synthetic features  │
                  └────────────┬─────────────┘
                               ▼
   ┌────────────────────────────────────────────────────┐
   │ data/processed/synthetic_investor_data_<ts>.csv    │   ◄── INVESTOR SIDE
   └────────────────────────────────────────────────────┘


╔═══════════════════════════════════════════════════════════════════════════════╗
║                          LAYER 2 — TRAINING PIPELINE                          ║
║                    (orchestrated by pipeline.py, steps 3-4)                   ║
╚═══════════════════════════════════════════════════════════════════════════════╝

   ETF CSV  ───┐                                          ┌───  Investor CSV
               │                                          │
               ▼                                          ▼
            ┌──────────────────────────────────────────────────┐
            │  src/prepare_data.py                             │
            │  ──────────────────                              │
            │  1. Load both CSVs                               │
            │  2. Call run_allocations_pipeline()  ────────────┼──┐
            │  3. Extract 12 features → scale + onehot → 19    │  │
            │  4. Extract 3 targets:                           │  │
            │       y1 = risk_profile (Aggressive/Mod/Cons)    │  │
            │       y2 = (equity_pct, bond_pct)                │  │
            │       y3 = total_etfs                            │  │
            │  5. Train/test split 80/20, stratified on y1     │  │
            │  6. Save artifacts                               │  │
            └────────────────────┬─────────────────────────────┘  │
                                 │                                │
                                 │  uses                          │
                                 │                                │
                                 ▼                                ▼
                    ┌──────────────────────────────────────────────┐
                    │  src/allocations.py                          │
                    │  ───────────────────                         │
                    │  Rule-based label generator:                 │
                    │   • get_profile_probabilities()              │
                    │   • get_allocation_mix()                     │
                    │   • get_num_etfs()                           │
                    │   • assign_portfolio_numeric_aum() (Stage 4) │
                    │   • CONFLICT_GROUPS  (no VOO + SPY etc.)     │
                    └──────────────────────────────────────────────┘
                                 │
                                 ▼
            ┌──────────────────────────────────────────────────┐
            │  data/training/                                  │
            │    X_train.csv, X_test.csv                       │
            │    y_train_profile.csv, y_test_profile.csv       │
            │    y_train_allocation.csv, y_test_allocation.csv │
            │    y_train_etfs.csv, y_test_etfs.csv             │
            │    preprocessor.pkl                              │
            └────────────────────┬─────────────────────────────┘
                                 ▼
            ┌──────────────────────────────────────────────────┐
            │  src/train_models.py                             │
            │  ────────────────────                            │
            │  Trains 3 models:                                │
            │   Stage 1: classifier  (risk_profile)            │
            │   Stage 2: regressor   (allocation %)            │
            │   Stage 3: regressor   (ETF count)               │
            │  Evaluates, picks "winner" per stage             │
            └────────────────────┬─────────────────────────────┘
                                 ▼
            ┌──────────────────────────────────────────────────┐
            │  models/                                         │
            │    risk_profile_classifier.pkl                   │
            │    allocation_regressor.pkl                      │
            │    etf_count_regressor.pkl                       │
            │    label_map.json                                │
            │  evaluation_reports/  (metrics, plots)           │
            └──────────────────────────────────────────────────┘


╔═══════════════════════════════════════════════════════════════════════════════╗
║                          LAYER 3 — INFERENCE (LIVE)                           ║
║                       (runs per-user, hot path)                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝

   USER (natural language)               USER (structured form)
            │                                     │
            ▼                                     │
   ┌──────────────────────────┐                   │
   │  src/llm.py              │                   │
   │  OpenAI tool-calling     │                   │
   │  NL → structured dict    │                   │
   │  Also: adjust_allocation │                   │
   └────────────┬─────────────┘                   │
                │                                 │
                └─────────────┬───────────────────┘
                              ▼
                  ┌────────────────────────────┐
                  │  app.py  (Streamlit UI)    │
                  └────────────┬───────────────┘
                               │
                               ▼  calls recommend(user_dict)
            ┌──────────────────────────────────────────────────┐
            │  src/inference.py                                │
            │  ────────────────                                │
            │   ① validate_input(user)                         │
            │   ② preprocessor.transform  (12 → 19 features)   │
            │   ③ Stage 1: classifier → risk_profile           │
            │   ④ Stage 2: regressor → (equity_pct, bond_pct)  │
            │             + PROFILE_BOUNDS guardrails          │
            │   ⑤ Stage 3: regressor → total_etfs (clipped)    │
            │   ⑥ Stage 4: assign_portfolio_numeric_aum()      │
            │              from allocations.py                 │
            └────────────────────┬─────────────────────────────┘
                                 ▼
                ┌────────────────────────────────────┐
                │  Returned dict:                    │
                │    risk_profile                    │
                │    allocation (eq/bond/alt %)      │
                │    total_etfs                      │
                │    etf_counts                      │
                │    portfolio (specific tickers)    │
                └────────────────────────────────────┘