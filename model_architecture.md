
# Machine Learning Model Architecture ("Phase 4")

This document outlines the architecture of the ETF Recommendation Engine. It is designed to answer **how**, **what**, and **when** the models operate, specifically highlighting the **parallel execution** flow.

## 1. High-Level Architecture Diagram

The system uses a **Parallel Ensemble** architecture. Unlike a sequential chain (where errors compound), three specialized models independently analyze the user's profile to determine different constraints of the portfolio.

```text
┌─────────────────────────────────────────────────────────────┐
│                    INPUT: Investor Profile                  │
│  (Age: 35, Income: $150k, Risk Tolerance: High, etc.)       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  PREPROCESSING (Inference)                  │
│  • One-Hot Encoding (Categorical Variables)                 │
│  • Standard Scaling (Numerical Variables)                   │
│  • Output: Feature Vector 'X' (1xN Array)                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
            ┌──────────────┼──────────────┐
            │              │              │
            ▼              ▼              ▼
┌──────────────────┐┌──────────────────┐┌──────────────────┐
│  MODEL A: RISK   ││MODEL B: ALLOC.   ││MODEL C: STRUCT.  │
│  (Classifier)    ││(Regressor)       ││(Regressor)       │
│                  ││                  ││                  │
│ Input: X         ││Input: X          ││Input: X          │
│ Task: Classify   ││Task: % Split     ││Task: How many?   │
│ Risk Category    ││Stocks vs Bonds   ││                  │
└────────┬─────────┘└────────┬─────────┘└────────┬─────────┘
         │                   │                   │
         │ (Parallel)        │ (Parallel)        │ (Parallel)
         ▼                   ▼                   ▼
┌──────────────────┐┌──────────────────┐┌──────────────────┐
│ PREDICTION 1     ││ PREDICTION 2     ││ PREDICTION 3     │
│ Risk Profile:    ││ Equity: 70%      ││ Counts Vector:   │
│ "Aggressive"     ││ Bond:   25%      ││ [5, 3, 2]        │
│                  ││        │         ││                  │
│                  ││        ▼         ││                  │
│                  ││ Alt = 100-(E+B)  ││                  │
│                  ││ Alt = 5%         ││                  │
└────────┬─────────┘└────────┬─────────┘└────────┬─────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│               SYNTHESIS (Rank & Select)                     │
│  1. Filter ETFs by Risk Profile (e.g., Growth for Aggr.)    │
│  2. Calculate Simple Score Vector:                          │
│     Weight = (AUM^1.5 * StarRating) / Volatility            │
│  3. Select Top N ETFs & Normalize Weights                   │
│  • Input: Predictions + Live Market Data                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  OUTPUT: Final Portfolio                    │
│  • Specific Tickers (e.g., VOO, BND, GLD)                   │
│  • Exact Dollar Amounts / Quantities                        │
└─────────────────────────────────────────────────────────────┘
```

## 2. Process Flow Description

### Step 1: Input & Transformation
*   **What:** Raw user data (e.g., Age: 30, Income: 100k, Goal: Growth).
*   **How:** Data is transformed into a numerical format suitable for ML. Categorical variables (like 'Gender') are one-hot encoded; numerical variables (like 'Income') are scaled.
*   **Result:** A single Feature Vector ($X$) representing the user.

### Step 2: Parallel Inference (The "Brain")
All three models run **simultaneously** and **independently**. They do not wait for each other.
*   **Model A (Risk):** "Given $X$, what is the investor's psychological risk category?"
*   **Model B (Allocation):** "Given $X$, what is the mathematical split between stocks and bonds?"
    *   *Note on Vectors:* This outputs a **Weight Vector** (e.g., `[0.60, 0.35]` for 60% Equity, 35% Bond). The remaining 5% is implicitly Alternative.
*   **Model C (Structure):** "Given $X$, how diversified should the portfolio be?" (e.g., 4 Equity ETFs, 2 Bond ETFs).

### Step 3: Synthesis (The "Assembler")
This rule-based layer combines the ML predictions with market reality.
1.  **Filter:** Uses **Model A** (Risk) to filter the universe of available ETFs (e.g., "Aggressive" users get Growth ETFs).
2.  **Select:** Uses **Model C** (Counts) to determine how many ETFs to pick from the filtered list (e.g., "Pick the top 4 Equity ETFs").
3.  **Rank:** Uses **Market Data** (Morningstar Ratings) to rank the candidates.
4.  **Allocate:** Uses **Model B** (Weights) to distribute capital across the selected ETFs.

## 3. Why Parallel?
*   **Robustness:** If the Risk model makes a mistake, it doesn't skew the Allocation numbers. The Allocation model looks at the *raw user data*, not the (potentially flawed) output of the Risk model.
*   **Speed:** Inference can be batched and parallelized.
*   **Explainability:** We can debug each constraint independently (e.g., "Why is the equity allocation so high?" -> Check Model B. "Why is the profile Conservative?" -> Check Model A).
