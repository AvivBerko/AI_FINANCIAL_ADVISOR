# 🎯 ETF Recommendation System - Complete Walkthrough

## 📖 Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Complete Workflow (Step-by-Step)](#complete-workflow-step-by-step)
4. [How to Showcase This Project](#how-to-showcase-this-project)
5. [File-by-File Breakdown](#file-by-file-breakdown)
6. [Troubleshooting](#troubleshooting)

---

## 🎯 Project Overview

### What Does This System Do?
This is a **Machine Learning-based Financial Planning System** that recommends personalized ETF portfolios based on an investor's profile.

### The Problem It Solves
- Traditional financial advisors are expensive and not accessible to everyone
- Robo-advisors use simple rule-based systems
- This system uses **ML to learn patterns** from investor profiles and recommend optimal portfolios

### The Approach
**Hybrid Model:** ML Models + Rule-Based Selection
1. **ML Models** predict: Risk Profile, Asset Allocation, Portfolio Size
2. **Rule-Based Logic** selects specific ETF tickers (avoiding conflicts like VOO + SPY)

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT: Investor Profile                   │
│  (Age, Income, Capital, Risk Tolerance, Goals, etc.)        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   PHASE 1: Data Generation                   │
│  • Generate 5,000 synthetic investors                        │
│  • Maintain realistic correlations (age ↔ income, etc.)     │
│  • Output: synthetic_investor_data.csv                       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  PHASE 2: Data Preparation                   │
│  • Load investor data + ETF data                             │
│  • Run rule-based allocation to create labels                │
│  • Extract features (X) and targets (y1, y2, y3)            │
│  • Output: X_train.csv, y_train_*.csv                        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   PHASE 3: Model Training                    │
│  • Train 3 models                              │
│  • Evaluate performance (accuracy, MAE, R²)                  │
│  • Output: *.pkl models in models/                           │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    PHASE 4: Inference                        │
│  • Load trained models                                       │
│  • Predict for new investor                                  │
│  • Select specific ETF tickers                               │
│  • Output: Complete portfolio recommendation                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Complete Workflow (Step-by-Step)

### **STEP 0: Prerequisites**

#### Install Dependencies
```bash
pip install -r requirements.txt
```

**What this installs:**
- `pandas`, `numpy` - Data manipulation
- `scikit-learn` - Machine learning
- `matplotlib`, `seaborn` - Visualization
- `yfinance`, `requests` - ETF data fetching
- `jupyter`, `notebook` - Interactive notebooks

---

### **STEP 1: Fetch ETF Data** ⏭️ (Optional - Already Done)

#### What It Does:
Fetches real-time ETF data from financial APIs (yFinance + Finnhub)

#### File: `src/etf_fetcher.py`

#### What It Fetches:
- 90+ popular ETF tickers (VOO, SPY, QQQ, BND, etc.)
- Metrics: Returns (1Y, 3Y, 5Y), Volatility, AUM, Expense Ratio, etc.
- Geographic allocations, sector weightings

#### How to Run:
```bash
python src/etf_fetcher.py
```

#### Output:
- `data/raw/combined_etf_data_YYYYMMDD_HHMMSS.csv`

#### ⚠️ Note:
**You already have this data!** File exists at:
- `data/raw/combined_etf_with_morning_star.csv`

**Skip this step unless you want fresh data.**

---

### **STEP 2: Generate Synthetic Investor Data** ✅ (Core Step)

#### What It Does:
Creates 5,000 realistic synthetic investors with correlated features

#### File: `src/data_generation.py`

#### How It Works:
1. **Generates 12 features** for each investor:
   - Demographics: Age, Gender, Education, Marital Status
   - Financial: Income, Investment Capital, Household Size
   - Investment: Goals, Time Horizon, Risk Tolerance
   - Experience: Financial Involvement, Experience Years

2. **Maintains Realistic Correlations:**
   - Young + High Income → Higher risk tolerance
   - Older + Low Income → Lower risk tolerance
   - Married + "Child education" goal → Larger household

3. **Balanced Risk Distribution:**
   - Uses score-based quantile binning
   - Ensures exactly 33% High / 33% Medium / 33% Low risk

#### How to Run:
```bash
python src/data_generation.py
```

#### Output:
- `synthetic_investor_data_YYYYMMDD_HHMMSS.csv` (5,000 rows)

#### What to Look For:
```
=== Risk Tolerance Distribution ===
High: 1667 (33.34%)
Medium: 1667 (33.34%)
Low: 1666 (33.32%)

=== Logical Consistency Verification ===
Young (<35) + High Income (>100k): XXX investors
  Risk Profile Distribution:
    High: XX.X%    ← Should be highest
    Medium: XX.X%
    Low: XX.X%     ← Should be lowest
```

#### ✅ Already Done:
You have: `data/processed/synthetic_investor_data_20260104184056.csv`

---

### **STEP 3: Prepare Training Data** ✅ (Core Step)

#### What It Does:
Transforms raw investor data into ML-ready training datasets

#### File: `notebooks/phase2_data_preparation.ipynb`

#### How It Works:

**Part 1: Load Data**
- Loads synthetic investor data (5,000 samples)
- Loads ETF data (90+ ETFs)

**Part 2: Generate Labels (Targets)**
- Runs rule-based allocation logic from `src/allocations.py`
- For each investor, calculates:
  - **y1:** Risk Profile (Aggressive/Moderate/Conservative)
  - **y2:** Asset Allocation (Equity %, Bond %, Alternative %)
  - **y3:** ETF Count (Number of ETFs per asset class)

**Part 3: Feature Engineering**
- Extracts features (X): Age, Income, Capital, etc.
- Encodes categorical variables (Education, Marital Status)
- Normalizes numerical features

**Part 4: Train/Test Split**
- 80% training, 20% testing
- Stratified split (maintains risk profile balance)

#### How to Run:
```bash
# Option 1: Open in Jupyter
jupyter notebook notebooks/phase2_data_preparation.ipynb

# Option 2: Run programmatically
jupyter nbconvert --execute --to notebook --inplace notebooks/phase2_data_preparation.ipynb
```

#### Output Files:
```
data/training/
├── X_train.csv          # Training features (4,000 rows)
├── X_test.csv           # Test features (1,000 rows)
├── y_train_risk.csv     # Risk profile labels
├── y_train_allocation.csv  # Allocation percentages
└── y_train_etf_count.csv   # ETF counts
```

#### Key Cells to Review:
1. **Cell: "Load Data"** - Verify data loaded correctly
2. **Cell: "Run Allocations"** - See rule-based logic in action
3. **Cell: "Feature Engineering"** - Check encoded features
4. **Cell: "Train/Test Split"** - Verify split ratios

---

### **STEP 4: Train ML Models** ✅ (Core Step)

#### What It Does:
Trains 3 Random Forest models to predict portfolio recommendations

#### File: `notebooks/phase3_model_training.ipynb`

#### The 3 Models:

**Model 1: Risk Profile Classifier**
- **Input:** Investor features (Age, Income, etc.)
- **Output:** Risk Profile (Aggressive/Moderate/Conservative)
- **Algorithm:** Random Forest Classifier
- **Metric:** Accuracy, Precision, Recall, F1-Score

**Model 2: Allocation Regressor**
- **Input:** Investor features + Predicted Risk Profile
- **Output:** Asset Allocation (Equity %, Bond %)
- **Algorithm:** Random Forest Regressor (Multi-output)
- **Metric:** MAE (Mean Absolute Error), R² Score

**Model 3: ETF Count Regressor**
- **Input:** Investor features + Predicted Allocation
- **Output:** Number of ETFs per asset class
- **Algorithm:** Random Forest Regressor
- **Metric:** MAE, Exact Match Accuracy

#### How to Run:
```bash
# Option 1: Open in Jupyter
jupyter notebook notebooks/phase3_model_training.ipynb

# Option 2: Run programmatically
jupyter nbconvert --execute --to notebook --inplace notebooks/phase3_model_training.ipynb
```

#### Output Files:
```
models/
├── risk_profile_classifier.pkl    # Model 1
├── allocation_regressor.pkl       # Model 2
├── etf_count_regressor.pkl        # Model 3
└── preprocessor.pkl               # Feature preprocessor

evaluation_reports/
├── model_metrics.json             # Performance metrics
├── confusion_matrix.png           # Risk profile confusion matrix
├── allocation_predictions.png     # Allocation scatter plots
└── feature_importance.png         # Feature importance charts
```

#### Current Performance (from `evaluation_reports/model_metrics.json`):
```json
{
  "model_1_risk_profile": {
    "test_accuracy": 0.46,           ← 46% accuracy
    "macro_f1": 0.458
  },
  "model_2_allocation": {
    "equity_test_mae": 0.149,        ← 14.9% error
    "bond_test_mae": 0.149
  },
  "model_3_total_etfs": {
    "test_mae": 1.055,               ← ~1 ETF error
    "exact_match_accuracy": 0.265
  }
}
```

#### ⚠️ Why Low Accuracy?
- **Synthetic data** - Not real investor behavior
- **Proof-of-concept** - Demonstrates workflow, not production accuracy
- **Expected** - Real data would improve performance significantly

#### Key Cells to Review:
1. **Cell: "Train Model 1"** - See training progress
2. **Cell: "Evaluate Model 1"** - Check confusion matrix
3. **Cell: "Feature Importance"** - See which features matter most
4. **Cell: "Save Models"** - Verify models saved to `models/`

---

### **STEP 5: Make Predictions** 🔮 (Showcase Step)

#### What It Does:
Uses trained models to recommend portfolios for new investors

#### File: `notebooks/phase4_etf_prediction.ipynb`

#### How It Works:

**Part 1: Load Models**
```python
import joblib

risk_classifier = joblib.load('models/risk_profile_classifier.pkl')
allocation_regressor = joblib.load('models/allocation_regressor.pkl')
etf_count_regressor = joblib.load('models/etf_count_regressor.pkl')
preprocessor = joblib.load('models/preprocessor.pkl')
```

**Part 2: Create Test Investor**
```python
new_investor = {
    'Age': 35,
    'Income': 75000,
    'InvestmentCapital': 50000,
    'RiskTolerance': 'Medium',
    'Education': 'Bachelor',
    'MaritalStatus': 'Married',
    'HouseholdSize': 2,
    'InvestmentGoal': 'Retirement',
    'InvestmentHorizon': 20,
    'FinancialInvolvement': 'Medium',
    'ExperienceYears': 5
}
```

**Part 3: Make Predictions**
```python
# Step 1: Preprocess
X_new = preprocessor.transform([new_investor])

# Step 2: Predict risk profile
risk_profile = risk_classifier.predict(X_new)[0]
# Output: "Moderate"

# Step 3: Predict allocation
allocation = allocation_regressor.predict(X_new)[0]
# Output: [0.70, 0.25]  → 70% Equity, 25% Bond

# Step 4: Predict ETF counts
etf_counts = etf_count_regressor.predict(X_new)[0]
# Output: [3, 2, 1]  → 3 Equity, 2 Bond, 1 Alternative
```

**Part 4: Select Specific ETFs**
```python
from src.allocations import assign_portfolio_numeric_aum

# Load ETF data
etf_df = pd.read_csv('data/raw/combined_etf_with_morning_star.csv')

# Select tickers
portfolio = assign_portfolio_numeric_aum(new_investor, etf_df)
# Output: {'Equity': ['VOO', 'VTI', 'QQQ'], 'Bond': ['BND', 'AGG'], ...}
```

#### How to Run:
```bash
jupyter notebook notebooks/phase4_etf_prediction.ipynb
```

#### Expected Output:
```
📊 Portfolio Recommendation for New Investor
═══════════════════════════════════════════

Investor Profile:
  Age: 35
  Income: $75,000
  Capital: $50,000
  Risk Tolerance: Medium

Predicted Risk Profile: Moderate

Asset Allocation:
  ├─ Equity:       70.0% ($35,000)
  ├─ Bond:         25.0% ($12,500)
  └─ Alternative:   5.0% ($2,500)

Recommended ETFs:
  Equity (3 ETFs):
    • VOO - Vanguard S&P 500 ETF
    • VTI - Vanguard Total Stock Market ETF
    • QQQ - Invesco QQQ Trust

  Bond (2 ETFs):
    • BND - Vanguard Total Bond Market ETF
    • AGG - iShares Core U.S. Aggregate Bond ETF

  Alternative (1 ETF):
    • VNQ - Vanguard Real Estate ETF
```

---

### **STEP 6: Run Full Pipeline** 🚀 (Automation)

#### What It Does:
Automates the entire workflow from data generation to model training

#### File: `pipeline.py`

#### How to Run:
```bash
# Full pipeline (all steps)
python pipeline.py

# Skip ETF data fetching (use cached data)
python pipeline.py --skip-fetch

# Skip model training (use existing models)
python pipeline.py --skip-training
```

#### What It Executes:
```
Step 1: Fetch ETF Data (or use cached)
Step 2: Generate Synthetic Investors
Step 3: Prepare Training Data (calls notebook)
Step 4: Train ML Models (calls notebook)
Step 5: Summary Report
```

#### ⚠️ Current Limitation:
Steps 3 & 4 require manual notebook execution. The pipeline prints instructions:
```
⚠️  This step requires running:
   notebooks/phase2_data_preparation.ipynb
```

---

## 🎤 How to Showcase This Project

### **Scenario 1: Quick Demo (5 minutes)**

**What to Show:**
1. **Open `phase4_etf_prediction.ipynb`**
2. **Run all cells** to show prediction for a sample investor
3. **Highlight the output:** Risk profile, allocation, specific ETFs

**Talking Points:**
- "This system uses ML to recommend personalized ETF portfolios"
- "It learns from 5,000 synthetic investors to predict optimal allocations"
- "The hybrid approach combines ML predictions with rule-based ETF selection"

---

### **Scenario 2: Technical Deep Dive (15 minutes)**

**What to Show:**

**Part 1: Data Generation (3 min)**
- Open `src/data_generation.py`
- Show correlation logic (lines 124-189: `calculate_risk_score`)
- Run: `python src/data_generation.py`
- Show balanced risk distribution output

**Part 2: Model Training (5 min)**
- Open `notebooks/phase3_model_training.ipynb`
- Show confusion matrix for risk classifier
- Show feature importance chart
- Explain why accuracy is low (synthetic data)

**Part 3: Prediction (5 min)**
- Open `notebooks/phase4_etf_prediction.ipynb`
- Create custom investor profile
- Run prediction cells
- Show final portfolio recommendation

**Part 4: Code Quality (2 min)**
- Show `src/allocations.py` conflict resolution logic (lines 141-172)
- Explain mutually exclusive ETF groups (VOO vs SPY)

---

### **Scenario 3: Full Walkthrough (30 minutes)**

**Follow this guide step-by-step:**
1. Explain project overview (5 min)
2. Show data generation (5 min)
3. Walk through phase2 notebook (7 min)
4. Walk through phase3 notebook (8 min)
5. Demo prediction (5 min)

---

## 📂 File-by-File Breakdown

### **Core Python Scripts**

| File | Purpose | When to Run | Output |
|------|---------|-------------|--------|
| `src/data_generation.py` | Generate synthetic investors | Once (or when you need fresh data) | `synthetic_investor_data_*.csv` |
| `src/etf_fetcher.py` | Fetch ETF data from APIs | Once (or monthly for updates) | `combined_etf_data_*.csv` |
| `src/allocations.py` | Rule-based portfolio logic | Called by notebooks (not standalone) | N/A (library) |
| `morning_star_rating.py` | Calculate ETF star ratings | Called by `etf_fetcher.py` | N/A (library) |
| `pipeline.py` | Orchestrate full workflow | When you want to automate | Runs all steps |

---

### **Jupyter Notebooks**

| Notebook | Purpose | Input | Output |
|----------|---------|-------|--------|
| `phase2_data_preparation.ipynb` | Prepare training data | Investor CSV + ETF CSV | `X_train.csv`, `y_train_*.csv` |
| `phase3_model_training.ipynb` | Train ML models | Training CSVs | `*.pkl` models, metrics |
| `phase4_etf_prediction.ipynb` | Make predictions | Trained models + new investor | Portfolio recommendation |

---

### **Data Files**

| Directory | Contents | Purpose |
|-----------|----------|---------|
| `data/raw/` | ETF data (90+ tickers) | Source data for ETF selection |
| `data/processed/` | Synthetic investor data (5,000 samples) | Training data source |
| `data/training/` | X_train, y_train splits | ML model inputs |
| `models/` | Trained .pkl models | Inference |
| `evaluation_reports/` | Metrics, charts | Model performance analysis |

---

## 🐛 Troubleshooting

### **Issue: "No module named 'src'"**
**Solution:**
```bash
# Make sure you're in the project root
cd c:\Users\avivi\Projects\ai_financial_advisor

# Run Python with module flag
python -m src.data_generation
```

---

### **Issue: "File not found: combined_etf_with_morning_star.csv"**
**Solution:**
```bash
# Run ETF fetcher to generate data
python src/etf_fetcher.py

# Or use existing data (check data/raw/)
```

---

### **Issue: "Models not found in models/"**
**Solution:**
```bash
# Run phase3 notebook to train models
jupyter notebook notebooks/phase3_model_training.ipynb
```

---

### **Issue: Notebook kernel crashes**
**Solution:**
```bash
# Reduce dataset size in data_generation.py
# Change line 9: N = 5000 → N = 1000
```

---

## 🎯 Key Takeaways for Showcasing

### **What Makes This Project Impressive:**

1. ✅ **End-to-End ML Pipeline** - Data generation → Training → Inference
2. ✅ **Hybrid Approach** - ML + Rule-based logic
3. ✅ **Production-Ready Structure** - Modular, documented, reproducible
4. ✅ **Realistic Data** - Correlated features, balanced classes
5. ✅ **Conflict Resolution** - Smart ETF selection (no duplicates)
6. ✅ **Comprehensive Evaluation** - Metrics, visualizations, reports

### **What to Emphasize:**

- **Problem-solving:** "Traditional robo-advisors use simple rules. This uses ML to learn patterns."
- **Technical depth:** "Implemented feature engineering, stratified splitting, multi-output regression."
- **Production mindset:** "Modular code, automated pipeline, saved models for deployment."

### **What to Acknowledge:**

- **Limitations:** "Low accuracy due to synthetic data. Real investor data would improve performance."
- **Future work:** "Next steps: REST API, web UI, backtesting framework."

---

## 📚 Next Steps

### **To Improve This Project:**

1. **Create `src/inference.py`** - Standalone prediction script
2. **Add unit tests** - pytest for each module
3. **Build REST API** - FastAPI endpoint for predictions
4. **Create web UI** - React/Vue frontend for investor input
5. **Add real data** - Replace synthetic data with historical investor portfolios
6. **Implement backtesting** - Test portfolio performance over time

---

## 🎓 Learning Resources

### **To Understand the Code Better:**

- **Random Forest:** [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/ensemble.html#forest)
- **Feature Engineering:** [Kaggle Learn](https://www.kaggle.com/learn/feature-engineering)
- **Portfolio Theory:** [Modern Portfolio Theory Basics](https://www.investopedia.com/terms/m/modernportfoliotheory.asp)

---

**Good luck with your showcase! 🚀**

If you have questions about any specific step, refer back to this guide or check the inline comments in each file.
