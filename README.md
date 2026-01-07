# ETF Recommendation System

A machine learning-based financial planning system that recommends personalized ETF portfolios based on investor profiles.

## 🎯 Project Overview

This system uses **3 ML models** to predict:
1. **Risk Profile** (Aggressive/Moderate/Conservative)
2. **Asset Allocation** (Equity % / Bond %)
3. **Portfolio Size** (Number of ETFs)

Then uses **rule-based selection** to pick specific ETF tickers.

## 📁 Project Structure

```
etf_recommendation_sys/
├── data/                          # Generated datasets
│   ├── raw/                       # Raw ETF data from APIs
│   ├── processed/                 # Cleaned investor data
│   └── training/                  # Train/test splits
├── models/                        # Trained ML models (.pkl)
├── notebooks/                     # Jupyter notebooks
│   ├── phase2_data_preparation.ipynb
│   └── phase3_model_training.ipynb
├── src/                           # Source code
│   ├── data_generation.py         # Generate synthetic investors
│   ├── etf_fetcher.py             # Fetch ETF data from APIs
│   ├── allocations.py             # Rule-based allocation logic
│   └── inference.py               # ML model inference
├── evaluation_reports/            # Model metrics & plots
├── requirements.txt               # Python dependencies
├── pipeline.py                    # Main pipeline script
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Full Pipeline
```bash
python pipeline.py
```

This will:
- ✅ Fetch ETF data (or use cached)
- ✅ Generate 5,000 synthetic investors
- ✅ Prepare training data
- ✅ Train 3 ML models
- ✅ Save models & evaluation reports

### 3. Make Predictions
```python
from src.inference import predict_portfolio

investor = {
    'Age': 35,
    'Income': 75000,
    'InvestmentCapital': 50000,
    'RiskTolerance': 'Medium',
    # ... other features
}

portfolio = predict_portfolio(investor)
print(portfolio)  # {'risk_profile': 'Moderate', 'allocation': {...}, 'etfs': [...]}
```

## 📊 Model Performance

| Model | Metric | Score |
|-------|--------|-------|
| Risk Profile Classifier | Accuracy | ~46% |
| Allocation Regressor | MAE (Equity) | ~14.9% |
| Allocation Regressor | MAE (Bond) | ~14.9% |
| ETF Count Regressor | MAE | ~X.X ETFs |

*Note: Low accuracy is expected - this is a proof-of-concept with synthetic data*

## 🔄 Development Workflow

### Phase 1: Data Generation
```bash
python src/data_generation.py
```
- Generates 5,000 investors with balanced risk profiles (33/33/33)
- Maintains correlations (age ↔ income, capital ↔ horizon, etc.)

### Phase 2: Data Preparation
Run `notebooks/phase2_data_preparation.ipynb`
- Loads investor data
- Runs rule-based allocation to generate labels
- Extracts features (X) and targets (y1, y2, y3)
- Saves train/test splits

### Phase 3: Model Training
Run `notebooks/phase3_model_training.ipynb`
- Trains 3 Random Forest models
- Evaluates performance
- Saves models to `models/`

### Phase 4: Inference
```python
from src.inference import ETFRecommender

recommender = ETFRecommender()
portfolio = recommender.recommend(investor_data)
```

## 🛠️ Tech Stack

- **ML**: scikit-learn (Random Forest)
- **Data**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **APIs**: yfinance, requests (Morningstar)

## 📝 Key Files

| File | Purpose |
|------|---------|
| `pipeline.py` | End-to-end automation |
| `src/data_generation.py` | Synthetic investor generation |
| `src/allocations.py` | Rule-based allocation logic |
| `src/inference.py` | ML model inference |
| `notebooks/phase2_*.ipynb` | Data preparation |
| `notebooks/phase3_*.ipynb` | Model training |

## 🎓 Next Steps

- [ ] Improve model accuracy with real data
- [ ] Add ML-based ticker selection (Phase 4B)
- [ ] Build REST API for predictions
- [ ] Create web UI for investor input
- [ ] Add backtesting framework

## 📧 Contact

For questions or feedback, contact [Your Name]

---

**Last Updated**: January 2026
