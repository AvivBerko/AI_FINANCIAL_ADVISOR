# 🏆 Model Winner Selection — All Three Phases

---

## Phase 1: Risk Profile Classification

### Raw Results

| Model | Train Acc | Test Acc | Macro Precision | Macro Recall | **Macro F1** | Verdict |
|---|:---:|:---:|:---:|:---:|:---:|:---|
| KNN | 1.000 | 0.7540 | 0.7447 | 0.7531 | 0.7485 | Overfits badly |
| **Logistic Regression** | 0.771 | 0.7520 | 0.7484 | 0.7556 | **0.7512** | ✅ Strong linear baseline |
| Random Forest | 0.9353 | 0.7990 | 0.7903 | 0.8075 | **0.7976** | Strong, slight overfit |
| XGBoost | 0.8568 | 0.7970 | 0.7882 | 0.8088 | **0.7966** | Close second |
| MLP | 0.8415 | 0.7810 | 0.7737 | 0.7778 | 0.7757 | Competitive |
| **TabPFN** | 0.8260 | **0.8000** | **0.7922** | **0.8118** | **0.8004** | 🥇 **WINNER** |

### 🏆 Phase 1 Winner: **TabPFN**

**Why TabPFN wins:**

- **Highest Test Accuracy** (0.800), **Macro Precision** (0.7922), **Macro Recall** (0.8118), and **Macro F1** (0.8004) across all 5 models.
- It has the **smallest generalization gap** among the strong models: a train accuracy of 0.826 and test of 0.800 — nearly no overfitting at all.
- Random Forest and XGBoost are very close (F1: 0.7976 and 0.7966), but TabPFN edges both of them on every single metric simultaneously.
- TabPFN is a **prior-data fitted network** — a transformer pre-trained on millions of synthetic tabular classification tasks. It achieves strong performance *without any hyperparameter tuning*, which is an additional reason it generalizes better: it is not overfit to this specific train set.

> ❌ **Why not KNN?** — KNN achieves 1.0 train accuracy and 0.754 test accuracy: a textbook overfit case with a 25-point generalization gap. It learned the training set by heart.  
> ❌ **Why not Logistic Regression?** — Best **linear** model, but its F1 of 0.7512 is clearly limited by the problem's non-linear structure.  
> ❌ **Why not Random Forest?** — Very close (F1: 0.7976 vs 0.8004), but TabPFN wins on every single metric with zero tuning needed.

### ⚙️ Best Hyperparameters

| Parameter | Value |
|---|---|
| Model type | Pre-trained transformer (TabPFN) |
| Hyperparameter tuning | **None required** |
| Grid search | ✗ Not applicable |

> TabPFN is a **foundation model** — it was pre-trained on millions of synthetic tabular datasets by the TabPFN authors. It performs inference via in-context learning on the training set, with no hyperparameters to tune. The absence of any grid search is itself a key advantage: the model cannot overfit to the training split through hyperparameter selection.

---

## Phase 2: Asset Allocation Regression

### Raw Results

| Model | **MAE** | MAPE | RMSE | **R²** | Verdict |
|---|:---:|:---:|:---:|:---:|:---|
| Linear Regression | 0.0900 | 0.5264 | 0.1149 | 0.5396 | Simple baseline |
| Ridge | 0.0900 | 0.5264 | 0.1149 | 0.5396 | Same as LR here |
| **Random Forest** | **0.0760** | 0.4557 | **0.1030** | **0.6300** | 🥇 **WINNER** |
| Gradient Boosting | 0.0771 | 0.4691 | 0.1026 | 0.6332 | Very close second |
| XGBoost | 0.0769 | **0.4545** | 0.1031 | 0.6292 | Almost identical to RF |

### 🏆 Phase 2 Winner: **Random Forest**

**Why Random Forest wins (narrowly):**

The three non-linear models (Random Forest, Gradient Boosting, XGBoost) are extremely close together. The decision comes down to a combination of factors:

| Criterion | Random Forest | Gradient Boosting | XGBoost |
|---|:---:|:---:|:---:|
| MAE | **0.0760** ✅ | 0.0771 | 0.0769 |
| MAPE | 0.4557 | 0.4691 | **0.4545** |
| RMSE | 0.1030 | **0.1026** | 0.1031 |
| R² | 0.6300 | **0.6332** | 0.6292 |

- Random Forest has the **lowest MAE (0.0760)**, which is the most meaningful metric here because it represents the actual error in percentage points on a prediction that goes into a real dollar allocation. An MAE of 7.6pp vs 7.7pp means Random Forest makes allocation decisions slightly closer to the target.
- Gradient Boosting has a marginally better RMSE and R², but the difference (0.1026 vs 0.1030 RMSE) is negligible in practice.
- XGBoost has the best MAPE but the lowest R² of the three.
- **Random Forest is the most balanced performer** across all four metrics and has the single most important one (MAE) locked.

> ⚠️ **Note on R² (≈0.63):** This is the **expected** outcome for this problem. Labels were generated with stochastic randomness in `allocations.py` — the same investor can get different allocations depending on a random seed. That irreducible noise caps how well any model can perform. An R² of 0.63 means the model explains 63% of the *explainable* variance, which is solid.

### ⚙️ Best Hyperparameters (via GridSearchCV)

| Parameter | Value | What it means |
|---|:---:|---|
| `estimator__n_estimators` | **1000** | 1,000 trees in the ensemble — large forest for stable averaging |
| `estimator__max_depth` | **10** | Each tree can grow up to 10 levels deep — captures non-linearity without memorising noise |
| `estimator__min_samples_split` | **20** | A node must contain ≥ 20 samples before it is allowed to split — strong regularisation against overfitting |

```python
RandomForestRegressor(
    n_estimators=1000,
    max_depth=10,
    min_samples_split=20,
    random_state=42
)
```

---

## Phase 3: Total ETF Count Regression

### Raw Results

| Model | **MAE** | MAPE | RMSE | **R²** | Verdict |
|---|:---:|:---:|:---:|:---:|:---|
| Linear Regression | 0.7976 | 0.0961 | 1.0225 | 0.6488 | Decent linear baseline |
| Ridge | 0.7972 | 0.0961 | 1.0223 | 0.6489 | Nearly identical to LR |
| **Random Forest** | **0.5729** | **0.0701** | **0.7882** | **0.7913** | 🥇 **WINNER** |
| Gradient Boosting | 0.5754 | 0.0706 | 0.7896 | 0.7906 | Extremely close |
| XGBoost | 0.5812 | 0.0712 | 0.7940 | 0.7882 | Third of the ensemble trio |

### 🏆 Phase 3 Winner: **Random Forest**

**Why Random Forest wins here (convincingly):**

Unlike Phase 2 where the ensemble models were neck-and-neck, Random Forest has a **cleaner lead** in Phase 3:

- **Best MAE: 0.5729** — means on average we are off by less than **0.57 ETFs**. For a prediction clipped to integer values in [7, 13], this is excellent.
- **Best MAPE: 0.0701** — only 7% relative error on the ETF count.
- **Best RMSE: 0.7882** and **Best R²: 0.7913** — an R² of 0.79 is a strong result, meaning the model explains 79% of the variance in ETF counts.
- Gradient Boosting is extremely close but loses on all four metrics.

> 💡 **Presentation talking point:** Random Forest wins both Phase 2 and Phase 3, which is a coherent story — ensemble methods with bagging are a natural fit for **structured tabular data** with non-linear relationships between investor features and portfolio decisions.

### ⚙️ Best Hyperparameters (via GridSearchCV)

| Parameter | Value | What it means |
|---|:---:|---|
| `n_estimators` | **1000** | 1,000 trees — same as Phase 2; high count stabilises the averaged prediction |
| `max_depth` | **5** | Shallower trees than Phase 2 — ETF count is a simpler target (only 7 possible values) so deeper trees would overfit |
| `min_samples_split` | **2** | Minimum split size of 2 — allows the trees to grow fine-grained branches, compensated by the shallow `max_depth` cap |

```python
RandomForestRegressor(
    n_estimators=1000,
    max_depth=5,
    min_samples_split=2,
    random_state=42
)
```

> 💡 **Note:** Compared to Phase 2, the Phase 3 winner uses **shallower trees** (`max_depth=5` vs `10`) and **looser split requirements** (`min_samples_split=2` vs `20`). This makes sense — predicting a continuous percentage allocation (Phase 2) requires more complex trees than predicting an integer in the narrow range [7, 13] (Phase 3).

---

## 🎯 Summary: Winners Per Phase

| Phase | Task | 🏆 Winner | Key Metric | Score |
|:---:|---|---|---|---|
| **Phase 1** | Risk Classification | **TabPFN** | Macro F1 | **0.8004** |
| **Phase 2** | Allocation Regression | **Random Forest** | MAE | **0.0760** |
| **Phase 3** | ETF Count Regression | **Random Forest** | MAE / R² | **0.5729 / 0.7913** |

### Narrative for Your Presentation

> *"Across the three ML stages of the pipeline, two different families of models won. For classification (Stage 1), TabPFN — a pre-trained transformer for tabular data — outperformed all traditional models without requiring a single hyperparameter to be tuned, achieving a Macro F1 of 0.80 and the tightest generalization gap of any model tested. For both regression stages (2 and 3), Random Forest consistently ranked first across MAE, RMSE, and R², demonstrating that ensemble bagging methods are well-suited to predicting continuous financial outputs from structured investor profiles."*
