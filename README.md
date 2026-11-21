# MSCS 634 – Project Deliverable 2: Regression Modeling and Performance Evaluation

**Student:** Ruchitha Musku
**Course project:** Advanced Data Mining for Data-Driven Insights and Predictive Modeling
**Dataset:** IBM Telco Customer Churn (cleaned in Deliverable 1)

---

## Repository contents

| File / folder | Description |
|---|---|
| `MSCS_634_ProjectDeliverable_2.ipynb` | Main notebook: feature engineering, five regression models, cross-validation, evaluation, visualisations and interpretation |
| `data/telco_churn_cleaned.csv` | Cleaned dataset from Deliverable 1 (7,043 rows) |
| `model_comparison_results.csv` | Cross-validation and test metrics for every TotalCharges model |
| `tenure_model_results.csv` | Metrics for the tenure-prediction extension |
| `figures/` | All visualisations exported as PNG files |
| `requirements.txt` | Python packages needed to run the notebook |

**To run:** `pip install -r requirements.txt`, then open the notebook and choose *Run All* (about 3 minutes).

---

## Dataset summary

The IBM Telco Customer Churn dataset contains **7,043 customers** with demographics, account details (tenure, contract, billing,
payment method), nine service subscriptions, monthly and total charges, and churn status. Deliverable 1 cleaned it: blank
`TotalCharges` values were imputed, categories were standardised, and no duplicates or outliers were found.

## Modelling process

**Outcome variable: `TotalCharges`**, the total revenue each customer has generated (mean \$2,280, range \$0–\$8,685).
Predicting customer revenue supports customer-value estimation, revenue forecasting and billing-anomaly detection.

**Leakage prevention:** `MonthlyCharges` was excluded because TotalCharges ≈ tenure × MonthlyCharges, so including it would
nearly hand the answer to the model. Churn status was also excluded. The models must learn what each service costs from the service columns.

**Feature engineering (21 baseline → 51 engineered features):**
- **`tenure × service` interactions** for every service, contract, billing and payment indicator. Revenue is months × price, so these
  terms turn a multiplicative relationship into an additive one that a linear model can fit.
- Summary features: `num_addons`, `num_services`, `auto_payment`, `has_family`, `contract_months` and `tenure_sq`.
- Features derived from charges were deliberately not created, to avoid leakage.

**Models:**
1. Linear Regression on the baseline features (reference)
2. Multiple Linear Regression on the engineered features (isolates the effect of feature engineering)
3. Ridge Regression (L2), with α tuned by `RidgeCV`
4. Lasso Regression (L1), with α tuned by `LassoCV`
5. Random Forest on the baseline features (non-linear benchmark)

**Evaluation:** an 80/20 train/test split; **10-fold cross-validation** on the training set, with scaling inside pipelines and nested CV
for the tuned α; metrics R², MSE, RMSE and MAE; validation curves for α and learning curves.

---

## Evaluation results

| Model | CV R² | CV RMSE | Test R² | Test RMSE | Test MAE |
|---|---|---|---|---|---|
| **Lasso (engineered)** – best | **0.9987** | **\$81.66** | **0.9986** | **\$85.00** | **\$54.20** |
| Linear (engineered) | 0.9987 | \$81.85 | 0.9986 | \$85.32 | \$54.53 |
| Ridge (engineered) | 0.9987 | \$81.85 | 0.9986 | \$85.32 | \$54.54 |
| Random Forest (baseline) | 0.9956 | \$149.49 | 0.9953 | \$155.72 | \$94.67 |
| Linear (baseline) | 0.9068 | \$689.86 | 0.9043 | \$705.44 | \$571.20 |

Selected hyperparameters: Ridge α = 0.373, Lasso α = 0.869.

**Extension: predicting tenure** (a behavioural target; `TotalCharges` and `Churn` excluded): the best test R² was **0.670** (Ridge, RMSE 14.4 months).
The baseline linear model reached 0.663 and the Random Forest 0.646.

---

## Key insights and observations

1. **Feature engineering made the biggest difference.** Using the same OLS algorithm, engineered features cut test RMSE from \$705 to \$85
   (**−87.9%**) and raised R² from 0.904 to 0.999. The baseline model's U-shaped residuals (including negative revenue predictions for new
   customers) disappeared.
2. **Lasso is the best and recommended model.** Its accuracy is essentially tied with Ridge and OLS, but it is far simpler: it removed **34 of 51**
   features, and **7 tenure-based pricing features carry 99.6%** of the coefficient weight.
3. **Lasso confirmed the engineering choices.** Four add-ons that each cost about \$5/month were folded into the single engineered feature
   `tenure_x_num_services`, the largest coefficient in the model.
4. **The model recovers the company's price list** from revenue alone: fiber optic adds about \$25/month over DSL, phone service about \$20, each
   streaming service about \$10, and each support/backup add-on about \$5. A side check showed that `MonthlyCharges` is almost exactly additive in
   services (CV R² = 0.9988).
5. **Domain-driven features beat model complexity.** The Random Forest learned interactions on its own and beat the baseline, but its
   error was still about 1.8× that of the engineered linear models, and it is harder to interpret.
6. **The models generalise well.** CV and test scores agree closely, fold-to-fold variation is tiny (R² std = 0.0001), and the training and
   validation curves overlap (no overfitting). The baseline model's learning curve shows high bias that more data would not fix.
7. **Demographics do not drive revenue.** Lasso removed every demographic dummy.
8. **Behaviour is harder to model than billing.** Tenure prediction reaches only R² ≈ 0.67, which motivates the classification and clustering
   work in Deliverable 3.

---

## Challenges and how they were addressed

| Challenge | Solution |
|---|---|
| **Choosing a meaningful target.** `MonthlyCharges` turned out to be almost perfectly explained by the services (R² = 0.999), leaving nothing for feature engineering or model comparison to show. | Chose `TotalCharges` (revenue), which has a non-trivial multiplicative structure, and reported the MonthlyCharges finding as an insight. Added the tenure extension as a harder, behavioural test. |
| **Target leakage.** `MonthlyCharges` (and any feature derived from charges) would nearly give the model the answer. | Excluded it and documented why. Features were built only from non-charge predictors. |
| **The multiplicative relationship** (tenure × price) that a linear model cannot fit. | Engineered tenure × service interaction features based on domain knowledge. |
| **Multicollinearity.** `num_addons` and `num_services` are exact sums of the service dummies, making OLS coefficients non-unique. | Used Ridge/Lasso for stable, interpretable coefficients; a small-α ridge was used for the price-list table. |
| **Fair hyperparameter tuning** without optimistic bias. | `RidgeCV`/`LassoCV` inside pipelines evaluated by outer 10-fold CV (nested CV); scaling fit only on the training folds; the test set was used once. |
| **Lasso convergence warnings** with many correlated features. | Increased `max_iter` to 50,000 and standardised all features. |

---

## References
- IBM. *Telco Customer Churn* sample dataset. https://github.com/IBM/telco-customer-churn-on-icp4d
- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830. https://scikit-learn.org
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning* (2nd ed.), Ch. 3 & 6. Springer.
