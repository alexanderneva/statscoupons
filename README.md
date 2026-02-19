# Coupon Usage Prediction Model

Machine learning model to predict whether a customer will use a coupon in a transaction.

## Data Setup

The dataset is provided as a zip file. Extract it before running the code:

```bash
# Extract the dataset
unzip cust_coupon_detail_train.zip
```

This will extract the following CSV files:
- `cust_wallet_detail_train.csv` - Customer wallet transactions
- `cust_coupon_detail_used_train.csv` - Coupon usage records
- `cust_coupon_detail_send_train.csv` - Coupon distribution records

## Data Sources

| Dataset | Records | Description |
|---------|---------|-------------|
| Wallet Transactions | 490,942 | Customer purchase transactions |
| Coupon Usage | 75,676 | Coupons actually used by customers |
| Coupon Distribution | 211,712 | Coupons sent to customers |

## Features

- **Original**: Transaction amounts, customer IDs, store codes, time features
- **Engineered**: Benefit ratios, discount ratios, savings percentages
- **Aggregated**: Coupon usage counts and amounts per customer (from merged data)
- **Optimized**: Reduced from 23 to 15 features based on importance analysis

### Feature Selection

Top features by importance:
1. station_code (23.5%)
2. attributionorgcode (9.4%)
3. hour (7.5%)
4. tran_amt (7.0%)
5. receivable_amt (6.9%)
6. tran_to_receivable_ratio (6.7%)

Removed: point_amt, point_to_discount_ratio (zero importance)

## Model Results (20,000 sample)

| Model | Accuracy | AUC | Precision | Recall | F1 Score |
|-------|----------|-----|-----------|--------|----------|
| Random Forest | 74.20% | 0.826 | 64.66% | 64.75% | 64.71% |
| Gradient Boosting | 73.78% | 0.818 | 64.27% | 63.52% | 63.89% |
| Neural Network | 71.85% | 0.787 | 62.29% | 58.11% | 60.13% |
| SVM | 69.83% | 0.749 | 61.10% | 47.84% | 53.67% |
| Logistic Regression | 62.15% | 0.687 | 48.64% | 65.09% | 55.68% |

## Optimizations Applied

- Best subset selection analysis
- Class weight balancing
- Hyperparameter tuning
- Feature importance-based selection

## Usage

```bash
# Train the model
uv run python train_model.py

# Generate visualizations (LDA, t-SNE, feature importance)
uv run python visualize.py

# Run tuned model
uv run python tune_model.py
```

## Output Files

- `dimensionality_reduction.png` - Density comparison and t-SNE visualizations
- `pca_visualization.png` - PCA visualization of the data
- `feature_importance.png` - Random Forest feature importance chart
- `correlation_matrix.png` - Feature correlation heatmap
- `presentation.tex` - Beamer presentation (compile with pdflatex)

## References

Academic papers on O2O coupon prediction using XGBoost/LightGBM report AUC values of 0.95-0.99 with extensive feature engineering (26+ domain-specific features).
