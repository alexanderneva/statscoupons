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
- **Engineered**: Benefit ratios, discount ratios, net amounts, savings percentages
- **Aggregated**: Coupon usage counts and amounts per customer (from merged data)

## Model Results (20,000 sample)

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Random Forest | 74.22% | 64.99% | 63.79% | 64.39% |
| Gradient Boosting | 73.62% | 63.98% | 63.59% | 63.78% |
| SVM | 69.13% | 60.72% | 43.81% | 50.89% |
| Neural Network | 63.00% | 49.64% | 90.14% | 64.03% |
| Logistic Regression | 64.18% | 50.73% | 66.87% | 57.69% |

## Sample Size Comparison

| Model | 20,000 Sample | 421,590 Sample | Improvement |
|-------|---------------|----------------|-------------|
| Random Forest | 74.22% | 75.60% | +1.38% |
| Gradient Boosting | 73.62% | 74.13% | +0.51% |

## Usage

```bash
# Train the model
uv run python train_model.py

# Generate visualizations (LDA, t-SNE, feature importance)
uv run python visualize.py
```

## Output Files

- `dimensionality_reduction.png` - Density comparison and t-SNE visualizations
- `feature_importance.png` - Random Forest feature importance chart
- `lda_tran_amt_discounts_amt.png` - LDA analysis of transaction vs station code
- `correlation_matrix.png` - Feature correlation heatmap
- `presentation.tex` - Beamer presentation (compile with pdflatex)
