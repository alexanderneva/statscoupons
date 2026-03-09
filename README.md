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

We used [uv](https://docs.astral.sh/uv/) for this project. It defaulted to `python` version 3.12. The dependecies can be found in `pyproject.toml`.

## Data Sources

| Dataset | Records | Description |
|---------|---------|-------------|
| Wallet Transactions | 490,942 | Customer purchase transactions |
| Coupon Usage | 75,676 | Coupons actually used by customers |
| Coupon Distribution | 211,712 | Coupons sent to customers |

## Features

- **Original**: Transaction amounts, customer IDs, station prefix, time features
- **Engineered**: Benefit ratios, discount ratios, savings percentages
- **Aggregated**: Coupon usage counts and amounts per customer (from merged data)
- **Optimized**: Reduced from 23 to 15 features based on importance analysis
- **Key finding**: Replaced station_code (18K values) with station_prefix (19 values) for better generalization

### Feature Selection

Top features by importance:
1. hour (15.4%)
2. tran_to_receivable_ratio (10.5%)
3. tran_amt (10.5%)
4. receivable_amt (10.2%)
5. day_of_week (9.3%)
6. attribtuionorgcode (7.5%)

Removed: point_amt, point_to_discount_ratio (zero importance)
Encoded: station_code to station_prefix_encoded

### Features Used

- ['station_prefix_encoded', 'attributionorgcode', 'hour', 'tran_amt', 'receivable_amt', 'tran_to_receivable_ratio', 'day_of_week', 'total_coupon_send_amt', 'coupon_used_count', 'transactionorgcode', 'total_coupon_used_amt', 'coupon_send_count', 'savings_pct', 'benefit_ratio', 'discount_ratio']

## Model Results (20,000 sample)

| Model | Accuracy | AUC | Precision | Recall | F1 Score |
|-------|----------|-----|-----------|--------|----------|
| Random Forest | 74.20% | 0.826 | 64.66% | 64.75% | 64.71% |
| Gradient Boosting | 73.78% | 0.818 | 64.27% | 63.52% | 63.89% |
| Neural Network | 71.85% | 0.787 | 62.29% | 58.11% | 60.13% |
| SVM | 69.83% | 0.749 | 61.10% | 47.84% | 53.67% |
| Logistic Regression | 62.15% | 0.687 | 48.64% | 65.09% | 55.68% |

## Optimizations Applied

- Class weight balancing
- Hyperparameter tuning
- Feature importance-based selection

## Usage

```bash
# Train the model
uv run train_model.py

# Run neural netowrk
uv run train_nn.py

# Generate visualizations (LDA, t-SNE, feature importance)
uv run visualize.py

# Generate PCA visualization (excluding time features)
uv run visualize_pca.py

# Generate confusion matrix plots
uv run visualize_confusion_matrices.py

# Tune model
uv run tune_model.py

# Employ Grid search
uv run grid_search.py

# Run subset selection
uv run subset_selection.py
```

## Regarding the Neural Network

- layers can be added or taken away by editing the `neural_network.py` file in the `train_nn()` function. `train_nn.py` calls on this function
- GPU enabled for tensorflow

## Output Files

- `dimensionality_reduction.png` - Density comparison and t-SNE visualizations
- `pca_visualization.png` - PCA visualization (transaction/behavioral features only)
- `feature_importance.png` - Random Forest feature importance chart
- `correlation_matrix.png` - Feature correlation heatmap
- `confusion_matrices.png` - Confusion matrices for all models
- `presentation.tex` - Beamer presentation (compile with pdflatex)

## Under Construction

- `run_workflow.py` to streamline model changes and presentation updates
- `grid_search*` to do hyperparameter tuning. Takes time to run.

## References

Academic papers on O2O coupon prediction using XGBoost/LightGBM report AUC values of 0.95-0.99 with extensive feature engineering (26+ domain-specific features):

- [Forecast of O2O Coupon Consumption Based on XGBoost Model](https://francis-press.com/uploads/papers/mStT9Od4Whk7vxM7mDPWAIOfiOWNyg6rOYcwkyxG.pdf)
- [MDBIF: A Multi-Dimensional Feature and Boosting Integration Framework](https://www.clausiuspress.com/assets/default/article/2025/10/15/article_1760503641.pdf)
- [A Novel Digital Coupon Use Prediction Model Based on XGBoost](https://www.atlantis-press.com/proceedings/icmesd-18/25898711)
