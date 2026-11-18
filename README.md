# Data Directory

Source dataset: [Default of Credit Card Clients Dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset) (`UCI_Credit_Card.csv`, 30,000 rows, 25 columns).

## Files

- `UCI_Credit_Card.csv` (2.73 MB): Full raw dataset (30,000 observations).
- `development_set.csv` (2.81 MB): 80% stratified split (`24,000` observations) used for exploratory analysis, preprocessing, feature engineering, model training, and cross-validation.
- `reporting_set.csv` (719 KB): **20% stratified holdout reporting set (`6,000` observations)**. **Must remain untouched until the project's final evaluation phase.**

## Split Configuration

Generated reproducibly by `prepare_data_and_eda.py`:
- Method: `sklearn.model_selection.train_test_split`
- `test_size = 0.20`
- `random_state = 42`
- `stratify = default.payment.next.month` (preserving the 22.12% default rate across both splits)
