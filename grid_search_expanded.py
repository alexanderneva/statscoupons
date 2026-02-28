"""Expanded Grid search for hyperparameter tuning."""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from data_loader import load_data


def main():
    print("Loading data...")
    X, y = load_data()

    sample_size = 20000
    if len(X) > sample_size:
        X_sample, _, y_sample, _ = train_test_split(
            X, y, train_size=sample_size, random_state=42, stratify=y
        )
    else:
        X_sample, y_sample = X, y

    X_train, X_test, y_train, y_test = train_test_split(
        X_sample, y_sample, test_size=0.2, random_state=42, stratify=y_sample
    )

    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Random Forest Grid Search (expanded)
    print("\n" + "=" * 60)
    print("Grid Search: Random Forest (Expanded)")
    print("=" * 60)

    rf_params = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 15, 20, 25],
        "min_samples_split": [5, 10, 15],
        "min_samples_leaf": [1, 2, 3],
        "max_features": ["sqrt", "log2", 0.5],
    }

    rf = RandomForestClassifier(n_jobs=-1, random_state=42)
    rf_grid = GridSearchCV(
        rf, rf_params, cv=kfold, scoring="accuracy", n_jobs=-1, verbose=1
    )
    rf_grid.fit(X_train, y_train)

    print(f"Best RF params: {rf_grid.best_params_}")
    print(f"Best RF CV score: {rf_grid.best_score_:.4f}")

    rf_model = rf_grid.best_estimator_
    y_pred_rf = rf_model.predict(X_test)
    y_proba_rf = rf_model.predict_proba(X_test)[:, 1]
    acc_rf = accuracy_score(y_test, y_pred_rf)
    auc_rf = roc_auc_score(y_test, y_proba_rf)
    print(f"RF Test - Accuracy: {acc_rf:.4f}, AUC: {auc_rf:.4f}")

    # Gradient Boosting Grid Search (expanded)
    print("\n" + "=" * 60)
    print("Grid Search: Gradient Boosting (Expanded)")
    print("=" * 60)

    gb_params = {
        "max_iter": [100, 200, 300],
        "max_depth": [5, 10, 15, 20],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "min_samples_leaf": [5, 10, 20],
    }

    gb = HistGradientBoostingClassifier(random_state=42)
    gb_grid = GridSearchCV(
        gb, gb_params, cv=kfold, scoring="accuracy", n_jobs=-1, verbose=1
    )
    gb_grid.fit(X_train, y_train)

    print(f"Best GB params: {gb_grid.best_params_}")
    print(f"Best GB CV score: {gb_grid.best_score_:.4f}")

    gb_model = gb_grid.best_estimator_
    y_pred_gb = gb_model.predict(X_test)
    y_proba_gb = gb_model.predict_proba(X_test)[:, 1]
    acc_gb = accuracy_score(y_test, y_pred_gb)
    auc_gb = roc_auc_score(y_test, y_proba_gb)
    print(f"GB Test - Accuracy: {acc_gb:.4f}, AUC: {auc_gb:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("Expanded Grid Search Results Summary")
    print("=" * 60)
    print(f"{'Model':<25} {'Acc':<10} {'AUC':<10}")
    print("-" * 60)
    print(f"{'Random Forest (Grid)':<25} {acc_rf:.4f}     {auc_rf:.4f}")
    print(f"{'Gradient Boosting (Grid)':<25} {acc_gb:.4f}     {auc_gb:.4f}")


if __name__ == "__main__":
    main()
