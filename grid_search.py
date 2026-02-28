"""Grid search for hyperparameter tuning."""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
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

    # Random Forest Grid Search
    print("\n" + "=" * 50)
    print("Grid Search: Random Forest")
    print("=" * 50)

    rf_params = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, 30],
        "min_samples_split": [5, 15],
        "min_samples_leaf": [1, 3],
    }

    rf = RandomForestClassifier(max_features="sqrt", n_jobs=-1, random_state=42)
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

    # Gradient Boosting Grid Search
    print("\n" + "=" * 50)
    print("Grid Search: Gradient Boosting")
    print("=" * 50)

    gb_params = {
        "max_iter": [100, 200],
        "max_depth": [5, 10, 15],
        "learning_rate": [0.05, 0.1, 0.2],
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

    # SVM Grid Search (quick)
    print("\n" + "=" * 50)
    print("Grid Search: SVM")
    print("=" * 50)

    svm_params = {
        "C": [0.1, 1.0, 10],
        "gamma": ["scale", "auto"],
    }

    svm = SVC(kernel="rbf", random_state=42, probability=True)
    svm_grid = GridSearchCV(
        svm, svm_params, cv=kfold, scoring="accuracy", n_jobs=-1, verbose=1
    )
    svm_grid.fit(X_train_scaled, y_train)

    print(f"Best SVM params: {svm_grid.best_params_}")
    print(f"Best SVM CV score: {svm_grid.best_score_:.4f}")

    svm_model = svm_grid.best_estimator_
    y_pred_svm = svm_model.predict(X_test_scaled)
    y_proba_svm = svm_model.predict_proba(X_test_scaled)[:, 1]
    acc_svm = accuracy_score(y_test, y_pred_svm)
    auc_svm = roc_auc_score(y_test, y_proba_svm)
    print(f"SVM Test - Accuracy: {acc_svm:.4f}, AUC: {auc_svm:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("Grid Search Results Summary")
    print("=" * 60)
    print(f"{'Model':<25} {'Acc':<10} {'AUC':<10}")
    print("-" * 60)
    print(f"{'Random Forest (Grid)':<25} {acc_rf:.4f}     {auc_rf:.4f}")
    print(f"{'Gradient Boosting (Grid)':<25} {acc_gb:.4f}     {auc_gb:.4f}")
    print(f"{'SVM (Grid)':<25} {acc_svm:.4f}     {auc_svm:.4f}")


if __name__ == "__main__":
    main()
