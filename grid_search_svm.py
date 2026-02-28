"""Grid search for SVM with different kernels."""

import numpy as np
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

    # SVM Grid Search with multiple kernels
    print("\n" + "=" * 60)
    print("Grid Search: SVM (Multiple Kernels)")
    print("=" * 60)

    # RBF Kernel
    print("\n--- RBF Kernel ---")
    rbf_params = {
        "C": [0.1, 1],
        "gamma": [0.01, 0.1],
    }

    rbf = SVC(kernel="rbf", random_state=42, probability=True)
    rbf_grid = GridSearchCV(
        rbf, rbf_params, cv=kfold, scoring="accuracy", n_jobs=-1, verbose=1
    )
    rbf_grid.fit(X_train_scaled, y_train)

    print(f"Best RBF params: {rbf_grid.best_params_}")
    print(f"Best RBF CV score: {rbf_grid.best_score_:.4f}")

    rbf_model = rbf_grid.best_estimator_
    y_pred_rbf = rbf_model.predict(X_test_scaled)
    y_proba_rbf = rbf_model.predict_proba(X_test_scaled)[:, 1]
    acc_rbf = accuracy_score(y_test, y_pred_rbf)
    auc_rbf = roc_auc_score(y_test, y_proba_rbf)
    print(f"RBF Test - Accuracy: {acc_rbf:.4f}, AUC: {auc_rbf:.4f}")

    # Linear Kernel
    print("\n--- Linear Kernel ---")
    linear_params = {
        "C": [0.1, 1, 10],
    }

    linear = SVC(kernel="linear", random_state=42, probability=True)
    linear_grid = GridSearchCV(
        linear, linear_params, cv=kfold, scoring="accuracy", n_jobs=-1, verbose=1
    )
    linear_grid.fit(X_train_scaled, y_train)

    print(f"Best Linear params: {linear_grid.best_params_}")
    print(f"Best Linear CV score: {linear_grid.best_score_:.4f}")

    linear_model = linear_grid.best_estimator_
    y_pred_linear = linear_model.predict(X_test_scaled)
    y_proba_linear = linear_model.predict_proba(X_test_scaled)[:, 1]
    acc_linear = accuracy_score(y_test, y_pred_linear)
    auc_linear = roc_auc_score(y_test, y_proba_linear)
    print(f"Linear Test - Accuracy: {acc_linear:.4f}, AUC: {auc_linear:.4f}")

    # Polynomial Kernel
    print("\n--- Polynomial Kernel ---")
    poly_params = {
        "C": [0.1, 1, 10],
        "degree": [2, 3, 4],
        "gamma": ["scale", "auto"],
    }

    poly = SVC(kernel="poly", random_state=42, probability=True)
    poly_grid = GridSearchCV(
        poly, poly_params, cv=kfold, scoring="accuracy", n_jobs=-1, verbose=1
    )
    poly_grid.fit(X_train_scaled, y_train)

    print(f"Best Poly params: {poly_grid.best_params_}")
    print(f"Best Poly CV score: {poly_grid.best_score_:.4f}")

    poly_model = poly_grid.best_estimator_
    y_pred_poly = poly_model.predict(X_test_scaled)
    y_proba_poly = poly_model.predict_proba(X_test_scaled)[:, 1]
    acc_poly = accuracy_score(y_test, y_pred_poly)
    auc_poly = roc_auc_score(y_test, y_proba_poly)
    print(f"Poly Test - Accuracy: {acc_poly:.4f}, AUC: {auc_poly:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("SVM Grid Search Results Summary (All Kernels)")
    print("=" * 60)
    print(f"{'Kernel':<15} {'Best Params':<40} {'Acc':<10} {'AUC':<10}")
    print("-" * 60)
    print(
        f"{'RBF':<15} {str(rbf_grid.best_params_):<40} {acc_rbf:.4f}     {auc_rbf:.4f}"
    )
    print(
        f"{'Linear':<15} {str(linear_grid.best_params_):<40} {acc_linear:.4f}     {auc_linear:.4f}"
    )
    print(
        f"{'Poly':<15} {str(poly_grid.best_params_):<40} {acc_poly:.4f}     {auc_poly:.4f}"
    )


if __name__ == "__main__":
    main()
