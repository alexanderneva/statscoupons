"""Quick Grid search for hyperparameter tuning."""

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

    # Random Forest Grid Search
    print("\n" + "=" * 50)
    print("Grid Search: Random Forest")
    print("=" * 50)

    rf_params = {
        "n_estimators": [100, 200],
        "max_depth": [15, 20, 25],
        "min_samples_split": [5, 10],
        "min_samples_leaf": [1, 2],
    }

    rf = RandomForestClassifier(max_features="sqrt", n_jobs=-1, random_state=42)
    rf_grid = GridSearchCV(rf, rf_params, cv=kfold, scoring="accuracy", n_jobs=-1)
    rf_grid.fit(X_train, y_train)

    print(f"Best RF params: {rf_grid.best_params_}")
    rf_model = rf_grid.best_estimator_
    acc_rf = accuracy_score(y_test, rf_model.predict(X_test))
    auc_rf = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])
    print(f"RF: Accuracy: {acc_rf:.4f}, AUC: {auc_rf:.4f}")

    # Gradient Boosting Grid Search
    print("\n" + "=" * 50)
    print("Grid Search: Gradient Boosting")
    print("=" * 50)

    gb_params = {
        "max_iter": [100, 150],
        "max_depth": [8, 12, 15],
        "learning_rate": [0.05, 0.1],
    }

    gb = HistGradientBoostingClassifier(random_state=42)
    gb_grid = GridSearchCV(gb, gb_params, cv=kfold, scoring="accuracy", n_jobs=-1)
    gb_grid.fit(X_train, y_train)

    print(f"Best GB params: {gb_grid.best_params_}")
    gb_model = gb_grid.best_estimator_
    acc_gb = accuracy_score(y_test, gb_model.predict(X_test))
    auc_gb = roc_auc_score(y_test, gb_model.predict_proba(X_test)[:, 1])
    print(f"GB: Accuracy: {acc_gb:.4f}, AUC: {auc_gb:.4f}")

    # SVM with different kernels
    print("\n" + "=" * 50)
    print("Grid Search: SVM (RBF)")
    print("=" * 50)

    svm_params = {
        "C": [1, 10],
        "gamma": ["scale", "auto"],
    }

    svm = SVC(kernel="rbf", random_state=42, probability=True)
    svm_grid = GridSearchCV(svm, svm_params, cv=kfold, scoring="accuracy", n_jobs=-1)
    svm_grid.fit(X_train_scaled, y_train)

    print(f"Best SVM params: {svm_grid.best_params_}")
    svm_model = svm_grid.best_estimator_
    acc_svm = accuracy_score(y_test, svm_model.predict(X_test_scaled))
    auc_svm = roc_auc_score(y_test, svm_model.predict_proba(X_test_scaled)[:, 1])
    print(f"SVM: Accuracy: {acc_svm:.4f}, AUC: {auc_svm:.4f}")

    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"RF: {acc_rf:.4f} / {auc_rf:.4f}")
    print(f"GB: {acc_gb:.4f} / {auc_gb:.4f}")
    print(f"SVM: {acc_svm:.4f} / {auc_svm:.4f}")


if __name__ == "__main__":
    main()
