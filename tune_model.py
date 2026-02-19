import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from data_loader import load_data


def main():
    print("Loading data...")
    X, y = load_data()

    sample_size = 30000
    if len(X) > sample_size:
        X_sample, _, y_sample, _ = train_test_split(
            X, y, train_size=sample_size, random_state=42, stratify=y
        )
    else:
        X_sample, y_sample = X, y

    X_train, X_test, y_train, y_test = train_test_split(
        X_sample, y_sample, test_size=0.2, random_state=42, stratify=y_sample
    )

    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    rf_params = {
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": 0.5,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": 42,
    }

    hgb_params = {
        "max_iter": 300,
        "max_depth": 20,
        "learning_rate": 0.05,
        "min_samples_leaf": 5,
        "l2_regularization": 0.1,
        "random_state": 42,
    }

    print("\n=== Random Forest ===")
    rf = RandomForestClassifier(**rf_params)
    rf_scores = cross_val_score(rf, X_train, y_train, cv=kfold, scoring="accuracy")
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]
    print(f"CV: {rf_scores.mean():.4f}")
    print(
        f"Test - Acc: {accuracy_score(y_test, y_pred):.4f}, AUC: {roc_auc_score(y_test, y_proba):.4f}"
    )

    print("\n=== HistGradientBoosting ===")
    hgb = HistGradientBoostingClassifier(**hgb_params)
    hgb_scores = cross_val_score(hgb, X_train, y_train, cv=kfold, scoring="accuracy")
    hgb.fit(X_train, y_train)
    y_pred = hgb.predict(X_test)
    y_proba = hgb.predict_proba(X_test)[:, 1]
    print(f"CV: {hgb_scores.mean():.4f}")
    print(
        f"Test - Acc: {accuracy_score(y_test, y_pred):.4f}, AUC: {roc_auc_score(y_test, y_proba):.4f}"
    )


if __name__ == "__main__":
    main()
