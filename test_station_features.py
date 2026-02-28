import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from data_loader import load_data


def run_experiment(X, y, feature_names, name):
    sample_size = 20000
    X_sample, _, y_sample, _ = train_test_split(
        X, y, train_size=sample_size, random_state=42, stratify=y
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_sample, y_sample, test_size=0.2, random_state=42, stratify=y_sample
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=15,
        min_samples_leaf=3,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n{name}")
    print(f"Features ({len(feature_names)}): {feature_names}")
    print(
        f"Accuracy: {acc:.4f}, AUC: {auc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}"
    )

    return {
        "name": name,
        "n_features": len(feature_names),
        "accuracy": acc,
        "auc": auc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }


def main():
    print("Loading data...")
    X, y = load_data()

    current_features = X.columns.tolist()
    print(f"\nCurrent features ({len(current_features)}): {current_features}")

    results = []

    # Experiment 1: Full model (current)
    results.append(
        run_experiment(
            X.values, y.values, current_features, "Current (with station_code)"
        )
    )

    # Experiment 2: Without station_code
    features_no_station = [f for f in current_features if f != "station_code"]
    X_no_station = X[features_no_station]
    results.append(
        run_experiment(
            X_no_station.values, y.values, features_no_station, "Without station_code"
        )
    )

    # Experiment 3: Add station_prefix instead of station_code
    wallet_df = pd.read_csv("cust_wallet_detail_train.csv")
    wallet_df = wallet_df.iloc[X.index]
    X_with_prefix = X.copy()
    X_with_prefix["station_prefix"] = (
        wallet_df["station_code"].astype(str).str[:3].values
    )
    prefix_map = {p: i for i, p in enumerate(X_with_prefix["station_prefix"].unique())}
    X_with_prefix["station_prefix_encoded"] = X_with_prefix["station_prefix"].map(
        prefix_map
    )

    features_prefix = [f for f in current_features if f != "station_code"] + [
        "station_prefix_encoded"
    ]
    X_prefix = X_with_prefix[features_prefix]
    results.append(
        run_experiment(
            X_prefix.values,
            y.values,
            features_prefix,
            "With station_prefix instead of station_code",
        )
    )

    # Experiment 4: Just attributionorgcode (no station info)
    features_attr_only = [
        f
        for f in current_features
        if f not in ["station_code", "station_prefix_encoded"]
    ]
    X_attr = X_with_prefix[features_attr_only]
    results.append(
        run_experiment(
            X_attr.values,
            y.values,
            features_attr_only,
            "Without station_code or prefix",
        )
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Experiment':<40} {'Acc':<8} {'AUC':<8} {'Features'}")
    print("-" * 60)
    for r in results:
        print(
            f"{r['name']:<40} {r['accuracy']:.4f}   {r['auc']:.4f}   {r['n_features']}"
        )


if __name__ == "__main__":
    main()
