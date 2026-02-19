import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_loader import load_data


def analyze_feature_sets(X, y, sample_size=30000):
    print("=" * 60)
    print("BEST SUBSET SELECTION ANALYSIS")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    feature_names = X.columns.tolist()

    original_features = [
        "station_code",
        "tran_amt",
        "receivable_amt",
        "discounts_amt",
        "point_amt",
        "attributionorgcode",
        "transactionorgcode",
        "hour",
        "day_of_week",
        "is_weekend",
        "is_morning",
        "is_afternoon",
        "is_evening",
        "is_night",
        "coupon_used_count",
        "total_coupon_used_amt",
        "coupon_send_count",
        "total_coupon_send_amt",
    ]

    engineered_features = [
        "benefit_ratio",
        "discount_ratio",
        "savings_pct",
        "point_to_discount_ratio",
        "tran_to_receivable_ratio",
    ]

    original_available = [f for f in original_features if f in feature_names]
    engineered_available = [f for f in engineered_features if f in feature_names]

    print(f"\nOriginal features ({len(original_available)}): {original_available}")
    print(f"Engineered features ({len(engineered_available)}): {engineered_available}")

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

    results = []

    print("\n" + "=" * 60)
    print("TESTING DIFFERENT FEATURE SETS")
    print("=" * 60)

    sets_to_test = {
        "All Features": original_available + engineered_available,
        "Original Only": original_available,
        "Engineered Only": engineered_available,
        "Top 10 by Importance": None,
    }

    print("\n1. All Features:")
    X_all = X_train[original_available + engineered_available]
    model = RandomForestClassifier(**rf_params)
    scores = cross_val_score(model, X_all, y_train, cv=kfold, scoring="accuracy")
    print(f"   CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
    results.append(("All Features", scores.mean()))

    print("\n2. Original Features Only:")
    X_orig = X_train[original_available]
    model = RandomForestClassifier(**rf_params)
    scores = cross_val_score(model, X_orig, y_train, cv=kfold, scoring="accuracy")
    print(f"   CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
    results.append(("Original Only", scores.mean()))

    print("\n3. Engineered Features Only:")
    X_eng = X_train[engineered_available]
    model = RandomForestClassifier(**rf_params)
    scores = cross_val_score(model, X_eng, y_train, cv=kfold, scoring="accuracy")
    print(f"   CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
    results.append(("Engineered Only", scores.mean()))

    print("\n" + "=" * 60)
    print("FINDING TOP FEATURES BY IMPORTANCE")
    print("=" * 60)

    model.fit(X_train[original_available + engineered_available], y_train)
    importance = sorted(
        zip(original_available + engineered_available, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )

    print("\nFeature Importance Ranking:")
    for i, (feat, imp) in enumerate(importance, 1):
        print(f"  {i:2d}. {feat}: {imp:.4f}")

    top_features = [f for f, _ in importance]

    print("\n" + "=" * 60)
    print("TESTING TOP-K FEATURES")
    print("=" * 60)

    for k in [5, 8, 10, 12, 15, 18, 20]:
        top_k = top_features[:k]
        X_topk = X_train[top_k]
        model = RandomForestClassifier(**rf_params)
        scores = cross_val_score(model, X_topk, y_train, cv=kfold, scoring="accuracy")
        print(f"Top {k:2d} features: CV Accuracy = {scores.mean():.4f}")
        results.append((f"Top {k}", scores.mean()))

    print("\n" + "=" * 60)
    print("TESTING ORIGINAL + TOP ENGINEERED")
    print("=" * 60)

    top_eng = [f for f, _ in importance if f in engineered_available][:3]
    combo = original_available + top_eng
    X_combo = X_train[combo]
    model = RandomForestClassifier(**rf_params)
    scores = cross_val_score(model, X_combo, y_train, cv=kfold, scoring="accuracy")
    print(f"Original + Top 3 engineered: CV Accuracy = {scores.mean():.4f}")
    results.append(("Original + Top3 Eng", scores.mean()))

    print("\n" + "=" * 60)
    print("TESTING FEATURE GROUPS")
    print("=" * 60)

    groups = {
        "Transaction": ["tran_amt", "receivable_amt", "discounts_amt", "point_amt"],
        "Time": [
            "hour",
            "day_of_week",
            "is_weekend",
            "is_morning",
            "is_afternoon",
            "is_evening",
            "is_night",
        ],
        "Organization": ["station_code", "attributionorgcode", "transactionorgcode"],
        "Coupon History": [
            "coupon_used_count",
            "total_coupon_used_amt",
            "coupon_send_count",
            "total_coupon_send_amt",
        ],
        "Ratios": [
            "benefit_ratio",
            "discount_ratio",
            "savings_pct",
            "point_to_discount_ratio",
            "tran_to_receivable_ratio",
        ],
    }

    group_scores = {}
    for group_name, features in groups.items():
        available = [f for f in features if f in feature_names]
        if available:
            X_group = X_train[available]
            model = RandomForestClassifier(**rf_params)
            scores = cross_val_score(
                model, X_group, y_train, cv=kfold, scoring="accuracy"
            )
            print(f"{group_name}: CV Accuracy = {scores.mean():.4f}")
            group_scores[group_name] = scores.mean()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nFeature Set Comparison:")
    for name, score in sorted(results, key=lambda x: x[1], reverse=True):
        print(f"  {name}: {score:.4f}")

    print("\nBest performing individual feature groups:")
    for name, score in sorted(group_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {score:.4f}")

    print("\n" + "=" * 60)
    print("FINAL TEST SET EVALUATION")
    print("=" * 60)

    best_k = 15
    best_features = top_features[:best_k]

    X_train_best = X_train[best_features]
    X_test_best = X_test[best_features]

    model = RandomForestClassifier(**rf_params)
    model.fit(X_train_best, y_train)
    y_pred = model.predict(X_test_best)

    print(f"\nUsing top {best_k} features: {best_features}")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")

    return results, group_scores, importance


def main():
    print("Loading data...")
    X, y = load_data()

    sample_size = 30000
    if len(X) > sample_size:
        X_sample, _, y_sample, _ = train_test_split(
            X, y, train_size=sample_size, random_state=42, stratify=y
        )
        print(f"Using sample of {sample_size} records")
    else:
        X_sample, y_sample = X, y

    results, group_scores, importance = analyze_feature_sets(X_sample, y_sample)


if __name__ == "__main__":
    main()
