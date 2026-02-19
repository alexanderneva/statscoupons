import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from data_loader import load_data


def main():
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

    # Random Forest
    print("\n" + "=" * 50)
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=15,
        min_samples_leaf=3,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )
    rf_cv = cross_val_score(rf_model, X_train, y_train, cv=kfold, scoring="accuracy")
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    y_proba_rf = rf_model.predict_proba(X_test)[:, 1]
    acc_rf = accuracy_score(y_test, y_pred_rf)
    prec_rf = precision_score(y_test, y_pred_rf)
    rec_rf = recall_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf)
    auc_rf = roc_auc_score(y_test, y_proba_rf)
    print(f"Accuracy: {acc_rf:.4f}, AUC: {auc_rf:.4f}, CV: {rf_cv.mean():.4f}")

    # SVM
    print("\n" + "=" * 50)
    print("Training SVM...")
    svm_model = SVC(
        kernel="rbf", C=1.0, gamma="scale", random_state=42, probability=True
    )
    svm_cv = cross_val_score(
        svm_model, X_train_scaled, y_train, cv=kfold, scoring="accuracy"
    )
    svm_model.fit(X_train_scaled, y_train)
    y_pred_svm = svm_model.predict(X_test_scaled)
    y_proba_svm = svm_model.predict_proba(X_test_scaled)[:, 1]
    acc_svm = accuracy_score(y_test, y_pred_svm)
    prec_svm = precision_score(y_test, y_pred_svm)
    rec_svm = recall_score(y_test, y_pred_svm)
    f1_svm = f1_score(y_test, y_pred_svm)
    auc_svm = roc_auc_score(y_test, y_proba_svm)
    print(f"Accuracy: {acc_svm:.4f}, AUC: {auc_svm:.4f}, CV: {svm_cv.mean():.4f}")

    # Gradient Boosting
    print("\n" + "=" * 50)
    print("Training Gradient Boosting...")
    gb_model = HistGradientBoostingClassifier(
        max_iter=100, max_depth=10, learning_rate=0.1, random_state=42
    )
    gb_cv = cross_val_score(gb_model, X_train, y_train, cv=kfold, scoring="accuracy")
    gb_model.fit(X_train, y_train)
    y_pred_gb = gb_model.predict(X_test)
    y_proba_gb = gb_model.predict_proba(X_test)[:, 1]
    acc_gb = accuracy_score(y_test, y_pred_gb)
    prec_gb = precision_score(y_test, y_pred_gb)
    rec_gb = recall_score(y_test, y_pred_gb)
    f1_gb = f1_score(y_test, y_pred_gb)
    auc_gb = roc_auc_score(y_test, y_proba_gb)
    print(f"Accuracy: {acc_gb:.4f}, AUC: {auc_gb:.4f}, CV: {gb_cv.mean():.4f}")

    # Logistic Regression
    print("\n" + "=" * 50)
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(
        max_iter=1000, random_state=42, solver="lbfgs", class_weight="balanced"
    )
    lr_cv = cross_val_score(lr_model, X_train, y_train, cv=kfold, scoring="roc_auc")
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    y_proba_lr = lr_model.predict_proba(X_test)[:, 1]
    auc_lr = roc_auc_score(y_test, y_proba_lr)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    prec_lr = precision_score(y_test, y_pred_lr)
    rec_lr = recall_score(y_test, y_pred_lr)
    f1_lr = f1_score(y_test, y_pred_lr)
    print(f"Accuracy: {acc_lr:.4f}, AUC: {auc_lr:.4f}")

    # Summary
    print("\n" + "=" * 50)
    print("Model Comparison")
    print("=" * 50)
    print(f"{'Model':<20} {'Acc':<8} {'AUC':<8} {'Prec':<8} {'Recall':<8} {'F1':<8}")
    print("-" * 60)
    print(
        f"{'Random Forest':<20} {acc_rf:.4f}  {auc_rf:.4f}  {prec_rf:.4f}  {rec_rf:.4f}  {f1_rf:.4f}"
    )
    print(
        f"{'SVM':<20} {acc_svm:.4f}  {auc_svm:.4f}  {prec_svm:.4f}  {rec_svm:.4f}  {f1_svm:.4f}"
    )
    print(
        f"{'Gradient Boosting':<20} {acc_gb:.4f}  {auc_gb:.4f}  {prec_gb:.4f}  {rec_gb:.4f}  {f1_gb:.4f}"
    )
    print(
        f"{'Logistic Reg':<20} {acc_lr:.4f}  {auc_lr:.4f}  {prec_lr:.4f}  {rec_lr:.4f}  {f1_lr:.4f}"
    )


if __name__ == "__main__":
    main()
