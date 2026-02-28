import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
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

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

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
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred_rf,
        ax=axes[0, 0],
        cmap="Blues",
        display_labels=["No Coupon", "Used Coupon"],
    )
    axes[0, 0].set_title("Random Forest")

    print("Training Gradient Boosting...")
    gb_model = HistGradientBoostingClassifier(
        max_iter=100, max_depth=10, learning_rate=0.1, random_state=42
    )
    gb_model.fit(X_train, y_train)
    y_pred_gb = gb_model.predict(X_test)
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred_gb,
        ax=axes[0, 1],
        cmap="Blues",
        display_labels=["No Coupon", "Used Coupon"],
    )
    axes[0, 1].set_title("Gradient Boosting")

    print("Training Neural Network...")
    nn_model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
    )
    nn_model.fit(X_train_scaled, y_train)
    y_pred_nn = nn_model.predict(X_test_scaled)
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred_nn,
        ax=axes[1, 0],
        cmap="Blues",
        display_labels=["No Coupon", "Used Coupon"],
    )
    axes[1, 0].set_title("Neural Network")

    print("Training SVM...")
    svm_model = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    y_pred_svm = svm_model.predict(X_test_scaled)
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred_svm,
        ax=axes[1, 1],
        cmap="Blues",
        display_labels=["No Coupon", "Used Coupon"],
    )
    axes[1, 1].set_title("SVM")

    plt.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("images/confusion_matrices.png", dpi=150, bbox_inches="tight")
    print("Saved confusion_matrices.png")


if __name__ == "__main__":
    main()
