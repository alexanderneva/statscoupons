import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import StandardScaler

from data_loader import load_data
from neural_network import train_nn


def main():
    print("Loading data...")
    X, y = load_data()

    sample_size = 20000
    if len(X) > sample_size:
        X_sample, _, y_sample, _ = train_test_split(
            X, y, train_size=sample_size, random_state=42, stratify=y
        )
        print(f"\nUsing sample of {sample_size} records for faster training")
    else:
        X_sample, y_sample = X, y

    X_train, X_test, y_train, y_test = train_test_split(
        X_sample, y_sample, test_size=0.2, random_state=42, stratify=y_sample
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n" + "=" * 50)
    print("Training Neural Network (5 Hidden Layers)...")
    print("=" * 50)

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"Using GPU: {gpus[0]}")
    else:
        print("No GPU detected, using CPU")
        tf.config.set_visible_devices([], "GPU")
        tf.config.set_visible_devices(tf.config.list_physical_devices("CPU"), "CPU")

    y_pred_nn, y_pred_proba_nn = train_nn(
        X_train_scaled, y_train, X_test_scaled, y_test, epochs=50
    )

    print("\n=== Neural Network Results ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_nn):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_nn):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred_nn):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred_nn):.4f}")

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred_nn))

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred_nn))


if __name__ == "__main__":
    main()
