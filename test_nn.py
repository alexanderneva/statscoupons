import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from data_loader import load_data


def create_model(input_dim, architecture):
    model = keras.Sequential([layers.Input(shape=(input_dim,))])

    for units in architecture:
        model.add(layers.Dense(units, activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))

    model.add(layers.Dense(2, activation="softmax"))
    return model


def train_and_evaluate(architecture, class_weights, epochs=50):
    print(f"\n{'=' * 60}")
    print(f"Architecture: {architecture}")
    print(f"Class weights: {class_weights}")
    print(f"{'=' * 60}")

    model = create_model(X_train_scaled.shape[1], architecture)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        X_train_scaled,
        y_train_cat,
        epochs=epochs,
        batch_size=256,
        validation_split=0.1,
        class_weight=class_weights,
        verbose=0,
    )

    y_pred_proba = model.predict(X_test_scaled, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(
        f"Results: Acc={acc:.4f}, AUC={auc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}"
    )
    return acc, auc, prec, rec, f1


print("Loading data...")
X, y = load_data()

sample_size = 20000
X_sample, _, y_sample, _ = train_test_split(
    X, y, train_size=sample_size, random_state=42, stratify=y
)

X_train, X_test, y_train, y_test = train_test_split(
    X_sample, y_sample, test_size=0.2, random_state=42, stratify=y_sample
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_train_cat = keras.utils.to_categorical(y_train.values, num_classes=2)
y_test_cat = keras.utils.to_categorical(y_test.values, num_classes=2)

architectures = [
    [128, 64, 32],  # 3 layers (original)
    [128, 64, 32, 16],  # 4 layers
    [128, 64, 32, 16, 8],  # 5 layers (current)
    [256, 128, 64],  # wider 3 layers
    [256, 128, 64, 32],  # wider 4 layers
    [64, 32],  # simpler 2 layers
    [128, 128, 64, 64],  # double width per layer
]

weight_configs = [
    ({0: 1.0, 1: 2.0}, "unbalanced (1:2)"),
    ({0: 1.0, 1: 1.0}, "balanced (1:1)"),
    ({0: 1.0, 1: 1.5}, "slight (1:1.5)"),
]

results = []

for arch in architectures:
    for weights, weight_name in weight_configs:
        acc, auc, prec, rec, f1 = train_and_evaluate(arch, weights)
        results.append(
            {
                "architecture": str(arch),
                "weights": weight_name,
                "acc": acc,
                "auc": auc,
                "prec": prec,
                "rec": rec,
                "f1": f1,
            }
        )

print("\n" + "=" * 80)
print("SUMMARY - Sorted by Accuracy")
print("=" * 80)
sorted_results = sorted(results, key=lambda x: x["acc"], reverse=True)
for r in sorted_results[:10]:
    print(
        f"Acc:{r['acc']:.4f} AUC:{r['auc']:.4f} {r['weights']:20s} {r['architecture']}"
    )

print("\n" + "=" * 80)
print("SUMMARY - Sorted by AUC")
print("=" * 80)
sorted_results = sorted(results, key=lambda x: x["auc"], reverse=True)
for r in sorted_results[:10]:
    print(
        f"Acc:{r['acc']:.4f} AUC:{r['auc']:.4f} {r['weights']:20s} {r['architecture']}"
    )
