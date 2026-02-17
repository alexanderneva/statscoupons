import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def create_model(input_dim):
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(16, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(8, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(2, activation="softmax"),
        ]
    )
    return model


def train_nn(X_train, y_train, X_test, y_test, epochs=100, batch_size=256):
    y_train_cat = keras.utils.to_categorical(y_train.values, num_classes=2)
    y_test_cat = keras.utils.to_categorical(y_test.values, num_classes=2)

    class_weights = {0: 1.0, 1: 2.0}

    model = create_model(X_train.shape[1])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        X_train,
        y_train_cat,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        class_weight=class_weights,
        verbose=1,
    )

    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    return y_pred, y_pred_proba[:, 1]
