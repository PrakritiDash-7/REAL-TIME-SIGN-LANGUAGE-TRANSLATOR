"""Train a small TensorFlow classifier on noisy synthetic landmark data from prototypes."""

from __future__ import annotations

import os

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from config import MODEL_PATH, SIGN_CLASSES
from prototypes import PROTOTYPES


def _synthetic_dataset(samples_per_class: int = 900, noise: float = 0.12, seed: int = 42):
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []
    for yi, name in enumerate(SIGN_CLASSES):
        proto = PROTOTYPES[name].astype(np.float32)
        for _ in range(samples_per_class):
            x = proto + rng.normal(0.0, noise, size=proto.shape).astype(np.float32)
            X_list.append(np.clip(x, -4.0, 4.0))
            y_list.append(yi)
    return np.stack(X_list), np.array(y_list, dtype=np.int32)


def build_and_save(epochs: int | None = None, verbose: int = 1) -> str:
    if epochs is None:
        epochs = int(os.environ.get("SIGN_TRAIN_EPOCHS", "28"))
    os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)
    X, y = _synthetic_dataset()
    n = len(SIGN_CLASSES)
    model = keras.Sequential(
        [
            layers.Input(shape=(63,)),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.25),
            layers.Dense(72, activation="relu"),
            layers.Dense(n, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(X, y, epochs=epochs, batch_size=64, validation_split=0.1, verbose=verbose)
    model.save(MODEL_PATH)
    return MODEL_PATH


if __name__ == "__main__":
    path = build_and_save()
    print("Saved:", path)
