from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf


DATA_DIR = Path("processedData")
OUT_DIR = Path("models")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_split(name: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(DATA_DIR / f"{name}.npz")
    x = data["X"].astype(np.float32)
    y = data["y"].astype(np.int32)
    return x, y


def main() -> int:
    x_train, y_train = load_split("train")
    x_val, y_val = load_split("val")
    x_test, y_test = load_split("test")

    # Tiny baseline: flatten each 50x7 window to 350 features.
    x_train = x_train.reshape((x_train.shape[0], -1))
    x_val = x_val.reshape((x_val.shape[0], -1))
    x_test = x_test.reshape((x_test.shape[0], -1))

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(x_train.shape[1],)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(3, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=10,
        restore_best_weights=True,
    )

    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=2,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    keras_path = OUT_DIR / "gesture_model.h5"
    model.save(keras_path)
    print(f"Saved Keras model: {keras_path}")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_path = OUT_DIR / "gesture_model.tflite"
    tflite_path.write_bytes(tflite_model)
    print(f"Saved TFLite model: {tflite_path} ({len(tflite_model)} bytes)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

