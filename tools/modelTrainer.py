from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
import re

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score


DATA_DIR = Path("processedData")
OUT_DIR = Path("models")
OUT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR = Path("metrics")
METRICS_DIR.mkdir(parents=True, exist_ok=True)
RUNS_CSV = METRICS_DIR / "runs.csv"
CLASS_NAMES = ["idle", "wave", "shake"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train tiny baseline model and log metrics."
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run label for metrics/runs.csv (e.g. dense_v1_try3).",
    )
    return parser.parse_args()


def load_split(name: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(DATA_DIR / f"{name}.npz")
    x = data["X"].astype(np.float32)
    y = data["y"].astype(np.int32)
    return x, y


def main() -> int:
    args = parse_args()
    run_name = args.run_name or f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    safe_run_name = re.sub(r"[^A-Za-z0-9_.-]", "_", run_name)

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

    y_prob = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print("Confusion matrix:")
    print(cm)
    print(
        "Per-class F1:",
        ", ".join([f"{name}={report[name]['f1-score']:.4f}" for name in CLASS_NAMES]),
    )
    print(f"Macro F1: {macro_f1:.4f}")

    keras_path = OUT_DIR / f"{safe_run_name}.h5"
    model.save(keras_path)
    print(f"Saved Keras model: {keras_path}")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_path = OUT_DIR / f"{safe_run_name}.tflite"
    tflite_path.write_bytes(tflite_model)
    model_size_bytes = len(tflite_model)
    print(f"Saved TFLite model: {tflite_path} ({model_size_bytes} bytes)")

    run_record = {
        "run_name": run_name,
        "timestamp_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "test_accuracy": f"{test_acc:.6f}",
        "test_loss": f"{test_loss:.6f}",
        "macro_f1": f"{macro_f1:.6f}",
        "f1_idle": f"{report['idle']['f1-score']:.6f}",
        "f1_wave": f"{report['wave']['f1-score']:.6f}",
        "f1_shake": f"{report['shake']['f1-score']:.6f}",
        "model_size_bytes": str(model_size_bytes),
    }
    write_header = not RUNS_CSV.exists()
    with RUNS_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(run_record.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(run_record)
    print(f"Appended run metrics ({run_name}): {RUNS_CSV}")

    cm_path = METRICS_DIR / "confusion_matrix.json"
    cm_path.write_text(json.dumps(cm.tolist(), indent=2), encoding="utf-8")
    print(f"Wrote confusion matrix: {cm_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

