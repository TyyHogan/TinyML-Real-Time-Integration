from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import tensorflow as tf


DATA_DIR = Path("processedData")
MODELS_DIR = Path("models")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Post-training int8 TFLite quantization (representative dataset from train.npz)."
    )
    p.add_argument(
        "--h5",
        required=True,
        help="Path to Keras .h5 (e.g. models/my_run.h5)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output .tflite path (default: same stem as .h5 with _int8 suffix)",
    )
    p.add_argument(
        "--calibration-samples",
        type=int,
        default=500,
        help="Number of train windows to use for calibration (default 500)",
    )
    return p.parse_args()


def load_train_flat() -> np.ndarray:
    data = np.load(DATA_DIR / "train.npz")
    x = data["X"].astype(np.float32)
    return x.reshape((x.shape[0], -1))


def load_test_flat() -> tuple[np.ndarray, np.ndarray]:
    data = np.load(DATA_DIR / "test.npz")
    x = data["X"].astype(np.float32)
    y = data["y"].astype(np.int32)
    return x.reshape((x.shape[0], -1)), y


def quantize_input(x: np.ndarray, input_details: dict) -> np.ndarray:
    """Convert float32 batch to quantized dtype expected by the interpreter."""
    q = input_details["quantization_parameters"]
    scales = q["scales"]
    zps = q["zero_points"]
    dtype = input_details["dtype"]
    if scales is None or len(scales) == 0 or scales[0] == 0:
        return x.astype(dtype)
    scale = float(scales[0])
    zp = int(zps[0])
    qx = np.round(x / scale + zp).astype(np.float32)
    return np.clip(qx, np.iinfo(dtype).min, np.iinfo(dtype).max).astype(dtype)


def dequantize_output(y: np.ndarray, output_details: dict) -> np.ndarray:
    q = output_details["quantization_parameters"]
    scales = q["scales"]
    zps = q["zero_points"]
    if scales is None or len(scales) == 0 or scales[0] == 0:
        return y.astype(np.float32)
    scale = float(scales[0])
    zp = int(zps[0])
    return (y.astype(np.float32) - zp) * scale


def tflite_accuracy(tflite_bytes: bytes, x_test: np.ndarray, y_test: np.ndarray) -> float:
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    correct = 0
    for i in range(len(x_test)):
        xb = np.expand_dims(x_test[i], axis=0)
        x_in = quantize_input(xb, inp)
        interpreter.set_tensor(inp["index"], x_in)
        interpreter.invoke()
        yq = interpreter.get_tensor(out["index"])
        probs = dequantize_output(yq, out).flatten()
        if np.argmax(probs) == y_test[i]:
            correct += 1
    return correct / len(y_test)


def main() -> int:
    args = parse_args()
    h5_path = Path(args.h5)
    if not h5_path.is_file():
        raise SystemExit(f"Missing Keras model: {h5_path}")

    if args.out:
        out_path = Path(args.out)
    else:
        stem = h5_path.stem
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", stem)
        out_path = MODELS_DIR / f"{safe}_int8.tflite"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    x_train = load_train_flat()
    x_test, y_test = load_test_flat()

    n = min(args.calibration_samples, len(x_train))
    if n < 100:
        print("Warning: use at least ~100 calibration samples for stable int8.")

    model = tf.keras.models.load_model(h5_path, compile=False)

    def representative_dataset():
        rng = np.random.default_rng(42)
        idx = rng.choice(len(x_train), size=n, replace=False)
        for i in idx:
            yield [np.expand_dims(x_train[i], axis=0)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_int8 = converter.convert()
    out_path.write_bytes(tflite_int8)

    model_size_bytes = len(tflite_int8)
    print(f"Saved int8 TFLite: {out_path} ({model_size_bytes} bytes)")

    acc = tflite_accuracy(tflite_int8, x_test, y_test)
    print(f"Int8 TFLite test accuracy (same test.npz): {acc:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
