from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

FILE_RE = re.compile(r"^(idle|wave|shake)_(\d+)\.csv$", re.IGNORECASE)
FEATURE_COLS = ["ax", "ay", "az", "gx", "gy", "gz", "pot"]
LABEL_MAP = {"idle": 0, "wave": 1, "shake": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Day 8 preprocessing: normalize + windows + train/val/test split"
    )
    parser.add_argument("--data-dir", default="saveData", help="Directory containing CSV recordings")
    parser.add_argument("--out-dir", default="processed", help="Output directory for NPZ files")
    parser.add_argument("--window", type=int, default=50, help="Window size in samples")
    parser.add_argument("--stride", type=int, default=25, help="Window stride in samples")
    parser.add_argument("--pot-max", type=float, default=1024.0, help="Potentiometer max value")
    return parser.parse_args()


def normalize_features(df: pd.DataFrame, pot_max: float) -> np.ndarray:
    ax = np.clip(df["ax"].to_numpy(np.float32) / 16384.0, -1.0, 1.0)
    ay = np.clip(df["ay"].to_numpy(np.float32) / 16384.0, -1.0, 1.0)
    az = np.clip(df["az"].to_numpy(np.float32) / 16384.0, -1.0, 1.0)
    gx = np.clip(df["gx"].to_numpy(np.float32) / 32768.0, -1.0, 1.0)
    gy = np.clip(df["gy"].to_numpy(np.float32) / 32768.0, -1.0, 1.0)
    gz = np.clip(df["gz"].to_numpy(np.float32) / 32768.0, -1.0, 1.0)
    pot = np.clip(df["pot"].to_numpy(np.float32) / pot_max, 0.0, 1.0)
    return np.stack([ax, ay, az, gx, gy, gz, pot], axis=1)


def make_windows(features: np.ndarray, label_id: int, window: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    windows = []
    labels = []
    for start in range(0, len(features) - window + 1, stride):
        windows.append(features[start : start + window])
        labels.append(label_id)
    if not windows:
        return np.empty((0, window, len(FEATURE_COLS)), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.stack(windows).astype(np.float32), np.array(labels, dtype=np.int64)


def split_for_index(idx: int) -> str:
    if 1 <= idx <= 8:
        return "train"
    if 9 <= idx <= 10:
        return "val"
    if 11 <= idx <= 12:
        return "test"
    return "skip"


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    buckets_x: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    buckets_y: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0, "skipped": 0}

    for csv_path in sorted(data_dir.glob("*.csv")):
        match = FILE_RE.match(csv_path.name)
        if not match:
            counts["skipped"] += 1
            continue

        label_name = match.group(1).lower()
        idx = int(match.group(2))
        split = split_for_index(idx)
        if split == "skip":
            counts["skipped"] += 1
            continue

        df = pd.read_csv(csv_path)
        if not set(FEATURE_COLS).issubset(df.columns):
            counts["skipped"] += 1
            continue

        features = normalize_features(df, args.pot_max)
        x, y = make_windows(features, LABEL_MAP[label_name], args.window, args.stride)
        if len(x) == 0:
            counts["skipped"] += 1
            continue

        buckets_x[split].append(x)
        buckets_y[split].append(y)
        counts[split] += len(x)

        print(f"[{split}] {csv_path.name}: rows={len(df)}, windows={len(x)}")

    for split in ("train", "val", "test"):
        if buckets_x[split]:
            x_all = np.concatenate(buckets_x[split], axis=0)
            y_all = np.concatenate(buckets_y[split], axis=0)
        else:
            x_all = np.empty((0, args.window, len(FEATURE_COLS)), dtype=np.float32)
            y_all = np.empty((0,), dtype=np.int64)
        np.savez(out_dir / f"{split}.npz", X=x_all, y=y_all)
        print(f"{split}: X{tuple(x_all.shape)} y{tuple(y_all.shape)}")

    meta = {
        "feature_order": FEATURE_COLS,
        "label_map": LABEL_MAP,
        "window": args.window,
        "stride": args.stride,
        "normalization": {
            "accel_divisor": 16384.0,
            "gyro_divisor": 32768.0,
            "pot_divisor": args.pot_max,
            "accel_gyro_range": [-1.0, 1.0],
            "pot_range": [0.0, 1.0],
        },
        "window_counts": counts,
    }
    (out_dir / "scaler_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote metadata: {out_dir / 'scaler_meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
