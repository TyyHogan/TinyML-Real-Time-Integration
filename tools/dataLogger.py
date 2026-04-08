# Firmware line format (no header on wire): frame_id,ax,ay,az,gx,gy,gz,pot
# Install: pip install -r tools/requirements.txt
# Example: python tools/dataLogger.py --port COM5 --out data/idle.csv --seconds 30

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from dataclasses import dataclass

import serial

U32 = 1 << 32

_DATA_RE = re.compile(
    r"^(\d+),(-?\d+),(-?\d+),(-?\d+),(-?\d+),(-?\d+),(-?\d+),(\d+)\s*$"
)


def u32_forward_delta(prev: int, curr: int) -> int:
    return (curr - prev) % U32


@dataclass
class SessionStats:
    rows_written: int = 0
    frames_missing: int = 0
    duplicate_count: int = 0
    skipped_lines: int = 0
    last_frame_id: int | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Log 100 Hz IMU+pot CSV from STM32 UART2"
    )
    p.add_argument(
        "--port", required=True, help="Serial port (e.g. COM5 or /dev/ttyUSB0)"
    )
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--out", required=True, help="Output CSV path (e.g. data/wave.csv)")
    p.add_argument("--seconds", type=float, default=None, help="Stop after N seconds")
    p.add_argument("--max-rows", type=int, default=None, help="Stop after N data rows")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    stats = SessionStats()
    t0 = time.perf_counter()

    with serial.Serial(args.port, args.baud, timeout=0.1) as ser, open(
        args.out, "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["frame_id", "ax", "ay", "az", "gx", "gy", "gz", "pot"])

        while True:
            if args.seconds is not None and (time.perf_counter() - t0) >= args.seconds:
                break
            if args.max_rows is not None and stats.rows_written >= args.max_rows:
                break

            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace").strip()
            m = _DATA_RE.match(line)
            if not m:
                stats.skipped_lines += 1
                continue

            frame_id = int(m.group(1))
            # Keep all 8 captured fields: frame_id + 6 IMU axes + pot
            row = [frame_id] + [int(m.group(i)) for i in range(2, 9)]

            if stats.last_frame_id is not None:
                d = u32_forward_delta(stats.last_frame_id, frame_id)
                if d == 0:
                    stats.duplicate_count += 1
                elif d > 1:
                    stats.frames_missing += d - 1

            stats.last_frame_id = frame_id
            writer.writerow(row)
            stats.rows_written += 1
            f.flush()

    dt = time.perf_counter() - t0
    expected = stats.rows_written + stats.frames_missing
    drop_rate = (stats.frames_missing / expected * 100.0) if expected else 0.0

    print(f"Wrote {stats.rows_written} rows to {args.out}", file=sys.stderr)
    print(f"Duration: {dt:.3f}s", file=sys.stderr)
    print(
        f"Missing frame_ids (gaps): {stats.frames_missing}",
        file=sys.stderr,
    )
    print(f"Approx. gap rate: {drop_rate:.3f}%", file=sys.stderr)
    print(f"Skipped non-data lines: {stats.skipped_lines}", file=sys.stderr)
    print(f"Duplicate frame_ids: {stats.duplicate_count}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
