"""
Embed a .tflite file as a C header for TFLite Micro (flash-backed model).

Include Core/Inc/modelData.h in exactly one .c file to avoid duplicate symbols.

Example:
  python tools/embed_tflite.py -i models/model_v4_int8.tflite -o Core/Inc/modelData.h
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="Embed .tflite as modelData.h")
    p.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to .tflite",
    )
    p.add_argument(
        "--out",
        "-o",
        default="Core/Inc/modelData.h",
        help="Output header (default: Core/Inc/modelData.h)",
    )
    args = p.parse_args()

    blob = Path(args.input).read_bytes()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "#ifndef MODELDATA_H",
        "#define MODELDATA_H",
        "",
        "#include <stdint.h>",
        "",
        "/* Auto-generated — include in exactly one .c file */",
        "static const uint8_t g_modelData[] = {",
    ]
    for i in range(0, len(blob), 12):
        chunk = blob[i : i + 12]
        hexes = ", ".join(f"0x{b:02x}" for b in chunk)
        lines.append(f"    {hexes},")
    lines.extend(
        [
            "};",
            "",
            "#define G_MODELDATA_LEN (sizeof(g_modelData))",
            "",
            "#endif /* MODELDATA_H */",
            "",
        ]
    )
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out} ({len(blob)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
