# RT_Project — TinyML gesture pipeline (STM32F446 + MPU-6050)

Firmware on an **STM32F446RE** (Nucleo-64) samples IMU + potentiometer at **100 Hz**, streams labeled CSV over **UART2** for capture, then a Python toolchain builds **windowed tensors**, trains a small model, exports **TFLite**, and can embed weights as **`Core/Inc/modelData.h`** for on-device inference.

## Hardware

- **MCU:** STM32F446RE @ 180 MHz (CubeMX / STM32CubeIDE)
- **IMU:** MPU-6050 on **I2C1** (project uses PB8/PB9 in user MSP init)
- **Analog:** 10k pot on **ADC1** (e.g. PA0)
- **Debug / console:** ST-Link **VCP** on USART2 (default **115200** baud)

## Repository layout

| Path | Role |
|------|------|
| `Core/` | Cube-generated firmware + application (`main.c`, `freertos.c`, …) |
| `tools/dataLogger.py` | Record UART CSV to disk (`frame_id` + 7 features) |
| `tools/preprocess.py` | Normalize, sliding windows, train/val/test split → `processedData/*.npz` |
| `tools/modelTrainer.py` | Train Keras baseline, export `.h5` / `.tflite`, log `metrics/runs.csv` |
| `tools/quantizer.py` | Post-training int8 TFLite from a `.h5` |
| `tools/embed.py` | Generate `Core/Inc/modelData.h` from a `.tflite` |
| `saveData/` | Raw captured CSVs (optional; ignored if listed in `.gitignore`) |
| `processedData/` | Preprocessed tensors + `scaler_meta.json` |
| `models/` | Trained exports (`.h5`, `.tflite`) |

## Firmware (STM32CubeIDE)

1. Open the project in **STM32CubeIDE**.
2. **Project → Build Project** (or build the `Debug` configuration).
3. Flash the Nucleo and open a serial terminal on the ST-Link COM port at **115200 8N1**.

CSV lines look like:

```text
frame_id,ax,ay,az,gx,gy,gz,pot
```

Enable or disable high-rate UART CSV in `Core/Src/main.c` via `SENSOR_CSV_UART_LOG` (see comments there). Gesture runtime code lives under `gesture_runtime.*` when enabled.

## Python environment

From the repo root:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r tools/requirements.txt
```

For training and quantization you also need TensorFlow (not pinned in `tools/requirements.txt`):

```powershell
pip install tensorflow
```

## End-to-end data / model workflow

### 1) Capture data (host)

```powershell
python tools/dataLogger.py --port COMx --out saveData/wave_01.csv --seconds 30
```

Use your actual **COM** port. Repeat for each gesture class and split files by name (see `tools/preprocess.py` naming rules).

### 2) Preprocess

```powershell
python tools/preprocess.py --data-dir saveData --out-dir processedData
```

### 3) Train + float TFLite

```powershell
python tools/modelTrainer.py --run-name my_run
```

### 4) Int8 quantization (optional)

```powershell
python tools/quantizer.py --h5 models/my_run.h5
```

### 5) Embed model for the MCU

```powershell
python tools/embed.py -i models/my_run_int8.tflite -o Core/Inc/modelData.h
```

Rebuild firmware after regenerating the header.

## Metrics

Training runs append to `metrics/runs.csv`; the latest confusion matrix JSON is written to `metrics/confusion_matrix.json` when you run `modelTrainer.py`.

## License / third party

HAL, CMSIS, FreeRTOS, and TensorFlow Lite Micro (if present under `Middlewares/`) remain under their respective licenses. Application code in `Core/` user sections and `tools/` is yours to license as you choose.
