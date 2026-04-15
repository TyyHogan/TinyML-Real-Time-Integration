#ifndef GESTURE_RUNTIME_H
#define GESTURE_RUNTIME_H

#include <stdint.h>

/* Match tools/preprocess.py: 50 samples × 7 features → flattened Dense input */
#define GESTURE_WINDOW_LEN   50U
#define GESTURE_N_FEATURES   7U
#define GESTURE_INPUT_FLOATS   (GESTURE_WINDOW_LEN * GESTURE_N_FEATURES)

#define GESTURE_POT_DIVISOR    1024.0f

void GestureRuntime_Init(void);

/* Called from Sensor task after each successful IMU read (100 Hz). */
void GestureRuntime_OnSample(int16_t ax, int16_t ay, int16_t az,
                              int16_t gx, int16_t gy, int16_t gz,
                              uint32_t adc_raw);

/* Called from AITask: pulls windows, runs model hook (weak until TFLM linked). */
void GestureRuntime_AITaskLoop(void);

#endif /* GESTURE_RUNTIME_H */
