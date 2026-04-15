#include "gesture_runtime.h"

#include <stdio.h>
#include <string.h>

#include "cmsis_os2.h"

/* -------------------------------------------------------------------------- */
/* Normalization — must match tools/preprocess.py                             */
/* -------------------------------------------------------------------------- */
static float clipf(float x, float lo, float hi)
{
    if (x < lo) {
        return lo;
    }
    if (x > hi) {
        return hi;
    }
    return x;
}

static void normalize_sample(int16_t ax, int16_t ay, int16_t az,
                             int16_t gx, int16_t gy, int16_t gz,
                             uint32_t adc_raw, float out[7])
{
    out[0] = clipf((float)ax / 16384.0f, -1.0f, 1.0f);
    out[1] = clipf((float)ay / 16384.0f, -1.0f, 1.0f);
    out[2] = clipf((float)az / 16384.0f, -1.0f, 1.0f);
    out[3] = clipf((float)gx / 32768.0f, -1.0f, 1.0f);
    out[4] = clipf((float)gy / 32768.0f, -1.0f, 1.0f);
    out[5] = clipf((float)gz / 32768.0f, -1.0f, 1.0f);
    out[6] = clipf((float)adc_raw / GESTURE_POT_DIVISOR, 0.0f, 1.0f);
}

/* -------------------------------------------------------------------------- */
/* Weak hook: replace with TensorFlow Lite Micro interpreter in a .cpp file   */
/* Return 0 on success, negative on error.                                   */
/* -------------------------------------------------------------------------- */
__attribute__((weak)) int Gesture_TFLiteMicro_Run(const float input[GESTURE_INPUT_FLOATS],
                                                  int *class_index)
{
    (void)input;
    (void)class_index;
    return -1;
}

/* -------------------------------------------------------------------------- */
/* CMSIS-RTOS2 queue of 7-float samples                                       */
/* -------------------------------------------------------------------------- */
#define SAMPLE_Q_DEPTH 128U

typedef struct {
    float v[GESTURE_N_FEATURES];
} GestureSampleMsg;

static osMessageQueueId_t s_sample_q;

void GestureRuntime_Init(void)
{
    const osMessageQueueAttr_t attr = { .name = "gesture_samples" };
    s_sample_q = osMessageQueueNew(SAMPLE_Q_DEPTH, sizeof(GestureSampleMsg), &attr);
}

void GestureRuntime_OnSample(int16_t ax, int16_t ay, int16_t az,
                             int16_t gx, int16_t gy, int16_t gz,
                             uint32_t adc_raw)
{
    if (s_sample_q == NULL) {
        return;
    }
    GestureSampleMsg m;
    normalize_sample(ax, ay, az, gx, gy, gz, adc_raw, m.v);
    /* Drop if full — keep acquisition real-time */
    (void)osMessageQueuePut(s_sample_q, &m, 0U, 0U);
}

static const char *class_name(int idx)
{
    switch (idx) {
        case 0: return "idle";
        case 1: return "wave";
        case 2: return "shake";
        default: return "?";
    }
}

void GestureRuntime_AITaskLoop(void)
{
    float window[GESTURE_WINDOW_LEN][GESTURE_N_FEATURES];
    float flat[GESTURE_INPUT_FLOATS];
    uint32_t last_print_ms = 0U;
    uint8_t stub_warned = 0U;

    for (;;) {
        /* Collect one contiguous window */
        for (uint32_t i = 0U; i < GESTURE_WINDOW_LEN; i++) {
            GestureSampleMsg m;
            if (osMessageQueueGet(s_sample_q, &m, NULL, osWaitForever) != osOK) {
                return;
            }
            memcpy(window[i], m.v, sizeof(m.v));
        }

        uint32_t k = 0U;
        for (uint32_t t = 0U; t < GESTURE_WINDOW_LEN; t++) {
            for (uint32_t f = 0U; f < GESTURE_N_FEATURES; f++) {
                flat[k++] = window[t][f];
            }
        }

        int cls = -1;
        int err = Gesture_TFLiteMicro_Run(flat, &cls);

        uint32_t now = osKernelGetTickCount();
        if ((now - last_print_ms) >= 200U) {
            last_print_ms = now;
            if (err != 0) {
                if (stub_warned == 0U) {
                    stub_warned = 1U;
                    printf("[AI] TFLM runtime not linked — add generated TFLM + Gesture_TFLiteMicro_Run()\r\n");
                }
            } else {
                printf("[AI] class=%d (%s)\r\n", cls, class_name(cls));
            }
        }

        osDelay(5);
    }
}
