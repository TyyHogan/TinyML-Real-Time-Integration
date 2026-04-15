/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * File Name          : freertos.c
  * Description        : Code for freertos applications
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
#include "FreeRTOS.h"
#include "task.h"
#include "main.h"
#include "cmsis_os.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdio.h>
#include "gesture_runtime.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
/* TEMP_PRESENTATION_FALLBACK: set 0 to bypass new runtime while presenting */
#define GESTURE_RUNTIME_ENABLE 0

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN Variables */

/* USER CODE END Variables */
/* Definitions for Sensor1 */
osThreadId_t Sensor1Handle;
const osThreadAttr_t Sensor1_attributes = {
  .name = "Sensor1",
  .stack_size = 512 * 4,
  .priority = (osPriority_t) osPriorityAboveNormal,
};
/* Definitions for AITask */
osThreadId_t AITaskHandle;
const osThreadAttr_t AITask_attributes = {
  .name = "AITask",
  .stack_size = 2048 * 4,
  .priority = (osPriority_t) osPriorityLow,
};

/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN FunctionPrototypes */

/* USER CODE END FunctionPrototypes */

void StartDefaultTask(void *argument);
void StartTask02(void *argument);

void MX_FREERTOS_Init(void); /* (MISRA C 2004 rule 8.1) */

/**
  * @brief  FreeRTOS initialization
  * @param  None
  * @retval None
  */
void MX_FREERTOS_Init(void) {
  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* USER CODE BEGIN RTOS_MUTEX */
  /* add mutexes, ... */
  /* USER CODE END RTOS_MUTEX */

  /* USER CODE BEGIN RTOS_SEMAPHORES */
  /* add semaphores, ... */
  /* USER CODE END RTOS_SEMAPHORES */

  /* USER CODE BEGIN RTOS_TIMERS */
  /* start timers, add new ones, ... */
  /* USER CODE END RTOS_TIMERS */

  /* USER CODE BEGIN RTOS_QUEUES */
#if GESTURE_RUNTIME_ENABLE
  GestureRuntime_Init();
#endif
  /* USER CODE END RTOS_QUEUES */

  /* Create the thread(s) */
  /* creation of Sensor1 */
  Sensor1Handle = osThreadNew(StartDefaultTask, NULL, &Sensor1_attributes);

  /* creation of AITask */
  AITaskHandle = osThreadNew(StartTask02, NULL, &AITask_attributes);

  /* USER CODE BEGIN RTOS_THREADS */
  /* add threads, ... */
  /* USER CODE END RTOS_THREADS */

  /* USER CODE BEGIN RTOS_EVENTS */
  /* add events, ... */
  /* USER CODE END RTOS_EVENTS */

}

/* USER CODE BEGIN Header_StartDefaultTask */
/**
  * @brief  Function implementing the Sensor1 thread.
  * @param  argument: Not used
  * @retval None
  */
/* USER CODE END Header_StartDefaultTask */
void StartDefaultTask(void *argument)
{
  /* USER CODE BEGIN StartDefaultTask */
  /* Day6: monotonic frame_id every 10 ms tick — gaps in CSV = no row for that id (I2C drop or UART loss) */
  uint32_t next_wake_tick;
  uint32_t error_count = 0U;
  uint32_t frame_id = 0U;

  if (MPU6050_Init() != HAL_OK) {
    printf("MPU6050 init failed\r\n");
  } else {
    printf("MPU6050 init OK\r\n");
  }

  next_wake_tick = osKernelGetTickCount();
  for(;;) {
    frame_id++;
    if (Run_Sensor_Acquisition(frame_id) != HAL_OK) {
      error_count++;
      if ((error_count % 20U) == 0U) {
        printf("Sensor read error x%lu (last frame_id=%lu)\r\n",
               (unsigned long)error_count, (unsigned long)frame_id);
      }
    }
    next_wake_tick += 10U;
    osDelayUntil(next_wake_tick);
  }
  /* USER CODE END StartDefaultTask */
}

/* USER CODE BEGIN Header_StartTask02 */
/**
* @brief Function implementing the AITask thread.
* @param argument: Not used
* @retval None
*/
/* USER CODE END Header_StartTask02 */
void StartTask02(void *argument)
{
  /* USER CODE BEGIN StartTask02 */
#if GESTURE_RUNTIME_ENABLE
  GestureRuntime_AITaskLoop();
#else
  for(;;)
  {
    osDelay(100);
  }
#endif
  /* USER CODE END StartTask02 */
}

/* Private application code --------------------------------------------------*/
/* USER CODE BEGIN Application */

/* USER CODE END Application */

