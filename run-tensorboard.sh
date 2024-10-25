#!/bin/bash

HOME_DIR="/home/shodh/framework"
LOGS_PROFILER="/home/shodh/framework/logs/profiler"
LOGS_TRANSFORMER="/home/shodh/framework/logs/transformer"
LOGS_DIR="/home/shodh/framework/logs"

LOG_COUNT_PROFILER=$(ls -1 "$LOGS_PROFILER" | wc -l)
LOG_COUNT_TRANSFORMER=$(ls -1 "$LOGS_TRANSFORMER" | wc -l)

if [ "$LOG_COUNT_TRANSFORMER" -lt 5 ]; then
  tensorboard --logdir "$LOGS_DIR" --host=0.0.0.0 --port=6006
  exit 
fi

if [ "$LOG_COUNT_TRANSFORMER" -gt 5 ]; then
  find "$LOGS_TRANSFORMER" -mindepth 1 -maxdepth 1 | sort -r | tail -n +3 | xargs -d '\n' rm -rf
fi

if [ "$LOG_COUNT_PROFILER" -gt 5 ]; then
  find "$LOGS_PROFILER" -mindepth 1 -maxdepth 1 | sort -r | tail -n +3 | xargs -d '\n' rm -rf
fi

cd "$LOGS_TRANSFORMER"
TRANSFORMER_FILES=($(ls -1t))
if [ ${#TRANSFORMER_FILES[@]} -gt 0 ]; then
  mv "${TRANSFORMER_FILES[1]}" version_0
fi
if [ ${#TRANSFORMER_FILES[@]} -gt 1 ]; then
  mv "${TRANSFORMER_FILES[0]}" version_1
fi

cd "$LOGS_PROFILER"
PROFILER_FILES=($(ls -1t))
if [ ${#PROFILER_FILES[@]} -gt 0 ]; then
  mv "${PROFILER_FILES[1]}" version_0
fi
if [ ${#PROFILER_FILES[@]} -gt 1 ]; then
  mv "${PROFILER_FILES[0]}" version_1
fi

cd "$HOME_DIR"
tensorboard --logdir "$LOGS_DIR" --host=0.0.0.0 --port=6006
