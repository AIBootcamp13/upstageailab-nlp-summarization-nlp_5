#!/bin/bash
# 하이퍼파라미터 튜닝 Sweep 실행 스크립트

echo "Starting hyperparameter tuning sweep..."
echo "This will test various learning rates, batch sizes, and other training parameters."

python sweep_runner.py \
    --base-config ../config/base_config.yaml \
    --sweep-config hyperparameter_sweep \
    --count 50 \
    --log-level INFO

echo "Sweep completed! Check the results in outputs/sweep_hyperparameter_sweep/"
