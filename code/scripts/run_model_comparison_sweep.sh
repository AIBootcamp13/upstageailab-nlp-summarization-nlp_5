#!/bin/bash
# 모델 비교 Sweep 실행 스크립트

echo "Starting model comparison sweep..."
echo "This will compare KoBART, KoGPT2, T5-base, and mT5 models."

python sweep_runner.py \
    --base-config ../config/base_config.yaml \
    --sweep-config model_comparison_sweep \
    --count 20 \
    --log-level INFO

echo "Sweep completed! Check the results in outputs/sweep_model_comparison_sweep/"
