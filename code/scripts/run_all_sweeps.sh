#!/bin/bash
# 모든 Sweep 순차 실행 스크립트

echo "Running all sweep configurations sequentially..."
echo "This will take several hours to complete."

python parallel_sweep_runner.py \
    --base-config ../config/base_config.yaml \
    --multiple-serial hyperparameter_sweep model_comparison_sweep generation_params_sweep ablation_study_sweep \
    --runs-per-sweep 20 \
    --output-dir ./all_sweep_results \
    --log-level INFO

echo "All sweeps completed! Check the results in all_sweep_results/"
