#!/bin/bash
# 병렬 Sweep 실행 스크립트 (4개 워커 사용)

echo "Starting parallel hyperparameter sweep with 4 workers..."
echo "Each worker will run 10 experiments (total 40 runs)"

python parallel_sweep_runner.py \
    --base-config ../config/base_config.yaml \
    --single-parallel hyperparameter_sweep \
    --num-workers 4 \
    --runs-per-worker 10 \
    --output-dir ./parallel_sweep_results \
    --log-level INFO

echo "Parallel sweep completed! Check the results in parallel_sweep_results/"
