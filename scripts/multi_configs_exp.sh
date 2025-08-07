#!/bin/bash

# 스크립트가 오류 발생 시 바로 중단되도록 설정합니다.
set -e

# 로그를 저장할 디렉터리를 생성합니다. 이미 존재하면 오류를 무시합니다.
mkdir -p logs

# 실행할 YAML 설정 파일 목록을 정의합니다.
CONFIG_FILES=(
    "config_exps_08010324.yaml"
    "config_exps_08010337.yaml"
    "config_exps_08010408.yaml"
    "config_exps_08010421.yaml"
)

# 파일 목록을 순회하며 각 파일을 순차적으로 실행합니다.
for config in "${CONFIG_FILES[@]}"; do
    LOG_FILE="logs/log_${config}.log"
    echo "--- ${config} 파일 실행 시작 (로그 파일: $LOG_FILE) ---"
    python src/main_base.py --config "$config" > "$LOG_FILE" 2>&1
    echo "--- ${config} 파일 실행 완료 ---"
done

echo ""
echo "--- 모든 파일 실행 완료 ---"