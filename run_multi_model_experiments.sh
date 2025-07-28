#!/bin/bash
# Multi-model experiment runner
# 다양한 모델로 대화 요약 실험을 실행하는 스크립트

set -e  # 에러 발생 시 중단

# 실험 로그 디렉토리 생성
LOG_DIR="logs/multi_model_experiments"
mkdir -p $LOG_DIR

# 실험 시작 시간
START_TIME=$(date +%Y%m%d_%H%M%S)
echo "Starting multi-model experiments at $START_TIME"

# 사용 가능한 모델 목록
MODELS=(
    "bart_base"
    "t5_base"
    "mt5_base"
    "flan_t5_base"
    # "kogpt2"  # GPT 계열은 추가 수정 필요
    # "kobart_unsloth"  # unsloth 설치 필요
)

# 각 모델에 대해 실험 실행
for MODEL in "${MODELS[@]}"; do
    echo "=========================================="
    echo "Running experiment for: $MODEL"
    echo "=========================================="
    
    CONFIG_PATH="config/model_configs/${MODEL}.yaml"
    LOG_FILE="$LOG_DIR/${MODEL}_${START_TIME}.log"
    
    # 설정 파일 존재 확인
    if [ ! -f "$CONFIG_PATH" ]; then
        echo "Warning: Config file not found: $CONFIG_PATH"
        echo "Skipping $MODEL..."
        continue
    fi
    
    # 실험 실행
    echo "Running with config: $CONFIG_PATH"
    echo "Log file: $LOG_FILE"
    
    python code/trainer.py \
        --config "$CONFIG_PATH" \
        --train-data data/train.csv \
        --val-data data/dev.csv \
        --test-data data/test.csv \
        2>&1 | tee "$LOG_FILE"
    
    # 실행 결과 확인
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✅ $MODEL experiment completed successfully"
    else
        echo "❌ $MODEL experiment failed"
    fi
    
    echo ""
done

# 실험 종료
END_TIME=$(date +%Y%m%d_%H%M%S)
echo "All experiments completed at $END_TIME"

# 결과 요약
echo "=========================================="
echo "Experiment Summary"
echo "=========================================="
for MODEL in "${MODELS[@]}"; do
    RESULT_DIR="outputs/${MODEL}_*"
    if ls $RESULT_DIR 1> /dev/null 2>&1; then
        echo "✅ $MODEL: Results available"
        # 최신 결과 디렉토리 찾기
        LATEST_DIR=$(ls -td $RESULT_DIR | head -1)
        if [ -f "$LATEST_DIR/results/summary.txt" ]; then
            echo "   Best metrics:"
            grep -A 5 "Best Metrics:" "$LATEST_DIR/results/summary.txt" | tail -n +2
        fi
    else
        echo "❌ $MODEL: No results found"
    fi
    echo ""
done
