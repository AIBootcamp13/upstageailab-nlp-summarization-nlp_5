#!/bin/bash
# eenzeenee T5 한국어 요약 모델 실험 러너 (빠른 테스트 지원)
# eenzeenee/t5-base-korean-summarization 모델을 사용한 한국어 요약 실험

set -e  # 에러 발생 시 중단

# 빠른 테스트 모드 옵션 처리
QUICK_TEST=false
if [ "${1}" = "--quick-test" ] || [ "${EENZEENEE_QUICK_TEST:-false}" = "true" ]; then
    QUICK_TEST=true
    echo "🚀 Quick Test Mode Enabled"
fi

echo "🤖 Starting eenzeenee T5 Korean Summarization Experiment"
if [ "$QUICK_TEST" = "true" ]; then
    echo "======================================================= (QUICK TEST)"
else
    echo "======================================================="
fi

# 프로젝트 루트에 있는지 확인
if [ ! -f "config.yaml" ]; then
    echo "❌ Please run this script from the project root directory"
    exit 1
fi

# eenzeenee 설정 파일 확인
CONFIG_SECTION="eenzeenee"
CONFIG_FILE="config.yaml"

if ! grep -q "^${CONFIG_SECTION}:" "$CONFIG_FILE"; then
    echo "❌ eenzeenee configuration not found in $CONFIG_FILE"
    echo "Please ensure the eenzeenee section is properly configured"
    exit 1
fi

echo "✅ Found eenzeenee configuration in $CONFIG_FILE"

# 시스템 리소스 확인
echo "🔍 Checking system resources..."
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,nounits,noheader
else
    echo "⚠️  No GPU detected. Training will be slower on CPU."
fi

# 실험을 위한 출력 디렉토리 생성
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [ "$QUICK_TEST" = "true" ]; then
    OUTPUT_DIR="outputs/eenzeenee_quick_test_${TIMESTAMP}"
else
    OUTPUT_DIR="outputs/eenzeenee_experiment_${TIMESTAMP}"
fi
mkdir -p "$OUTPUT_DIR"

echo "📁 Results will be saved to: $OUTPUT_DIR"

# 로그 파일 설정
LOG_FILE="$OUTPUT_DIR/training.log"
echo "📝 Training log: $LOG_FILE"

# 실험 메타데이터 저장
cat > "$OUTPUT_DIR/experiment_info.json" << EOF
{
    "experiment_name": "eenzeenee_korean_summarization",
    "model_name": "eenzeenee/t5-base-korean-summarization",
    "config_section": "$CONFIG_SECTION",
    "config_file": "$CONFIG_FILE",
    "start_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "quick_test": $QUICK_TEST,
    "description": "Korean dialogue summarization using eenzeenee T5 model with automatic prefix handling",
    "model_type": "seq2seq",
    "architecture": "T5",
    "prefix": "summarize: "
}
EOF

echo "📄 Experiment metadata saved to: $OUTPUT_DIR/experiment_info.json"
echo ""

# 데이터 파일 확인
echo "🔍 Checking data files..."
DATA_FILES=("data/train.csv" "data/dev.csv" "data/test.csv")
for data_file in "${DATA_FILES[@]}"; do
    if [ -f "$data_file" ]; then
        echo "✅ Found: $data_file"
    else
        echo "⚠️  Missing: $data_file"
    fi
done
echo ""

# eenzeenee 모델의 특성 안내
echo "🔧 eenzeenee Model Configuration:"
echo "   - Model: eenzeenee/t5-base-korean-summarization"
echo "   - Type: T5-based sequence-to-sequence model"
echo "   - Language: Korean-optimized"
echo "   - Prefix: 'summarize: ' (automatically added)"
if [ "$QUICK_TEST" = "true" ]; then
    echo "   - Quick Test: 1 epoch, ~50 samples"
    echo "   - Batch size: 2 (reduced for speed)"
else
    echo "   - Recommended batch size: 4-8 (depending on GPU memory)"
    echo "   - Recommended max length: Input 512, Output 200"
fi
echo ""

# 실험 실행 옵션
if [ "$QUICK_TEST" = "true" ]; then
    echo "🚀 Starting eenzeenee quick test..."
    echo "This quick test will:"
    echo "   1. Load the eenzeenee/t5-base-korean-summarization model"
    echo "   2. Use only ~50 training samples for speed"
    echo "   3. Run for 1 epoch only"
    echo "   4. Verify the entire pipeline works without errors"
else
    echo "🚀 Starting eenzeenee experiment..."
    echo "This experiment will:"
    echo "   1. Load the eenzeenee/t5-base-korean-summarization model"
    echo "   2. Apply 'summarize: ' prefix automatically to all inputs"
    echo "   3. Fine-tune on Korean dialogue summarization data"
    echo "   4. Evaluate performance using ROUGE metrics"
fi
echo ""

# 실제 훈련 명령어 표시
echo "To run the actual training, execute:"
if [ "$QUICK_TEST" = "true" ]; then
    echo "python quick_test.py \\"
    echo "    --config $CONFIG_FILE \\"
    echo "    --model-section $CONFIG_SECTION \\"
    echo "    --max-samples 50"
else
    echo "python code/trainer.py \\"
    echo "    --config $CONFIG_FILE \\"
    echo "    --config-section $CONFIG_SECTION \\"
    echo "    --output-dir $OUTPUT_DIR \\"
    echo "    --train-data data/train.csv \\"
    echo "    --val-data data/dev.csv \\"
    echo "    --test-data data/test.csv"
fi
echo ""

# 개발/테스트 모드에서는 실제 실행하지 않음
if [ "${EENZEENEE_RUN_ACTUAL:-false}" = "true" ]; then
    if [ "$QUICK_TEST" = "true" ]; then
        echo "🏃 Running quick test..."
    else
        echo "🏃 Running actual training..."
    fi
    
    if [ "$QUICK_TEST" = "true" ]; then
        # 빠른 테스트 실행
        python quick_test.py \
            --config "$CONFIG_FILE" \
            --model-section "$CONFIG_SECTION" \
            --max-samples 50 \
            2>&1 | tee "$LOG_FILE"
    else
        # 정상 훈련 실행
        python code/trainer.py \
            --config "$CONFIG_FILE" \
            --config-section "$CONFIG_SECTION" \
            --output-dir "$OUTPUT_DIR" \
            --train-data data/train.csv \
            --val-data data/dev.csv \
            --test-data data/test.csv \
            2>&1 | tee "$LOG_FILE"
    fi
    
    # 실행 결과 확인
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        if [ "$QUICK_TEST" = "true" ]; then
            echo "✅ eenzeenee quick test completed successfully!"
        else
            echo "✅ eenzeenee experiment completed successfully!"
        fi
        
        # 결과 요약
        if [ -f "$OUTPUT_DIR/results/summary.txt" ]; then
            echo ""
            echo "📊 Experiment Results Summary:"
            echo "=============================="
            cat "$OUTPUT_DIR/results/summary.txt"
        fi
    else
        if [ "$QUICK_TEST" = "true" ]; then
            echo "❌ eenzeenee quick test failed. Check logs for details."
        else
            echo "❌ eenzeenee experiment failed. Check logs for details."
        fi
        exit 1
    fi
else
    echo "ℹ️  Script completed in setup mode."
    if [ "$QUICK_TEST" = "true" ]; then
        echo "   Set EENZEENEE_RUN_ACTUAL=true to run quick test:"
        echo "   EENZEENEE_RUN_ACTUAL=true ./run_eenzeenee_experiment.sh --quick-test"
        echo "   또는 직접 실행:"
        echo "   python quick_test.py --config $CONFIG_FILE --model-section $CONFIG_SECTION"
    else
        echo "   Set EENZEENEE_RUN_ACTUAL=true to run actual training:"
        echo "   EENZEENEE_RUN_ACTUAL=true ./run_eenzeenee_experiment.sh"
    fi
fi

echo ""
echo "📋 Experiment Setup Complete!"
echo ""
echo "💡 Tips for running eenzeenee experiments:"
if [ "$QUICK_TEST" = "true" ]; then
    echo "   • 빠른 테스트로 1에포크만 실행합니다"
    echo "   • 50개 샘플만 사용하여 빠르게 동작 검증"
    echo "   • 전체 훈련은 --quick-test 옵션 제거 후 실행"
    echo "   • 약 5-10분 내에 완료됩니다"
else
    echo "   • Ensure sufficient GPU memory (8GB+ recommended)"
    echo "   • Monitor training with WandB if configured"
    echo "   • The model automatically handles Korean text preprocessing"
    echo "   • Prefix 'summarize: ' is added automatically - no manual intervention needed"
    echo "   • 빠른 테스트: --quick-test 옵션 사용"
fi
echo ""
echo "🏁 Experiment: $OUTPUT_DIR"

# 최종 상태 저장
cat >> "$OUTPUT_DIR/experiment_info.json" << EOF
,
    "setup_complete_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "status": "ready"
}
EOF

# JSON 구문 수정 (마지막 콤마 제거)
sed -i '' 's/},$/}/' "$OUTPUT_DIR/experiment_info.json" 2>/dev/null || sed -i 's/},$/}/' "$OUTPUT_DIR/experiment_info.json"
