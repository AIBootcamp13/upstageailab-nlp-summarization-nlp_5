#!/bin/bash
# 통합 빠른 테스트 러너
# 모든 모델에 대해 빠른 테스트를 수행하여 파이프라인 동작을 검증

set -e

echo "🚀 NLP 대화 요약 통합 빠른 테스트"
echo "=================================="

# 옵션 처리
ALL_MODELS=false
SPECIFIC_MODEL=""
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            ALL_MODELS=true
            shift
            ;;
        --model)
            SPECIFIC_MODEL="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            echo "사용법: $0 [옵션]"
            echo ""
            echo "옵션:"
            echo "  --all                모든 모델 테스트"
            echo "  --model MODEL        특정 모델만 테스트 (eenzeenee, xlsum_mt5, baseline)"
            echo "  --verbose            상세 로깅"
            echo "  --help               도움말 표시"
            echo ""
            echo "예제:"
            echo "  $0 --model eenzeenee     # eenzeenee 모델만 테스트"
            echo "  $0 --all                 # 모든 모델 테스트"
            exit 0
            ;;
        *)
            echo "알 수 없는 옵션: $1"
            exit 1
            ;;
    esac
done

# 기본값 설정
if [ "$ALL_MODELS" = "false" ] && [ -z "$SPECIFIC_MODEL" ]; then
    SPECIFIC_MODEL="eenzeenee"  # 기본값
fi

# 사용 가능한 모델 목록
AVAILABLE_MODELS=("eenzeenee" "xlsum_mt5" "baseline")

# 테스트할 모델 목록 결정
if [ "$ALL_MODELS" = "true" ]; then
    MODELS_TO_TEST=("${AVAILABLE_MODELS[@]}")
elif [ -n "$SPECIFIC_MODEL" ]; then
    # 모델이 유효한지 확인
    if [[ " ${AVAILABLE_MODELS[@]} " =~ " ${SPECIFIC_MODEL} " ]]; then
        MODELS_TO_TEST=("$SPECIFIC_MODEL")
    else
        echo "❌ 지원하지 않는 모델: $SPECIFIC_MODEL"
        echo "사용 가능한 모델: ${AVAILABLE_MODELS[*]}"
        exit 1
    fi
fi

echo "🎯 테스트할 모델: ${MODELS_TO_TEST[*]}"
echo ""

# 전체 시작 시간
OVERALL_START=$(date +%s)
TEST_RESULTS=()

# 각 모델에 대해 빠른 테스트 실행
for model in "${MODELS_TO_TEST[@]}"; do
    echo "🧪 Testing model: $model"
    echo "===================="
    
    START_TIME=$(date +%s)
    
    # 모델별 테스트 실행
    if [ "$VERBOSE" = "true" ]; then
        VERBOSE_FLAG="--verbose"
    else
        VERBOSE_FLAG=""
    fi
    
    if python quick_test.py --config config.yaml --model-section "$model" --max-samples 50 $VERBOSE_FLAG; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo "✅ $model 테스트 성공 (${DURATION}초)"
        TEST_RESULTS+=("$model:SUCCESS:${DURATION}")
    else
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo "❌ $model 테스트 실패 (${DURATION}초)"
        TEST_RESULTS+=("$model:FAILED:${DURATION}")
    fi
    
    echo ""
done

# 전체 결과 요약
OVERALL_END=$(date +%s)
TOTAL_DURATION=$((OVERALL_END - OVERALL_START))

echo "📊 빠른 테스트 결과 요약"
echo "======================"
echo "전체 소요 시간: ${TOTAL_DURATION}초"
echo ""

SUCCESS_COUNT=0
FAILED_COUNT=0

for result in "${TEST_RESULTS[@]}"; do
    IFS=':' read -r model status duration <<< "$result"
    if [ "$status" = "SUCCESS" ]; then
        echo "✅ $model: 성공 (${duration}초)"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "❌ $model: 실패 (${duration}초)"
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
done

echo ""
echo "📈 통계:"
echo "  총 테스트: $((SUCCESS_COUNT + FAILED_COUNT))"
echo "  성공: $SUCCESS_COUNT"
echo "  실패: $FAILED_COUNT"

if [ $FAILED_COUNT -eq 0 ]; then
    echo ""
    echo "🎉 모든 빠른 테스트가 성공했습니다!"
    echo "이제 전체 훈련을 실행할 수 있습니다."
    echo ""
    echo "💡 다음 단계:"
    if [ ${#MODELS_TO_TEST[@]} -eq 1 ]; then
        model="${MODELS_TO_TEST[0]}"
        case $model in
            eenzeenee)
                echo "  EENZEENEE_RUN_ACTUAL=true ./run_eenzeenee_experiment.sh"
                ;;
            xlsum_mt5)
                echo "  # xlsum_mt5 전체 훈련 스크립트 실행"
                echo "  python code/trainer.py --config config.yaml --config-section xlsum_mt5"
                ;;
            baseline)
                echo "  # baseline 전체 훈련 스크립트 실행"
                echo "  python code/trainer.py --config config.yaml"
                ;;
        esac
    else
        echo "  각 모델별로 전체 훈련을 실행하세요"
        echo "  또는 run_multi_model_experiments.sh 사용"
    fi
    
    exit 0
else
    echo ""
    echo "⚠️  일부 테스트가 실패했습니다."
    echo "실패한 모델들을 확인하고 문제를 해결한 후 다시 시도하세요."
    
    exit 1
fi
