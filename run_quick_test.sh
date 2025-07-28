#!/bin/bash
# 빠른 검증 테스트 러너 스크립트
# 1에포크만 실행하여 전체 파이프라인 동작 확인

set -e

echo "🚀 NLP 대화 요약 빠른 테스트 시작"
echo "=================================="

# 기본 설정
CONFIG_FILE="config.yaml"
MAX_SAMPLES=100
MODEL_SECTION=""
VERBOSE=false

# 명령행 인자 처리
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --model-section)
            MODEL_SECTION="$2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
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
            echo "  --config FILE           설정 파일 경로 (기본: config.yaml)"
            echo "  --model-section SECTION 사용할 모델 섹션 (eenzeenee, xlsum_mt5, baseline)"
            echo "  --max-samples N         최대 훈련 샘플 수 (기본: 100)"
            echo "  --verbose               상세 로깅 활성화"
            echo "  --help                  이 도움말 표시"
            echo ""
            echo "예제:"
            echo "  $0                                    # 기본 설정으로 실행"
            echo "  $0 --model-section eenzeenee         # eenzeenee 모델 테스트"
            echo "  $0 --max-samples 50 --verbose        # 50개 샘플로 상세 테스트"
            exit 0
            ;;
        *)
            echo "알 수 없는 옵션: $1"
            echo "도움말: $0 --help"
            exit 1
            ;;
    esac
done

# 설정 파일 확인
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 설정 파일을 찾을 수 없습니다: $CONFIG_FILE"
    exit 1
fi

echo "📁 설정 파일: $CONFIG_FILE"
if [ -n "$MODEL_SECTION" ]; then
    echo "🤖 모델 섹션: $MODEL_SECTION"
fi
echo "📊 최대 샘플 수: $MAX_SAMPLES"
echo ""

# Python 스크립트 실행
CMD="python quick_test.py --config $CONFIG_FILE --max-samples $MAX_SAMPLES"

if [ -n "$MODEL_SECTION" ]; then
    CMD="$CMD --model-section $MODEL_SECTION"
fi

if [ "$VERBOSE" = true ]; then
    CMD="$CMD --verbose"
fi

echo "🏃 실행 명령: $CMD"
echo ""

# 시작 시간 기록
START_TIME=$(date +%s)

# 실행
if eval $CMD; then
    # 종료 시간 계산
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    echo ""
    echo "🎉 빠른 테스트 완료!"
    echo "⏱️ 소요 시간: ${DURATION}초"
    echo ""
    echo "💡 팁:"
    echo "  - 전체 훈련을 실행하려면 epochs를 늘리고 samples 제한을 제거하세요"
    echo "  - 다른 모델을 테스트하려면 --model-section 옵션을 사용하세요"
    echo "  - 설정 파일을 수정하여 하이퍼파라미터를 조정할 수 있습니다"
    
    exit 0
else
    echo ""
    echo "❌ 빠른 테스트 실패"
    echo "💡 트러블슈팅:"
    echo "  - 로그를 확인하여 오류 원인을 파악하세요"
    echo "  - --verbose 옵션으로 상세 정보를 확인하세요"
    echo "  - GPU 메모리가 부족하면 --max-samples를 줄여보세요"
    
    exit 1
fi
