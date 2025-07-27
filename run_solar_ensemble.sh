#!/bin/bash
# Solar API 앙상블 실행 스크립트

echo "========================================="
echo "Solar API 앙상블 시스템 실행"
echo "========================================="
echo ""

# 환경 변수 확인
if [ -z "$UPSTAGE_API_KEY" ]; then
    echo "ERROR: UPSTAGE_API_KEY 환경 변수가 설정되지 않았습니다."
    echo "다음 명령어로 설정하세요:"
    echo "export UPSTAGE_API_KEY='your-api-key'"
    exit 1
fi

# Python 경로 설정
export PYTHONPATH="${PYTHONPATH}:$(pwd)/code"

# 디바이스 설정
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# 출력 디렉토리 생성
mkdir -p outputs/solar_ensemble
mkdir -p logs
mkdir -p cache/solar_ensemble

# 최고 성능 모델 확인
echo "최고 성능 모델 확인 중..."
if [ ! -d "outputs/phase2_results/10c_all_optimizations" ]; then
    echo "WARNING: 2차 실험 결과가 없습니다. 기본 모델을 사용합니다."
fi

# 실험 모드 선택
if [ "$1" == "all" ]; then
    echo "모든 앙상블 모드 실행"
    python code/run_solar_ensemble.py \
        --config config/experiments/11_solar_ensemble.yaml \
        --all
elif [ "$1" == "test" ]; then
    echo "테스트 모드 실행 (100개 샘플)"
    python code/run_solar_ensemble.py \
        --config config/experiments/11_solar_ensemble.yaml \
        --mode dynamic_weights \
        --sample 100
else
    echo "동적 가중치 모드 실행 (권장)"
    python code/run_solar_ensemble.py \
        --config config/experiments/11_solar_ensemble.yaml \
        --mode dynamic_weights
fi

echo ""
echo "========================================="
echo "Solar API 앙상블 완료!"
echo "========================================="
echo ""
echo "결과 확인:"
echo "- 앙상블 결과: outputs/solar_ensemble/"
echo "- 실험 통계: outputs/solar_ensemble/*/experiment_stats.json"
echo "- 제출 파일: outputs/solar_ensemble/*/submission.csv"
