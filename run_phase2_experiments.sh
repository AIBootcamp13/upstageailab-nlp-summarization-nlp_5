#!/bin/bash
# 2차 조합 실험 실행 스크립트

echo "========================================="
echo "조합 실험 2차 - 고급 기능 통합"
echo "========================================="
echo ""

# 환경 변수 설정
export PYTHONPATH="${PYTHONPATH}:$(pwd)/code"
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# 실험 디렉토리 생성
mkdir -p logs/10_combination_phase2
mkdir -p outputs/phase2_results

# 실험 설정 경로
EXPERIMENT_DIR="config/experiments/10_combination_phase2"

echo "실험 1/3: Phase1 + Token Weight"
echo "================================"
python code/auto_experiment_runner.py \
    --config "${EXPERIMENT_DIR}/10a_phase1_plus_token_weight.yaml" \
    --output_dir "outputs/phase2_results/10a_token_weight" \
    --log_dir "logs/10_combination_phase2/10a_token_weight" \
    --device "auto"

echo ""
echo "실험 2/3: Phase1 + BackTranslation"
echo "==================================="
python code/auto_experiment_runner.py \
    --config "${EXPERIMENT_DIR}/10b_phase1_plus_backtrans.yaml" \
    --output_dir "outputs/phase2_results/10b_backtrans" \
    --log_dir "logs/10_combination_phase2/10b_backtrans" \
    --device "auto"

echo ""
echo "실험 3/3: All Optimizations"
echo "============================"
python code/auto_experiment_runner.py \
    --config "${EXPERIMENT_DIR}/10c_all_optimizations.yaml" \
    --output_dir "outputs/phase2_results/10c_all_optimizations" \
    --log_dir "logs/10_combination_phase2/10c_all_optimizations" \
    --device "auto"

echo ""
echo "========================================="
echo "모든 2차 실험 완료!"
echo "========================================="
echo ""

# 자동 분석 실행
echo "실험 결과 분석 중..."
python scripts/analyze_combination_phase2.py

echo ""
echo "분석 완료! 결과는 outputs/phase2_analysis에서 확인하세요."
