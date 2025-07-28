#!/bin/bash
# RTX 3090 24GB 서버 최적화 - 5개 실험 1 epoch 빠른 테스트

set -e

# 스크립트가 실행되는 디렉토리로 이동
cd "$(dirname "$0")"

echo "⚡ RTX 3090 24GB - 5개 실험 1 Epoch 빠른 테스트"
echo "=============================================="
echo ""

# 테스트 실험 목록 (1 epoch)
TEST_EXPERIMENTS=(
    "config/experiments/test_01_mt5_xlsum_1epoch.yaml:mT5_XL-Sum_1epoch_테스트:4-6분"
    "config/experiments/test_02_eenzeenee_1epoch.yaml:eenzeenee_T5_1epoch_테스트:2-3분"  
    "config/experiments/test_03_kobart_1epoch.yaml:KoBART_1epoch_테스트:1-2분"
    "config/experiments/test_04_high_lr_1epoch.yaml:고성능_학습률_1epoch_테스트:1분"
    "config/experiments/test_05_batch_opt_1epoch.yaml:배치_최적화_1epoch_테스트:1분"
)

# 시작 시간 기록
START_TIME=$(date +%s)
echo "⏰ 테스트 시작 시간: $(date)"
echo "📊 예상 총 소요 시간: 약 10-15분"
echo ""

# 테스트 결과 저장 배열
declare -a RESULTS

# 각 테스트 순차 실행
for i in "${!TEST_EXPERIMENTS[@]}"; do
    IFS=':' read -r YAML_FILE DESCRIPTION EXPECTED_TIME <<< "${TEST_EXPERIMENTS[$i]}"
    TEST_NUM=$((i + 1))
    
    echo "🧪 테스트 ${TEST_NUM}/5: ${DESCRIPTION}"
    echo "📄 설정 파일: ${YAML_FILE}"
    echo "⏱️ 예상 시간: ${EXPECTED_TIME}"
    echo "----------------------------------------"
    
    # GPU 메모리 상태 확인
    echo "📊 테스트 전 GPU 상태:"
    nvidia-smi --query-gpu=memory.free,memory.used --format=csv,noheader,nounits
    echo ""
    
    # 테스트 실행
    TEST_START=$(date +%s)
    
    if python code/auto_experiment_runner.py --experiment "${YAML_FILE}"; then
        TEST_END=$(date +%s)
        TEST_TIME=$((TEST_END - TEST_START))
        TEST_MINUTES=$((TEST_TIME / 60))
        TEST_SECONDS=$((TEST_TIME % 60))
        
        echo "✅ 테스트 ${TEST_NUM} 성공!"
        echo "⏱️ 실제 소요 시간: ${TEST_MINUTES}분 ${TEST_SECONDS}초"
        RESULTS+=("✅ ${DESCRIPTION}: ${TEST_MINUTES}분 ${TEST_SECONDS}초")
    else
        TEST_END=$(date +%s)
        TEST_TIME=$((TEST_END - TEST_START))
        TEST_MINUTES=$((TEST_TIME / 60))
        TEST_SECONDS=$((TEST_TIME % 60))
        
        echo "❌ 테스트 ${TEST_NUM} 실패!"
        echo "⏱️ 실패까지 시간: ${TEST_MINUTES}분 ${TEST_SECONDS}초"
        RESULTS+=("❌ ${DESCRIPTION}: 실패 (${TEST_MINUTES}분 ${TEST_SECONDS}초)")
        
        # 실패해도 다음 테스트 계속 진행
        echo "⚠️  다음 테스트를 계속 진행합니다..."
    fi
    
    echo ""
    
    # 테스트 간 짧은 휴식 (마지막 테스트 제외)
    if [ $i -lt $((${#TEST_EXPERIMENTS[@]} - 1)) ]; then
        echo "⏸️  다음 테스트 준비 중... (10초 대기)"
        
        # GPU 메모리 정리
        python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
        
        sleep 10
        echo ""
    fi
done

# 전체 테스트 시간 계산
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
TOTAL_MINUTES=$((TOTAL_TIME / 60))
TOTAL_SECONDS=$((TOTAL_TIME % 60))

echo "🎉 모든 1 Epoch 테스트 완료!"
echo "================================"
echo "⏰ 종료 시간: $(date)"
echo "⏱️ 총 소요 시간: ${TOTAL_MINUTES}분 ${TOTAL_SECONDS}초"
echo ""

echo "📊 테스트 결과 요약:"
echo "-------------------"
for result in "${RESULTS[@]}"; do
    echo "$result"
done

echo ""
echo "💡 다음 단계:"
echo "1. 모든 테스트가 성공하면 → 전체 실험 실행 (./run_sequential_5_experiments.sh)"
echo "2. 일부 테스트 실패 시 → 해당 설정 파일 디버깅 후 재시도"
echo "3. GPU 메모리 부족 시 → 배치 크기를 줄여서 재테스트"

echo ""
echo "🔍 GPU 최종 상태:"
nvidia-smi --query-gpu=memory.free,memory.used --format=csv,noheader,nounits

echo ""
echo "✨ 1 Epoch 빠른 테스트 완료!"
echo "   성공한 모델들로 전체 실험을 진행하세요!"
