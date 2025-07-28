#!/bin/bash
# RTX 3090 24GB 서버 최적화 5개 실험 순차 실행 스크립트

set -e  # 오류 시 중단

echo "🚀 RTX 3090 24GB 최적화 5개 실험 순차 실행"
echo "=========================================="
echo ""

# 실험 목록 정의
EXPERIMENTS=(
    "01_mt5_xlsum.yaml:mT5_XL-Sum_대형모델"
    "02_eenzeenee_t5.yaml:eenzeenee_T5_한국어모델"  
    "01_baseline_kobart.yaml:KoBART_베이스라인"
    "03_high_learning_rate.yaml:고성능_학습률"
    "04_batch_optimization.yaml:배치_최적화"
)

# 시작 시간 기록
START_TIME=$(date +%s)
echo "⏰ 실험 시작 시간: $(date)"
echo ""

# 실험 결과 저장 배열
declare -a RESULTS

# 각 실험 순차 실행
for i in "${!EXPERIMENTS[@]}"; do
    IFS=':' read -r YAML_FILE DESCRIPTION <<< "${EXPERIMENTS[$i]}"
    EXPERIMENT_NUM=$((i + 1))
    
    echo "🔥 실험 ${EXPERIMENT_NUM}/5: ${DESCRIPTION}"
    echo "📄 설정 파일: ${YAML_FILE}"
    echo "----------------------------------------"
    
    # GPU 메모리 상태 확인
    echo "📊 실험 전 GPU 상태:"
    nvidia-smi --query-gpu=memory.free,memory.used,memory.total --format=csv,noheader,nounits
    echo ""
    
    # 실험 실행
    EXPERIMENT_START=$(date +%s)
    
    if python code/auto_experiment_runner.py --experiment "${YAML_FILE}"; then
        EXPERIMENT_END=$(date +%s)
        EXPERIMENT_TIME=$((EXPERIMENT_END - EXPERIMENT_START))
        EXPERIMENT_MINUTES=$((EXPERIMENT_TIME / 60))
        
        echo "✅ 실험 ${EXPERIMENT_NUM} 완료!"
        echo "⏱️ 소요 시간: ${EXPERIMENT_MINUTES}분 (${EXPERIMENT_TIME}초)"
        RESULTS+=("✅ ${DESCRIPTION}: ${EXPERIMENT_MINUTES}분")
    else
        EXPERIMENT_END=$(date +%s)
        EXPERIMENT_TIME=$((EXPERIMENT_END - EXPERIMENT_START))
        EXPERIMENT_MINUTES=$((EXPERIMENT_TIME / 60))
        
        echo "❌ 실험 ${EXPERIMENT_NUM} 실패!"
        echo "⏱️ 실패까지 시간: ${EXPERIMENT_MINUTES}분"
        RESULTS+=("❌ ${DESCRIPTION}: 실패 (${EXPERIMENT_MINUTES}분)")
        
        # 실패 시에도 계속 진행할지 선택
        echo "⚠️  다음 실험을 계속 진행하시겠습니까? (y/n)"
        read -r CONTINUE
        if [ "$CONTINUE" != "y" ]; then
            echo "실험 중단됨"
            break
        fi
    fi
    
    echo ""
    
    # 실험 간 휴식 (마지막 실험 제외)
    if [ $i -lt $((${#EXPERIMENTS[@]} - 1)) ]; then
        echo "⏸️  다음 실험 준비 중... (30초 대기)"
        echo "🧹 GPU 메모리 정리 및 캐시 클리어"
        
        # Python 캐시 정리
        python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
        
        # 30초 대기
        sleep 30
        echo ""
    fi
done

# 전체 실행 시간 계산
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
TOTAL_HOURS=$((TOTAL_TIME / 3600))
TOTAL_MINUTES=$(((TOTAL_TIME % 3600) / 60))

echo "🎉 모든 실험 완료!"
echo "==============================="
echo "⏰ 종료 시간: $(date)"
echo "⏱️ 총 소요 시간: ${TOTAL_HOURS}시간 ${TOTAL_MINUTES}분"
echo ""

echo "📊 실험 결과 요약:"
echo "-------------------"
for result in "${RESULTS[@]}"; do
    echo "$result"
done

echo ""
echo "📁 결과 파일 위치:"
echo "- 실험 요약: outputs/auto_experiments/experiment_summary.json"
echo "- 개별 결과: outputs/auto_experiments/experiments/"
echo "- WandB 프로젝트: https://wandb.ai/lyjune37-juneictlab/nlp-5"

echo ""
echo "🔍 GPU 최종 상태:"
nvidia-smi --query-gpu=memory.free,memory.used,memory.total --format=csv,noheader,nounits

echo ""
echo "✨ RTX 3090 24GB 최적화 실험 시퀀스 완료!"
