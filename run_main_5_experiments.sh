#!/bin/bash
# 5개 주요 모델 정상 실험 순차 실행 스크립트 (RTX 3090 24GB 최적화)

set -e  # 오류 시 중단

# .env 파일에서 환경 변수 로드
if [ -f .env ]; then
    echo "🔑 .env 파일에서 환경 변수 로드 중..."
    set -a  # 모든 변수를 export
    source .env
    set +a  # export 모드 해제
    echo "✅ WandB API 키 로드 완료"
    echo "   Entity: $WANDB_ENTITY"
    echo "   Project: $WANDB_PROJECT"
else
    echo "⚠️  .env 파일을 찾을 수 없습니다!"
    echo "👉 .env.template을 복사하여 .env 파일을 생성하세요."
    exit 1
fi

# 추가 환경 변수 설정
export TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS="ignore"
echo "🚀 5개 주요 모델 정상 실험 순차 실행"
echo "======================================="
echo "📋 실험 목록:"
echo "  1. mT5 XL-Sum (1.2B parameters)"
echo "  2. eenzeenee T5 Korean"  
echo "  3. KoBART Baseline"
echo "  4. High Learning Rate"
echo "  5. Batch Optimization"
echo ""

# 실험 목록 정의 (설정 파일과 설명)
EXPERIMENTS=(
    "config/experiments/01_mt5_xlsum.yaml:mT5_XL-Sum_대형모델:3-4시간"
    "config/experiments/02_eenzeenee_t5.yaml:eenzeenee_T5_한국어:2-3시간"  
    "config/experiments/01_baseline_kobart.yaml:KoBART_베이스라인:1-2시간"
    "config/experiments/03_high_learning_rate.yaml:고성능_학습률:1시간"
    "config/experiments/04_batch_optimization.yaml:배치_최적화:1-2시간"
)

# 시작 시간 기록
START_TIME=$(date +%s)
echo "⏰ 실험 시작 시간: $(date)"
echo "📊 예상 총 소요 시간: 약 8-12시간"
echo ""

# 실험 결과 저장 배열
declare -a RESULTS

# 로그 디렉토리 생성
LOG_DIR="logs/main_experiments_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "📁 로그 디렉토리: $LOG_DIR"
echo ""

# 각 실험 순차 실행
for i in "${!EXPERIMENTS[@]}"; do
    IFS=':' read -r YAML_FILE DESCRIPTION EXPECTED_TIME <<< "${EXPERIMENTS[$i]}"
    EXPERIMENT_NUM=$((i + 1))
    
    echo "🔬 실험 ${EXPERIMENT_NUM}/5: ${DESCRIPTION}"
    echo "📄 설정 파일: ${YAML_FILE}"
    echo "⏱️  예상 시간: ${EXPECTED_TIME}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # GPU 메모리 상태 확인
    echo "📊 실험 전 GPU 상태:"
    nvidia-smi --query-gpu=name,memory.free,memory.used,memory.total,temperature.gpu,utilization.gpu --format=csv,noheader
    echo ""
    
    # 실험별 로그 파일
    LOG_FILE="${LOG_DIR}/experiment_${EXPERIMENT_NUM}_${DESCRIPTION// /_}.log"
    
    # 실험 실행
    EXPERIMENT_START=$(date +%s)
    echo "🚀 실험 시작: $(date)"
    
    # 실험 실행 (로그 파일에 저장하면서 화면에도 출력)
    if python code/auto_experiment_runner.py --config "${YAML_FILE}" 2>&1 | tee "$LOG_FILE"; then
        EXPERIMENT_END=$(date +%s)
        EXPERIMENT_TIME=$((EXPERIMENT_END - EXPERIMENT_START))
        EXPERIMENT_HOURS=$((EXPERIMENT_TIME / 3600))
        EXPERIMENT_MINUTES=$(((EXPERIMENT_TIME % 3600) / 60))
        
        echo ""
        echo "✅ 실험 ${EXPERIMENT_NUM} 완료!"
        echo "⏱️  소요 시간: ${EXPERIMENT_HOURS}시간 ${EXPERIMENT_MINUTES}분"
        RESULTS+=("✅ ${DESCRIPTION}: ${EXPERIMENT_HOURS}시간 ${EXPERIMENT_MINUTES}분")
    else
        EXPERIMENT_END=$(date +%s)
        EXPERIMENT_TIME=$((EXPERIMENT_END - EXPERIMENT_START))
        EXPERIMENT_HOURS=$((EXPERIMENT_TIME / 3600))
        EXPERIMENT_MINUTES=$(((EXPERIMENT_TIME % 3600) / 60))
        
        echo ""
        echo "❌ 실험 ${EXPERIMENT_NUM} 실패!"
        echo "⏱️  실패까지 시간: ${EXPERIMENT_HOURS}시간 ${EXPERIMENT_MINUTES}분"
        echo "📄 로그 파일 확인: $LOG_FILE"
        RESULTS+=("❌ ${DESCRIPTION}: 실패 (${EXPERIMENT_HOURS}시간 ${EXPERIMENT_MINUTES}분)")
        
        # 실패 시에도 계속 진행
        echo "⚠️  다음 실험을 계속 진행합니다..."
    fi
    
    echo ""
    
    # 실험 간 휴식 (마지막 실험 제외)
    if [ $i -lt $((${#EXPERIMENTS[@]} - 1)) ]; then
        echo "⏸️  다음 실험 준비 중... (60초 대기)"
        echo "🧹 GPU 메모리 정리 및 캐시 클리어"
        
        # Python으로 GPU 메모리 정리
        python -c "
import torch
import gc
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print('✅ GPU 메모리 정리 완료')
gc.collect()
print('✅ Python 가비지 컬렉션 완료')
" 2>/dev/null || true
        
        # 시스템 캐시 정리 (권한이 있는 경우)
        sync && echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
        
        # 60초 대기
        for ((j=60; j>0; j--)); do
            echo -ne "\r⏳ ${j}초 남음..."
            sleep 1
        done
        echo -e "\r✅ 준비 완료!     "
        echo ""
    fi
done

# 전체 실행 시간 계산
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
TOTAL_HOURS=$((TOTAL_TIME / 3600))
TOTAL_MINUTES=$(((TOTAL_TIME % 3600) / 60))

echo ""
echo "🎉 모든 실험 완료!"
echo "════════════════════════════════════════"
echo "⏰ 종료 시간: $(date)"
echo "⏱️  총 소요 시간: ${TOTAL_HOURS}시간 ${TOTAL_MINUTES}분"
echo ""

echo "📊 실험 결과 요약:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
for result in "${RESULTS[@]}"; do
    echo "  $result"
done
echo ""

echo "📁 결과 파일 위치:"
echo "  - 실험 로그: ${LOG_DIR}/"
echo "  - 실험 요약: outputs/auto_experiments/experiment_summary.json"
echo "  - 개별 결과: outputs/auto_experiments/experiments/"
echo "  - 모델 체크포인트: outputs/auto_experiments/"
echo "  - WandB 프로젝트: https://wandb.ai/lyjune37-juneictlab/nlp-5"
echo ""

echo "🔍 GPU 최종 상태:"
nvidia-smi --query-gpu=name,memory.free,memory.used,memory.total,temperature.gpu,utilization.gpu --format=csv,noheader
echo ""

# 실험 요약 파일 생성
SUMMARY_FILE="${LOG_DIR}/experiment_summary.txt"
{
    echo "5개 주요 모델 실험 요약"
    echo "======================"
    echo "실행 시간: $(date -d @$START_TIME) ~ $(date -d @$END_TIME)"
    echo "총 소요 시간: ${TOTAL_HOURS}시간 ${TOTAL_MINUTES}분"
    echo ""
    echo "실험 결과:"
    for result in "${RESULTS[@]}"; do
        echo "  $result"
    done
} > "$SUMMARY_FILE"

echo "📝 실험 요약 파일 저장: $SUMMARY_FILE"
echo ""
echo "✨ 5개 주요 모델 정상 실험 완료!"
echo "   WandB에서 상세 결과를 확인하세요."
