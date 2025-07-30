#!/bin/bash
# RTX 3090 맞춤 7개 주요 모델 실헐 스크립트 (mT5 XLSum 3단계 + 기존 4개)
# 사용법: bash run_main_7_experiments.sh [-1]
# -1 옵션: 1에포크만 실행 (빠른 테스트용)

set -e

# -1 옵션 처리 (1에포크 모드)
ONE_EPOCH_MODE=false
if [[ "$1" == "-1" ]]; then
    ONE_EPOCH_MODE=true
    echo "🚀 1에포크 모드 활성화: 빠른 테스트용"
fi

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color
# 실험 시작 시간
START_TIME=$(date +%s)
START_TIME_STR=$(date '+%Y-%m-%d %H:%M:%S')

# 1에포크 모드에 따른 메시지 조정
if [[ "$ONE_EPOCH_MODE" == "true" ]]; then
    echo -e "${CYAN}🚀 7개 RTX 3090 최적화 실험 (1에포크 빠른 테스트)${NC}"
    echo -e "${WHITE}════════════════════════════════════════════════════════${NC}"
    echo -e "⏰ 시작 시간: ${START_TIME_STR}"
    echo -e "🖥️  RTX 3090 24GB 최적화 실험 (1에포크 모드)"
    echo -e "⏱️  예상 소요 시간: 30-45분 (1에포크 모드 - 빠른 테스트)"
    echo -e "📝 방법: 사용법 - bash run_main_7_experiments.sh -1"
    else
    echo -e "${CYAN}🚀 7개 RTX 3090 최적화 실험 시작 (mT5 XLSum 3단계 + 고성능 4개)${NC}"
    echo -e "${WHITE}════════════════════════════════════════════════════════${NC}"
    echo -e "⏰ 시작 시간: ${START_TIME_STR}"
    echo -e "🖥️  RTX 3090 24GB 최적화 실헐"
    echo -e "⏱️  예상 소요 시간: 6.5-7.5시간 (mT5 3단계: 3.5시간 + RTX3090 고성능 4개: 3-4시간)"
    echo -e "💪 성능 바른: 안전모드 제거, RTX 3090 24GB 최대 활용"
    echo -e "🎯 mT5 XLSum 목표: ROUGE-1 25%+ 달성 (현재 10.23%에서 150% 향상)"
    echo -e "📝 방법: 사용법 - bash run_main_7_experiments.sh"
    fi
# 로그 디렉토리 생성
LOG_DIR="logs/main_experiments_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# 실험 목록 (mT5 XLSum 3개 + RTX 3090 최적화 4개 = 총 7개)
declare -a experiments=(
    # 🔥 mT5 XLSum 성능 개선 3단계 (RTX 3090 최적화)
    "01_mt5_xlsum_optimized_v2.yaml:🔥_mT5_XLSum_1단계_RTX3090_최적화:45분"
    "01_mt5_xlsum_korean_adapted.yaml:🔧_mT5_XLSum_2단계_한국어_도메인_적응:1시간"
    "01_mt5_xlsum_ultimate.yaml:🚀_mT5_XLSum_3단계_극한_최적화_세계대회도전:1.5시간"
    
    # 💪 RTX 3090 고성능 최적화 4개 (안전모드 제거)
    "02_eenzeenee_t5_rtx3090.yaml:💪_eenzeenee_T5_RTX3090_고성능_최적화:1시간"
    "01_baseline_kobart_rtx3090.yaml:💪_KoBART_RTX3090_고성능_최적화:1시간"
    "03_high_learning_rate_rtx3090.yaml:💪_극한_고성능_학습률_RTX3090:1시간"
    "04_batch_optimization_rtx3090.yaml:💪_배치_극한_최적화_RTX3090:1.5시간"
)

# GPU 정보 출력 함수
print_gpu_info() {
    echo -e "${BLUE}📊 실험 전 GPU 상태:${NC}"
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader || echo "GPU 정보를 가져올 수 없습니다."
    echo
}

# GPU 메모리 정리 함수
cleanup_gpu() {
    echo -e "${YELLOW}🧹 GPU 메모리 정리 및 캐시 클리어${NC}"
    
    # Python에서 GPU 메모리 정리
    python3 -c "
import torch
import gc
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
gc.collect()
" 2>/dev/null || true
    
    echo "✅ GPU 메모리 정리 완료"
    
    # Python 가비지 컬렉션
    python3 -c "import gc; gc.collect()" 2>/dev/null || true
    echo "✅ Python 가비지 컬렉션 완료"
    
    # 시스템 캐시 정리 (권한이 있는 경우)
    if [ -w /proc/sys/vm/drop_caches ]; then
        sync && echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
        echo "✅ 시스템 캐시 정리 완료"
    else
        echo "⚠️  시스템 캐시 정리 권한이 없습니다 (정상)"
    fi
    
    echo "✅ 준비 완료!"
    echo
}

# 초기 GPU 상태 확인
print_gpu_info

# 실행 결과 추적
declare -a results=()
TOTAL_EXPERIMENTS=${#experiments[@]}
COMPLETED=0

# 각 실험 실행
for i in "${!experiments[@]}"; do
    IFS=':' read -r config_file exp_name exp_time <<< "${experiments[$i]}"
    
    EXPERIMENT_NUM=$((i + 1))
    echo -e "${PURPLE}🔬 실험 ${EXPERIMENT_NUM}/${TOTAL_EXPERIMENTS}: ${exp_name}${NC}"
    echo -e "${WHITE}📄 설정 파일: config/experiments/${config_file}${NC}"
    echo -e "${WHITE}⏱️  예상 시간: ${exp_time}${NC}"
    echo -e "${WHITE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # GPU 상태 확인
    print_gpu_info
    
    # 실험 시작 시간
    EXP_START_TIME=$(date +%s)
    EXP_START_TIME_STR=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${GREEN}🚀 실험 시작: ${EXP_START_TIME_STR}${NC}"
    echo
    
    # 실험별 로그 파일
    LOG_FILE="${LOG_DIR}/experiment_${EXPERIMENT_NUM}_${exp_name// /_}.log"
    
    # 실험 실행 (1에포크 모드 옵션 처리)
    EXPERIMENT_CMD="python code/auto_experiment_runner.py --config config/experiments/${config_file}"
    
    # 1에포크 모드일 때 --one-epoch 옵션 추가
    if [[ "$ONE_EPOCH_MODE" == "true" ]]; then
        EXPERIMENT_CMD="$EXPERIMENT_CMD --one-epoch"
        echo -e "${YELLOW}1에포크 모드로 실행 중...${NC}"
    fi
    
    if eval "$EXPERIMENT_CMD > ${LOG_FILE} 2>&1"; then
        
        EXP_END_TIME=$(date +%s)
        EXP_DURATION=$((EXP_END_TIME - EXP_START_TIME))
        EXP_DURATION_MIN=$((EXP_DURATION / 60))
        EXP_DURATION_SEC=$((EXP_DURATION % 60))
        
        echo -e "${GREEN}✅ 실험 ${EXPERIMENT_NUM} 완료!${NC}"
        echo -e "⏱️  소요 시간: ${EXP_DURATION_MIN}분 ${EXP_DURATION_SEC}초"
        results+=("✅ ${exp_name}: ${EXP_DURATION_MIN}분 ${EXP_DURATION_SEC}초")
        COMPLETED=$((COMPLETED + 1))
    else
        echo -e "${RED}❌ 실험 ${EXPERIMENT_NUM} 실패!${NC}"
        echo -e "📋 로그 확인: ${LOG_FILE}"
        results+=("❌ ${exp_name}: 실패")
        
        # 에러 로그 출력
        echo -e "${YELLOW}최근 에러 로그:${NC}"
        tail -n 20 "${LOG_FILE}" | grep -E "(ERROR|Error|error|Traceback|Exception)" || true
    fi
    
    echo
    
    # 다음 실험 전 대기 및 정리 (마지막 실험 제외)
    if [ $i -lt $((TOTAL_EXPERIMENTS - 1)) ]; then
        echo -e "${YELLOW}⏸️  다음 실험 준비 중... (60초 대기)${NC}"
        cleanup_gpu
        sleep 60
    fi
done

# 전체 소요 시간 계산
END_TIME=$(date +%s)
END_TIME_STR=$(date '+%Y-%m-%d %H:%M:%S')
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MINUTES=$(((TOTAL_DURATION % 3600) / 60))
# 결과 요약
echo
if [[ "$ONE_EPOCH_MODE" == "true" ]]; then
    echo -e "${CYAN}🎉 7개 실헐 모두 완료! (1에포크 빠른 테스트)${NC}"
    echo -e "${WHITE}════════════════════════════════════════════════════════${NC}"
else
    echo -e "${CYAN}🎉 7개 실헐 모두 완료! (mT5 XLSum 3단계 + RTX3090 고성능 4개)${NC}"
    echo -e "${WHITE}════════════════════════════════════════════════════════${NC}"
fi
echo -e "⏰ 종료 시간: ${END_TIME_STR}"
echo -e "⏱️  총 소요 시간: ${TOTAL_HOURS}시간 ${TOTAL_MINUTES}분"
echo
echo -e "${BLUE}📊 실험 결과 요약:${NC}"
echo -e "${WHITE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
for result in "${results[@]}"; do
    echo -e "  ${result}"
done
echo
echo -e "${GREEN}📁 결과 파일 위치:${NC}"
echo -e "  - 실험 로그: ${LOG_DIR}/"
echo -e "  - 실험 요약: outputs/auto_experiments/experiment_summary.json"
echo -e "  - 개별 결과: outputs/auto_experiments/experiments/"
echo -e "  - 모델 체크포인트: outputs/auto_experiments/"
echo -e "  - WandB 프로젝트: https://wandb.ai/lyjune37-juneictlab/nlp-5"
echo

# 최종 GPU 상태
echo -e "${BLUE}🔍 GPU 최종 상태:${NC}"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader || echo "GPU 정보를 가져올 수 없습니다."

# 실험 요약 파일 생성
SUMMARY_FILE="${LOG_DIR}/experiment_summary.txt"
{
    echo "5개 주요 모델 실험 요약 (수정된 버전)"
    echo "======================"
    echo "실행 시간: ${START_TIME_STR} ~ ${END_TIME_STR}"
    echo "총 소요 시간: ${TOTAL_HOURS}시간 ${TOTAL_MINUTES}분"
    echo
    echo "실험 결과:"
    for result in "${results[@]}"; do
        echo "  ${result}"
    done
} > "${SUMMARY_FILE}"

echo
echo -e "${WHITE}📝 실험 요약 파일 저장: ${SUMMARY_FILE}${NC}"
echo
echo -e "${CYAN}✨ 5개 주요 모델 실험 완료!${NC}"
echo -e "   ${COMPLETED}/${TOTAL_EXPERIMENTS} 실험 성공"
echo -e "   WandB에서 상세 결과를 확인하세요."
