#!/bin/bash
# RTX 3090 극한 최적화 7개 주요 모델 실험 스크립트 (mT5 XLSum 3단계 + 기존 4개)
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

# 🔥 RTX 3090 극한 최적화 벤치마킹 전역 변수
BENCHMARK_LOG="benchmark_$(date +%Y%m%d_%H%M%S).log"
TOTAL_MEMORY_SAVED=0
TOTAL_TIME_SAVED=0

# 🔥 극한 최적화 GPU 메모리 모니터링 함수
enhanced_gpu_monitor() {
    local prefix="$1"
    local gpu_data=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits)
    
    if [ -n "$gpu_data" ]; then
        IFS=',' read -r memory_used memory_total gpu_util temperature <<< "$gpu_data"
        memory_used=$(echo $memory_used | xargs)  # trim whitespace
        memory_total=$(echo $memory_total | xargs)
        gpu_util=$(echo $gpu_util | xargs)
        temperature=$(echo $temperature | xargs)
        
        local memory_percent=$((memory_used * 100 / memory_total))
        local memory_free=$((memory_total - memory_used))
        
        echo -e "${BLUE}📊 $prefix GPU 상태:${NC}"
        echo "  🗜 GPU 메모리: ${memory_used}MB/${memory_total}MB (${memory_percent}%)"
        echo "  🌡️  GPU 온도: ${temperature}°C"
        echo "  ⚡ GPU 활용률: ${gpu_util}%"
        echo "  🔓 사용 가능: ${memory_free}MB"
        
        # 경고 사항 처리
        if [ "$memory_used" -gt 22000 ]; then
            echo -e "  ${RED}⚠️  경고: GPU 메모리 임계 상태 (22GB 초과)${NC}"
            return 1
        elif [ "$memory_used" -gt 20000 ]; then
            echo -e "  ${YELLOW}⚠️  주의: GPU 메모리 높음 (20GB 초과)${NC}"
        elif [ "$memory_used" -lt 5000 ]; then
            echo -e "  ${GREEN}✅ 안전: GPU 메모리 여유량 충분${NC}"
        fi
        
        if [ "$temperature" -gt 80 ]; then
            echo -e "  ${RED}⚠️  경고: GPU 온도 높음 (80°C 초과)${NC}"
        fi
        
        return 0
    else
        echo -e "${RED}❌ GPU 정보를 가져올 수 없습니다${NC}"
        return 1
    fi
}

# 🔥 스마트 대기 함수 (동적 대기 시간 최적화)
smart_wait() {
    local target_memory=${1:-5000}  # 기본 5GB 아래로 대기
    local max_wait_time=${2:-300}   # 최대 5분 대기
    local wait_start=$(date +%s)
    
    echo -e "${YELLOW}⏳ 스마트 대기: GPU 메모리 ${target_memory}MB 아래로 대기 중...${NC}"
    
    while true; do
        local current_memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | xargs)
        local current_wait_time=$(($(date +%s) - wait_start))
        
        if [ "$current_memory" -le "$target_memory" ]; then
            echo -e "${GREEN}✅ 대기 완료: GPU 메모리 ${current_memory}MB (${current_wait_time}초 대기)${NC}"
            TOTAL_TIME_SAVED=$((TOTAL_TIME_SAVED + 60 - current_wait_time))
            break
        fi
        
        if [ "$current_wait_time" -ge "$max_wait_time" ]; then
            echo -e "${YELLOW}⚠️  대기 시간 초과: 강제 진행 (${max_wait_time}초 대기)${NC}"
            break
        fi
        
        echo "  🔄 대기 중... 현재: ${current_memory}MB (목표: ${target_memory}MB 아래, ${current_wait_time}/${max_wait_time}초)"
        sleep 10
    done
}

# 🔥 실험 시간 추적 함수
track_experiment_time() {
    local exp_name="$1"
    local start_time="$2"
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local duration_min=$((duration / 60))
    local duration_sec=$((duration % 60))
    
    echo -e "${GREEN}📈 실험 '$exp_name': ${duration_min}분 ${duration_sec}초 소요${NC}"
    
    # 벤치마크 로그에 기록
    echo "$(date '+%Y-%m-%d %H:%M:%S') | $exp_name | ${duration_min}m ${duration_sec}s" >> "$BENCHMARK_LOG"
    
    return $duration
}

# 🔥 에러 처리 및 안전 폴백 함수
handle_experiment_error() {
    local exp_name="$1"
    local log_file="$2"
    local exp_num="$3"
    
    echo -e "${RED}❌ 실험 $exp_num 실패: $exp_name${NC}"
    echo -e "${YELLOW}📄 로그 파일: $log_file${NC}"
    
    # 에러 로그 분석 및 출력
    echo -e "${YELLOW}🔍 최근 에러 로그:${NC}"
    if [ -f "$log_file" ]; then
        tail -n 30 "$log_file" | grep -E "(ERROR|Error|error|Traceback|Exception|CUDA|OutOfMemoryError|RuntimeError)" | tail -n 10 || echo "에러 로그를 찾을 수 없습니다."
    fi
    
    # GPU 메모리 과부하 감지
    local current_memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | xargs)
    if [ "$current_memory" -gt 20000 ]; then
        echo -e "${RED}⚠️  GPU 메모리 과부하 감지! 긴급 정리 실행...${NC}"
        cleanup_gpu_emergency
    fi
    
    # 벤치마크 로그에 에러 기록
    echo "$(date '+%Y-%m-%d %H:%M:%S') | $exp_name | ERROR | Memory: ${current_memory}MB" >> "$BENCHMARK_LOG"
}

# 🔥 긴급 GPU 리셋 함수
cleanup_gpu_emergency() {
    echo -e "${RED}🚑 긴급 GPU 메모리 리셋 실행 중...${NC}"
    
    # 강제 CUDA 프로세스 종료
    pkill -f "python.*cuda" 2>/dev/null || true
    pkill -f "python.*torch" 2>/dev/null || true
    
    # 강제 메모리 정리
    python3 -c "
import torch
import gc
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
gc.collect()
" 2>/dev/null || true
    
    sleep 15  # 긴급 대기
    echo -e "${GREEN}✅ 긴급 GPU 리셋 완료${NC}"
}
# 실험 시작 시간
START_TIME=$(date +%s)
START_TIME_STR=$(date '+%Y-%m-%d %H:%M:%S')

# 🔥 벤치마크 로그 초기화
echo "=== RTX 3090 극한 최적화 벤치마크 로그 ==="> "$BENCHMARK_LOG"
echo "시작 시간: $START_TIME_STR" >> "$BENCHMARK_LOG"
echo "" >> "$BENCHMARK_LOG"

# 1에포크 모드에 따른 메시지 조정
if [[ "$ONE_EPOCH_MODE" == "true" ]]; then
    echo -e "${CYAN}🚀 7개 RTX 3090 최적화 실험 (1에포크 빠른 테스트)${NC}"
    echo -e "${WHITE}════════════════════════════════════════════════════════${NC}"
    echo -e "⏰ 시작 시간: ${START_TIME_STR}"
    echo -e "🖥️  RTX 3090 24GB 최적화 실험 (1에포크 모드)"
    echo -e "⏱️  예상 소요 시간: 25-35분 (1에포크 모드 - 극한 최적화 빠른 테스트)"
    echo -e "📝 방법: 사용법 - bash run_main_7_experiments.sh -1"
else
    echo -e "${CYAN}🚀 7개 RTX 3090 최적화 실험 시작 (mT5 XLSum 3단계 + 고성능 4개)${NC}"
    echo -e "${WHITE}════════════════════════════════════════════════════════${NC}"
    echo -e "⏰ 시작 시간: ${START_TIME_STR}"
    echo -e "🖥️  RTX 3090 24GB 최적화 실험"
    echo -e "⏱️  예상 소요 시간: 4.5-5.5시간 (mT5 3단계: 2.5시간 + RTX3090 극한최적화 4개: 2-3시간)"
    echo -e "💪 성능 반영: 안전모드 제거, RTX 3090 24GB 최대 활용"
    echo -e "🎯 mT5 XLSum 목표: ROUGE-1 25%+ 달성 (현재 10.23%에서 150% 향상)"
    echo -e "📝 방법: 사용법 - bash run_main_7_experiments.sh"
fi

# 로그 디렉토리 생성
LOG_DIR="logs/main_experiments_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# 실험 목록 (mT5 XLSum 3개 + RTX 3090 극한 최적화 4개 = 총 7개)
declare -a experiments=(
    # 🔥 mT5 XLSum 성능 개선 3단계 (RTX 3090 극한 최적화)
    "01_mt5_xlsum_optimized_v2.yaml:🔥_mT5_XLSum_1단계_RTX3090_극한최적화:35분"
    "01_mt5_xlsum_korean_adapted.yaml:🔧_mT5_XLSum_2단계_한국어_도메인_적응:45분"
    "01_mt5_xlsum_ultimate.yaml:🚀_mT5_XLSum_3단계_극한최적화_세계대회도전:55분"

    # 💪 RTX 3090 극한 최적화 4개 (안전모드 제거)
    "02_eenzeenee_t5_rtx3090.yaml:💪_eenzeenee_T5_RTX3090_극한최적화:40분"
    "01_baseline_kobart_rtx3090.yaml:💪_KoBART_RTX3090_극한최적화:45분"
    "03_high_learning_rate_rtx3090.yaml:💪_극한_고성능_학습률_RTX3090:35분"
    "04_batch_optimization_rtx3090.yaml:💪_배치_극한최적화_RTX3090:40분"
)

# GPU 정보 출력 함수
print_gpu_info() {
    echo -e "${BLUE}📊 실험 전 GPU 상태:${NC}"
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader || echo "GPU 정보를 가져올 수 없습니다."
    echo
}

# RTX 3090 극한 최적화용 GPU 메모리 모니터링 및 정리 함수
cleanup_gpu() {
    echo -e "${YELLOW}🧹 RTX 3090 극한 최적화 GPU 메모리 정리 및 모니터링${NC}"

    # GPU 상태 확인
    echo -e "${BLUE}📊 정리 전 GPU 상태:${NC}"
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | while IFS=',' read -r used total util; do
        used=$(echo "$used" | tr -d ' ,')
        total=$(echo "$total" | tr -d ' ,')
        util=$(echo "$util" | tr -d ' ,')
        echo "GPU 메모리: ${used}MB/${total}MB (사용률: ${util}%)"
        if [ "$used" -gt 22000 ]; then
            echo -e "${RED}⚠️  임계 상태: GPU 메모리가 22GB 초과 (${used}MB)${NC}"
        fi
    done

    # Python에서 GPU 메모리 정리
    python3 -c "
import torch
import gc
if torch.cuda.is_available():
    # 메모리 정리 전 상태 출력
    memory_before = torch.cuda.memory_allocated() / (1024**3)
    print(f'정리 전 GPU 메모리: {memory_before:.2f}GB')
    
    # 메모리 정리
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # 정리 후 상태 출력
    memory_after = torch.cuda.memory_allocated() / (1024**3)
    print(f'정리 후 GPU 메모리: {memory_after:.2f}GB')
    print(f'해제된 메모리: {memory_before - memory_after:.2f}GB')
gc.collect()
" 2>/dev/null || true

    echo "✅ GPU 메모리 정리 완료"

    # Python 가비지 컬렉션
    python3 -c "import gc; gc.collect()" 2>/dev/null || true
    echo "✅ Python 가비지 컬렉션 완료"

    # 시스템 캐시 정리 (권한이 있는 경우)
    if [ -w /proc/sys/vm/drop_caches ]; then
        if sync; then
            echo 3 >/proc/sys/vm/drop_caches 2>/dev/null || true
        fi
        echo "✅ 시스템 캐시 정리 완료"
    else
        echo "⚠️  시스템 캐시 정리 권한이 없습니다 (정상)"
    fi

    # 정리 후 GPU 상태 재확인
    echo -e "${BLUE}📊 정리 후 GPU 상태:${NC}"
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | while IFS=',' read -r used total util; do
        used=$(echo "$used" | tr -d ' ,')
        total=$(echo "$total" | tr -d ' ,')
        util=$(echo "$util" | tr -d ' ,')
        echo "GPU 메모리: ${used}MB/${total}MB (사용률: ${util}%)"
        if [ "$used" -lt 5000 ]; then
            echo -e "${GREEN}✅ GPU 메모리가 안전한 수준으로 정리됨${NC}"
        fi
        done
        
        echo "✅ RTX 3090 극한 최적화 준비 완료!"
        echo
        }
# 초기 GPU 상태 확인 (향상된 모니터링)
enhanced_gpu_monitor "실험 전"

# 실행 결과 추적
declare -a results=()
TOTAL_EXPERIMENTS=${#experiments[@]}
COMPLETED=0

# 각 실험 실행
for i in "${!experiments[@]}"; do
    IFS=':' read -r config_file exp_name exp_time <<<"${experiments[$i]}"

    EXPERIMENT_NUM=$((i + 1))
    echo -e "${PURPLE}🔬 실험 ${EXPERIMENT_NUM}/${TOTAL_EXPERIMENTS}: ${exp_name}${NC}"
    echo -e "${WHITE}📄 설정 파일: config/experiments/${config_file}${NC}"
    echo -e "${WHITE}⏱️  예상 시간: ${exp_time}${NC}"
    echo -e "${WHITE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    # GPU 상태 확인 (향상된 모니터링)
    enhanced_gpu_monitor "실험 $EXPERIMENT_NUM 시작 전"

    # 실험 시작 시간
    EXP_START_TIME=$(date +%s)
    EXP_START_TIME_STR=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${GREEN}🚀 실험 시작: ${EXP_START_TIME_STR}${NC}"
    echo

    # 파일명에서 특수문자 제거 (공백과 이모지 대체)
    exp_name_clean="${exp_name// /_}"
    exp_name_clean="${exp_name_clean//🔥/_}"
    exp_name_clean="${exp_name_clean//🔧/_}"
    exp_name_clean="${exp_name_clean//🚀/_}"
    exp_name_clean="${exp_name_clean//💪/_}"
    LOG_FILE="${LOG_DIR}/experiment_${EXPERIMENT_NUM}_${exp_name_clean}.log"
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
        
        # 향상된 실험 시간 추적
        track_experiment_time "$exp_name" "$EXP_START_TIME"
        local actual_duration=$?
        
        EXP_DURATION_MIN=$((EXP_DURATION / 60))
        EXP_DURATION_SEC=$((EXP_DURATION % 60))

        echo -e "${GREEN}✅ 실험 ${EXPERIMENT_NUM} 완료!${NC}"
        echo -e "⏱️  소요 시간: ${EXP_DURATION_MIN}분 ${EXP_DURATION_SEC}초"
        
        # 벤치마크 기록
        results+=("✅ ${exp_name}: ${EXP_DURATION_MIN}분 ${EXP_DURATION_SEC}초")
        COMPLETED=$((COMPLETED + 1))
        
        # GPU 상태 최종 확인
        enhanced_gpu_monitor "실험 $EXPERIMENT_NUM 완료 후"
    else
        # 향상된 에러 처리
        handle_experiment_error "$exp_name" "$LOG_FILE" "$EXPERIMENT_NUM"
        results+=("❌ ${exp_name}: 실패")
    fi

    echo

    # 다음 실험 전 스마트 대기 및 정리 (마지막 실험 제외)
    if [ "$i" -lt $((TOTAL_EXPERIMENTS - 1)) ]; then
        echo -e "${YELLOW}⏸️  다음 실험 준비 중... (스마트 대기)${NC}"
        cleanup_gpu
        smart_wait 5000 240  # 5GB 아래로 대기, 최대 4분
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
    echo -e "${CYAN}🎉 7개 실험 모두 완료! (1에포크 빠른 테스트)${NC}"
    echo -e "${WHITE}════════════════════════════════════════════════════════${NC}"
else
    echo -e "${CYAN}🎉 7개 실험 모두 완료! (mT5 XLSum 3단계 + RTX3090 고성능 4개)${NC}"
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

# 최종 GPU 상태 (향상된 모니터링)
echo -e "${BLUE}🔍 GPU 최종 상태:${NC}"
enhanced_gpu_monitor "모든 실험 완료 후"

# 📈 최적화 성과 리포트
echo
echo -e "${CYAN}📈 RTX 3090 극한 최적화 성과:${NC}"
echo -e "${WHITE}──────────────────────────────────────${NC}"
printf "🗜 총 메모리 절약: %.2fGB\n" $TOTAL_MEMORY_SAVED
echo -e "⏱️  총 시간 절약: ${TOTAL_TIME_SAVED}초 (${TOTAL_TIME_SAVED} / 60 = $((TOTAL_TIME_SAVED / 60))분)"
echo -e "🏆 성공률: ${COMPLETED}/${TOTAL_EXPERIMENTS} ($((COMPLETED * 100 / TOTAL_EXPERIMENTS))%)"
echo

# 실험 요약 파일 생성 (벤치마크 정보 포함)
SUMMARY_FILE="${LOG_DIR}/experiment_summary.txt"
{
    echo "7개 주요 모델 실험 요약 (mT5 XLSum 3단계 + RTX3090 극한최적화 4개)"
    echo "======================"
    echo "실행 시간: ${START_TIME_STR} ~ ${END_TIME_STR}"
    echo "총 소요 시간: ${TOTAL_HOURS}시간 ${TOTAL_MINUTES}분"
    echo
    echo "실험 결과:"
    for result in "${results[@]}"; do
        echo "  ${result}"
    done
    echo
    echo "RTX 3090 극한 최적화 성과:"
    printf "  총 메모리 절약: %.2fGB\n" $TOTAL_MEMORY_SAVED
    echo "  총 시간 절약: ${TOTAL_TIME_SAVED}초 ($((TOTAL_TIME_SAVED / 60))분)"
    echo "  성공률: ${COMPLETED}/${TOTAL_EXPERIMENTS} ($((COMPLETED * 100 / TOTAL_EXPERIMENTS))%)"
    echo
    echo "벤치마크 로그: $BENCHMARK_LOG"
} >"${SUMMARY_FILE}"

echo
echo -e "${WHITE}📝 실험 요약 파일 저장: ${SUMMARY_FILE}${NC}"
echo
echo -e "${CYAN}✨ 7개 주요 모델 실험 완료! (RTX 3090 극한 최적화)${NC}"
echo -e "   ${COMPLETED}/${TOTAL_EXPERIMENTS} 실험 성공 (성공률: $((COMPLETED * 100 / TOTAL_EXPERIMENTS))%)"
printf "   📈 메모리 절약: %.2fGB, 시간 절약: %d분\n" $TOTAL_MEMORY_SAVED $((TOTAL_TIME_SAVED / 60))
echo -e "   🏆 최적화 성과를 WandB에서 상세 결과를 확인하세요."
echo -e "   📄 벤치마크 상세: $BENCHMARK_LOG"
