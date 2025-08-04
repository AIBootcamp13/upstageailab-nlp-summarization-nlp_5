#!/bin/bash

# 🚀 GPU 메모리 최적화 스크립트
# AIStages 서버에서 불필요한 GPU 사용을 정리하여 실험 성능 향상
# 
# 사용법:
#   ./gpu_memory_optimizer.sh [옵션]
#   
# 옵션:
#   --check-only    현재 상태만 확인 (정리 안함)
#   --deep-clean    강력한 정리 (모든 캐시 삭제)
#   --auto          자동 정리 (권장)
#   --help          도움말 표시

set -euo pipefail

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 스크립트 시작 시간
START_TIME=$(date +%s)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# PROJECT_ROOT 자동 감지 (경로 독립적으로 수정)
if [[ "$SCRIPT_DIR" == */scripts/server_sync ]]; then
    # 스크립트가 scripts/server_sync 디렉토리에 있는 경우
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
else
    # 다른 위치에서 실행될 경우 현재 디렉토리 사용
    PROJECT_ROOT="$(pwd)"
fi

# nvidia-smi 절대 경로 설정
NVIDIA_SMI="/usr/bin/nvidia-smi"

# 로그 파일 설정
LOG_FILE="$PROJECT_ROOT/logs/gpu_optimizer_$(date '+%Y%m%d_%H%M%S').log"
mkdir -p "$(dirname "$LOG_FILE")"

# 전역 변수 (정리 전후 비교용)
MEMORY_BEFORE=0
MEMORY_AFTER=0

# 로깅 함수
log_info() {
    echo -e "${CYAN}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

# GPU 메모리 상태 가져오기 함수 (숫자만 반환)
get_gpu_memory() {
    local memory_used
    memory_used=$($NVIDIA_SMI --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | xargs | tr -d ',')
    
    if [ -n "$memory_used" ]; then
        # 소수점 값을 정수로 변환
        memory_used=$(echo "$memory_used" | cut -d'.' -f1)
        echo "$memory_used"
    else
        echo "0"
    fi
}

# GPU 상태 확인 함수
check_gpu_status() {
    local title="$1"
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}📊 $title${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    if command -v "$NVIDIA_SMI" &> /dev/null; then
        # GPU 메모리 사용량 파싱
        local gpu_info
        gpu_info=$($NVIDIA_SMI --query-gpu=memory.used,memory.total,temperature.gpu,utilization.gpu --format=csv,noheader,nounits)
        
        if [ -n "$gpu_info" ]; then
            local memory_used memory_total temperature utilization
            IFS=',' read -r memory_used memory_total temperature utilization <<< "$gpu_info"
            
            # 공백 제거
            memory_used=$(echo "$memory_used" | xargs)
            memory_total=$(echo "$memory_total" | xargs)
            temperature=$(echo "$temperature" | xargs)
            utilization=$(echo "$utilization" | xargs)
            
            local memory_free=$((memory_total - memory_used))
            local memory_percent=$((memory_used * 100 / memory_total))
            
            echo "  🗜️  GPU 메모리: ${memory_used}MB / ${memory_total}MB (${memory_percent}%)"
            echo "  🔓 사용 가능: ${memory_free}MB"
            echo "  🌡️  온도: ${temperature}°C"
            echo "  ⚡ GPU 활용률: ${utilization}%"
            
            # 상태 평가
            if [ "$memory_used" -gt 20000 ]; then
                echo -e "  ${RED}⚠️  위험: GPU 메모리 과부하 (20GB 초과)${NC}"
                return 2
            elif [ "$memory_used" -gt 15000 ]; then
                echo -e "  ${YELLOW}⚠️  주의: GPU 메모리 사용량 높음 (15GB 초과)${NC}"
                return 1
            elif [ "$memory_used" -gt 10000 ]; then
                echo -e "  ${CYAN}ℹ️  보통: GPU 메모리 적정 사용 (10GB 초과)${NC}"
                return 0
            else
                echo -e "  ${GREEN}✅ 양호: GPU 메모리 여유량 충분 (10GB 미만)${NC}"
                return 0
            fi
        else
            log_error "GPU 정보를 가져올 수 없습니다"
            return 3
        fi
    else
        log_error "$NVIDIA_SMI를 찾을 수 없습니다"
        return 3
    fi
}

# Python 프로세스 확인 및 정리
cleanup_python_processes() {
    log_info "Python 프로세스 분석 중..."
    
    # 현재 Python 프로세스 찾기
    local python_pids
    python_pids=$(pgrep -f python || true)
    
    if [ -n "$python_pids" ]; then
        echo -e "\n${YELLOW}🐍 실행 중인 Python 프로세스:${NC}"
        for pid in $python_pids; do
            local cmd
            cmd=$(ps -p "$pid" -o cmd --no-headers 2>/dev/null || echo "프로세스 종료됨")
            echo "  PID $pid: $cmd"
        done
        
        # GPU 사용 프로세스 확인
        local gpu_processes
        gpu_processes=$($NVIDIA_SMI --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)
        
        if [ -n "$gpu_processes" ]; then
            echo -e "\n${RED}🎮 GPU 사용 프로세스:${NC}"
            echo "$gpu_processes"
            
            if [ "$1" = "--deep-clean" ]; then
                log_warn "GPU 사용 프로세스 종료 중..."
                echo "$gpu_processes" | while IFS=',' read -r pid process_name used_memory; do
                    pid=$(echo "$pid" | xargs)
                    if [ -n "$pid" ] && [ "$pid" != "pid" ]; then
                        log_info "프로세스 $pid ($process_name) 종료 중..."
                        kill -TERM "$pid" 2>/dev/null || true
                        sleep 2
                        if kill -0 "$pid" 2>/dev/null; then
                            log_warn "강제 종료: $pid"
                            kill -KILL "$pid" 2>/dev/null || true
                        fi
                    fi
                done
            fi
        else
            log_info "GPU 사용 프로세스 없음"
        fi
    else
        log_info "실행 중인 Python 프로세스 없음"
    fi
}

# PyTorch 캐시 정리
cleanup_pytorch_cache() {
    log_info "PyTorch 캐시 정리 중..."
    
    # Python을 사용하여 PyTorch 캐시 정리
    python3 -c "
import torch
import gc
import os

try:
    # CUDA 캐시 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print('✅ CUDA 캐시 정리 완료')
    else:
        print('ℹ️  CUDA 사용 불가')
    
    # 가비지 컬렉션
    gc.collect()
    print('✅ 가비지 컬렉션 완료')
    
    # 캐시 디렉토리 정리
    cache_dirs = [
        os.path.expanduser('~/.cache/torch'),
        os.path.expanduser('~/.cache/huggingface'),
        '/tmp/torch_*',
        '$PROJECT_ROOT/hf_cache/transformers',
        '$PROJECT_ROOT/.cache'
    ]
    
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            print(f'📁 캐시 디렉토리 발견: {cache_dir}')
        else:
            print(f'📁 캐시 디렉토리 없음: {cache_dir}')
            
except Exception as e:
    print(f'❌ 오류 발생: {e}')
" 2>/dev/null || log_warn "PyTorch 캐시 정리 중 오류 발생"
}

# HuggingFace 캐시 정리
cleanup_huggingface_cache() {
    log_info "HuggingFace 캐시 분석 중..."
    
    local hf_cache_dirs=(
        "$HOME/.cache/huggingface"
        "$PROJECT_ROOT/hf_cache"
        "$PROJECT_ROOT/.cache/huggingface"
    )
    
    for cache_dir in "${hf_cache_dirs[@]}"; do
        if [ -d "$cache_dir" ]; then
            local cache_size
            cache_size=$(du -sh "$cache_dir" 2>/dev/null | cut -f1 || echo "0")
            echo "  📁 $cache_dir: $cache_size"
            
            if [ "$1" = "--deep-clean" ]; then
                log_warn "$cache_dir 정리 중..."
                find "$cache_dir" -type f -mtime +7 -delete 2>/dev/null || true
                find "$cache_dir" -type d -empty -delete 2>/dev/null || true
            fi
        fi
    done
}

# 시스템 메모리 정리
cleanup_system_memory() {
    log_info "시스템 메모리 정리 중..."
    
    # 페이지 캐시 정리 (권한이 있을 경우)
    if [ -w /proc/sys/vm/drop_caches ]; then
        sync
        echo 3 > /proc/sys/vm/drop_caches
        log_success "시스템 캐시 정리 완료"
    else
        log_warn "시스템 캐시 정리 권한 없음 (정상)"
    fi
    
    # 임시 파일 정리
    local temp_dirs=(
        "/tmp/torch_*"
        "/tmp/transformers_*"
        "/tmp/huggingface_*"
        "$PROJECT_ROOT/wandb/run-*"
    )
    
    for pattern in "${temp_dirs[@]}"; do
        if ls $pattern 2>/dev/null | head -1 | grep -q .; then
            log_info "임시 파일 정리: $pattern"
            rm -rf $pattern 2>/dev/null || true
        fi
    done
}

# CUDA 장치 재시작 (실험적)
reset_cuda_device() {
    if [ "$1" = "--deep-clean" ]; then
        log_warn "CUDA 장치 재시작 시도 중..."
        
        python3 -c "
import torch
try:
    if torch.cuda.is_available():
        # 모든 CUDA 텐서 해제
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # CUDA 컨텍스트 재설정 시도
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        print('✅ CUDA 장치 재시작 완료')
    else:
        print('ℹ️  CUDA 사용 불가')
except Exception as e:
    print(f'❌ CUDA 재시작 실패: {e}')
" 2>/dev/null || log_warn "CUDA 장치 재시작 실패"
    fi
}

# 정리 결과 비교 표시 함수
show_cleanup_results() {
    local memory_before="$1"
    local memory_after="$2"
    
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}📈 GPU 메모리 정리 결과${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    local memory_cleaned=$((memory_before - memory_after))
    local memory_cleaned_gb=$((memory_cleaned / 1024))
    local memory_cleaned_percent=0
    
    if [ "$memory_before" -gt 0 ]; then
        memory_cleaned_percent=$((memory_cleaned * 100 / memory_before))
    fi
    
    echo "  📊 정리 전: ${memory_before}MB"
    echo "  📊 정리 후: ${memory_after}MB"
    
    if [ "$memory_cleaned" -gt 0 ]; then
        echo -e "  ${GREEN}✅ 정리된 메모리: ${memory_cleaned}MB (${memory_cleaned_gb}GB)${NC}"
        echo -e "  ${GREEN}📈 정리 효율: ${memory_cleaned_percent}% 감소${NC}"
        
        if [ "$memory_cleaned" -gt 5000 ]; then
            echo -e "  ${GREEN}🎉 탁월한 정리 성과! (5GB 이상 정리)${NC}"
        elif [ "$memory_cleaned" -gt 2000 ]; then
            echo -e "  ${CYAN}👍 좋은 정리 성과! (2GB 이상 정리)${NC}"
        elif [ "$memory_cleaned" -gt 500 ]; then
            echo -e "  ${YELLOW}👌 적당한 정리 성과 (500MB 이상 정리)${NC}"
        fi
    elif [ "$memory_cleaned" -eq 0 ]; then
        echo -e "  ${YELLOW}ℹ️  정리된 메모리: 변화 없음${NC}"
    else
        echo -e "  ${RED}⚠️  메모리 증가: $((memory_after - memory_before))MB (일시적 현상일 수 있음)${NC}"
    fi
}

# 메인 최적화 함수
optimize_gpu_memory() {
    local mode="$1"
    
    echo -e "\n${GREEN}🚀 GPU 메모리 최적화 시작${NC}"
    log_info "모드: $mode"
    
    # 1. 현재 상태 확인 및 정리 전 메모리 저장
    check_gpu_status "최적화 전 상태"
    local initial_status=$?
    MEMORY_BEFORE=$(get_gpu_memory)
    
    if [ "$mode" = "--check-only" ]; then
        log_info "체크 모드: 정리 작업 생략"
        return 0
    fi
    
    # 2. Python 프로세스 정리
    cleanup_python_processes "$mode"
    
    # 3. PyTorch 캐시 정리
    cleanup_pytorch_cache
    
    # 4. HuggingFace 캐시 정리
    cleanup_huggingface_cache "$mode"
    
    # 5. 시스템 메모리 정리
    cleanup_system_memory
    
    # 6. CUDA 장치 재시작 (deep-clean 모드에서만)
    reset_cuda_device "$mode"
    
    # 7. 최종 상태 확인 및 정리 후 메모리 저장
    sleep 3
    MEMORY_AFTER=$(get_gpu_memory)
    check_gpu_status "최적화 후 상태"
    local final_status=$?
    
    # 8. 정리 결과 비교 표시
    show_cleanup_results "$MEMORY_BEFORE" "$MEMORY_AFTER"
    
    # 9. 결과 리포트
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}📋 최적화 결과 요약${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    if [ $final_status -lt $initial_status ]; then
        log_success "GPU 메모리 사용량 개선됨"
    elif [ $final_status -eq $initial_status ]; then
        log_info "GPU 메모리 사용량 유지됨"
    else
        log_warn "GPU 메모리 사용량 증가됨 (일시적 현상일 수 있음)"
    fi
    
    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - START_TIME))
    log_info "최적화 소요 시간: ${duration}초"
    log_info "로그 파일: $LOG_FILE"
}

# 도움말 출력
show_help() {
    cat << EOF
🚀 GPU 메모리 최적화 스크립트

사용법:
  $0 [옵션]

옵션:
  --check-only    현재 GPU 상태만 확인 (정리 안함)
  --auto          자동 정리 (권장, 기본값)
  --deep-clean    강력한 정리 (모든 캐시 삭제, GPU 프로세스 종료)
  --help          이 도움말 표시

예시:
  $0                    # 자동 정리
  $0 --check-only       # 상태 확인만
  $0 --deep-clean       # 강력한 정리

설명:
  - auto: 안전한 캐시 정리 및 메모리 최적화
  - deep-clean: GPU 프로세스 종료 + 모든 캐시 삭제 (주의 필요)
  
로그 위치: $PROJECT_ROOT/logs/gpu_optimizer_*.log
EOF
}

# 메인 실행부
main() {
    echo -e "${CYAN}🎯 AIStages GPU 메모리 최적화 도구${NC}"
    echo -e "프로젝트: $PROJECT_ROOT"
    echo -e "시작 시간: $(date '+%Y-%m-%d %H:%M:%S')"
    
    local mode="${1:---auto}"
    
    case "$mode" in
        --help|-h)
            show_help
            exit 0
            ;;
        --check-only|--auto|--deep-clean)
            optimize_gpu_memory "$mode"
            ;;
        *)
            log_error "알 수 없는 옵션: $mode"
            echo -e "\n사용법: $0 [--check-only|--auto|--deep-clean|--help]"
            exit 1
            ;;
    esac
}

# 스크립트가 직접 실행될 때만 main 함수 호출
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
