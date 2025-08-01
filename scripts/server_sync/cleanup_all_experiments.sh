#!/bin/bash

#####################################################################
# AIStages 실험 결과 완전 삭제 스크립트
# 
# 용도: 로컬 및 원격 서버에서 실험 결과를 모두 삭제
# 작성자: LYJ
# 날짜: 2025-08-01
# 
# 경고: 이 스크립트는 모든 실험 결과를 영구적으로 삭제합니다!
#####################################################################

set -e  # 에러 발생 시 스크립트 중단

# =================================================================
# 설정 파일 로드
# =================================================================

# 스크립트 디렉토리 경로
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/config.conf"

# 색상 코드
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# 로깅 함수들
log_info() {
    echo -e "${BLUE}[정보]${NC} $1" >&2
}

log_success() {
    echo -e "${GREEN}[성공]${NC} $1" >&2
}

log_warning() {
    echo -e "${YELLOW}[경고]${NC} $1" >&2
}

log_error() {
    echo -e "${RED}[에러]${NC} $1" >&2
}

log_danger() {
    echo -e "${MAGENTA}[위험]${NC} $1" >&2
}

# 설정 파일 존재 여부 확인
if [[ ! -f "$CONFIG_FILE" ]]; then
    log_error "설정 파일을 찾을 수 없습니다: $CONFIG_FILE"
    exit 1
fi

# 설정 파일 로드
source "$CONFIG_FILE"

# 필수 설정 검증
if [[ -z "$LOCAL_BASE" ]] || [[ -z "$REMOTE_BASE" ]] || [[ -z "$REMOTE_HOST" ]]; then
    log_error "필수 설정이 누락되었습니다. config.conf 파일을 확인하세요."
    exit 1
fi

# 경로 설정
LOCAL_OUTPUTS_DIR="${LOCAL_BASE}/${OUTPUTS_SUBDIR:-outputs}"
LOCAL_LOGS_DIR="${LOCAL_BASE}/${LOGS_SUBDIR:-logs}"
LOCAL_WANDB_DIR="${LOCAL_BASE}/${WANDB_SUBDIR:-wandb}"
LOCAL_MODELS_DIR="${LOCAL_BASE}/${MODELS_SUBDIR:-models}"
LOCAL_DATA_DIR="${LOCAL_BASE}/${DATA_SUBDIR:-data}"

REMOTE_OUTPUTS_DIR="${REMOTE_BASE}/${OUTPUTS_SUBDIR:-outputs}"
REMOTE_LOGS_DIR="${REMOTE_BASE}/${LOGS_SUBDIR:-logs}"
REMOTE_WANDB_DIR="${REMOTE_BASE}/${WANDB_SUBDIR:-wandb}"
REMOTE_MODELS_DIR="${REMOTE_BASE}/${MODELS_SUBDIR:-models}"
REMOTE_DATA_DIR="${REMOTE_BASE}/${DATA_SUBDIR:-data}"

# =================================================================
# 유틸리티 함수들
# =================================================================

# 크기 계산 함수
calculate_directory_size() {
    local dir_path="$1"
    local is_remote="$2"
    
    if [[ "$is_remote" == "true" ]]; then
        # 원격 디렉토리 크기
        ssh "$REMOTE_HOST" "du -sh '$dir_path' 2>/dev/null | cut -f1" 2>/dev/null || echo "0B"
    else
        # 로컬 디렉토리 크기
        if [[ -d "$dir_path" ]]; then
            du -sh "$dir_path" 2>/dev/null | cut -f1 || echo "0B"
        else
            echo "0B"
        fi
    fi
}

# 파일 개수 계산 함수
count_files() {
    local dir_path="$1"
    local is_remote="$2"
    
    if [[ "$is_remote" == "true" ]]; then
        # 원격 파일 개수
        ssh "$REMOTE_HOST" "find '$dir_path' -type f 2>/dev/null | wc -l" 2>/dev/null || echo "0"
    else
        # 로컬 파일 개수
        if [[ -d "$dir_path" ]]; then
            find "$dir_path" -type f 2>/dev/null | wc -l || echo "0"
        else
            echo "0"
        fi
    fi
}

# =================================================================
# 삭제 함수들
# =================================================================

# 로컬 실험 결과 삭제
cleanup_local_results() {
    log_info "로컬 실험 결과 분석 중..."
    
    # 각 디렉토리별 정보 수집
    local outputs_size=$(calculate_directory_size "$LOCAL_OUTPUTS_DIR" false)
    local logs_size=$(calculate_directory_size "$LOCAL_LOGS_DIR" false)
    local wandb_size=$(calculate_directory_size "$LOCAL_WANDB_DIR" false)
    local models_size=$(calculate_directory_size "$LOCAL_MODELS_DIR" false)
    local data_size=$(calculate_directory_size "$LOCAL_DATA_DIR" false)
    
    local outputs_files=$(count_files "$LOCAL_OUTPUTS_DIR" false)
    local logs_files=$(count_files "$LOCAL_LOGS_DIR" false)
    local wandb_files=$(count_files "$LOCAL_WANDB_DIR" false)
    local models_files=$(count_files "$LOCAL_MODELS_DIR" false)
    local data_files=$(count_files "$LOCAL_DATA_DIR" false)
    
    echo
    log_info "=== 로컬 삭제 대상 분석 ==="
    echo "📁 Outputs: $outputs_size ($outputs_files 파일)"
    echo "📁 Logs: $logs_size ($logs_files 파일)"
    echo "📁 WandB: $wandb_size ($wandb_files 파일)"
    echo "📁 Models: $models_size ($models_files 파일)"
    echo "📁 Data: $data_size ($data_files 파일) - 보존됨"
    
    # 전체 합계 (데이터 파일 제외)
    local total_files=$((outputs_files + logs_files + wandb_files + models_files))
    
    if [[ $total_files -eq 0 ]]; then
        log_info "로컬에 삭제할 실험 결과가 없습니다."
        return 0
    fi
    
    echo
    log_danger "⚠️  총 $total_files 개의 파일이 영구적으로 삭제됩니다!"
    echo
    read -p "정말로 로컬 실험 결과를 모두 삭제하시겠습니까? (DELETE 입력 필요): " -r
    echo
    
    if [[ "$REPLY" != "DELETE" ]]; then
        log_info "로컬 삭제가 취소되었습니다."
        return 0
    fi
    
    log_info "로컬 실험 결과 삭제 시작..."
    
    # 각 디렉토리 삭제 (데이터 파일 제외)
    for dir_info in "outputs:$LOCAL_OUTPUTS_DIR" "logs:$LOCAL_LOGS_DIR" "wandb:$LOCAL_WANDB_DIR" "models:$LOCAL_MODELS_DIR"; do
        local dir_name="${dir_info%%:*}"
        local dir_path="${dir_info##*:}"
        
        if [[ -d "$dir_path" ]]; then
            log_info "$dir_name 디렉토리 삭제 중..."
            rm -rf "$dir_path"/* 2>/dev/null || true
            log_success "✅ $dir_name 삭제 완료"
        else
            log_info "$dir_name 디렉토리가 존재하지 않습니다"
        fi
    done
    
    # 추가 정리 파일들
    local additional_files=(
        "$LOCAL_BASE/benchmark_*.log"
        "$LOCAL_BASE/mt5_training*.log"
        "$LOCAL_BASE/sync_report_*.txt"
        "$LOCAL_BASE/.synced_experiments"
    )
    
    for pattern in "${additional_files[@]}"; do
        if ls $pattern 1> /dev/null 2>&1; then
            log_info "추가 파일 삭제: $(basename $pattern)"
            rm -f $pattern
        fi
    done
    
    log_success "🎉 로컬 실험 결과 삭제 완료!"
}

# 원격 실험 결과 삭제
cleanup_remote_results() {
    log_info "원격 서버 연결 테스트 중..."
    
    if ! ssh "$REMOTE_HOST" "echo '연결 성공'" >/dev/null 2>&1; then
        log_error "원격 서버에 연결할 수 없습니다: $REMOTE_HOST"
        return 1
    fi
    
    log_info "원격 실험 결과 분석 중..."
    
    # 각 디렉토리별 정보 수집
    local outputs_size=$(calculate_directory_size "$REMOTE_OUTPUTS_DIR" true)
    local logs_size=$(calculate_directory_size "$REMOTE_LOGS_DIR" true)
    local wandb_size=$(calculate_directory_size "$REMOTE_WANDB_DIR" true)
    local models_size=$(calculate_directory_size "$REMOTE_MODELS_DIR" true)
    local data_size=$(calculate_directory_size "$REMOTE_DATA_DIR" true)
    
    local outputs_files=$(count_files "$REMOTE_OUTPUTS_DIR" true)
    local logs_files=$(count_files "$REMOTE_LOGS_DIR" true)
    local wandb_files=$(count_files "$REMOTE_WANDB_DIR" true)
    local models_files=$(count_files "$REMOTE_MODELS_DIR" true)
    local data_files=$(count_files "$REMOTE_DATA_DIR" true)
    
    echo
    log_info "=== 원격 서버 삭제 대상 분석 ==="
    echo "📁 Outputs: $outputs_size ($outputs_files 파일)"
    echo "📁 Logs: $logs_size ($logs_files 파일)"
    echo "📁 WandB: $wandb_size ($wandb_files 파일)"
    echo "📁 Models: $models_size ($models_files 파일)"
    echo "📁 Data: $data_size ($data_files 파일) - 보존됨"
    
    # 전체 합계 (데이터 파일 제외)
    local total_files=$((outputs_files + logs_files + wandb_files + models_files))
    
    if [[ $total_files -eq 0 ]]; then
        log_info "원격 서버에 삭제할 실험 결과가 없습니다."
        return 0
    fi
    
    echo
    log_danger "⚠️  원격 서버에서 총 $total_files 개의 파일이 영구적으로 삭제됩니다!"
    echo
    read -p "정말로 원격 서버의 실험 결과를 모두 삭제하시겠습니까? (DELETE 입력 필요): " -r
    echo
    
    if [[ "$REPLY" != "DELETE" ]]; then
        log_info "원격 서버 삭제가 취소되었습니다."
        return 0
    fi
    
    log_info "원격 서버 실험 결과 삭제 시작..."
    
    # 각 디렉토리 삭제 (데이터 파일 제외)
    for dir_info in "outputs:$REMOTE_OUTPUTS_DIR" "logs:$REMOTE_LOGS_DIR" "wandb:$REMOTE_WANDB_DIR" "models:$REMOTE_MODELS_DIR"; do
        local dir_name="${dir_info%%:*}"
        local dir_path="${dir_info##*:}"
        
        log_info "$dir_name 디렉토리 삭제 중..."
        ssh "$REMOTE_HOST" "if [ -d '$dir_path' ]; then rm -rf '$dir_path'/* 2>/dev/null || true; echo '$dir_name 삭제 완료'; else echo '$dir_name 디렉토리가 존재하지 않습니다'; fi"
    done
    
    # 추가 정리 파일들
    log_info "추가 파일들 삭제 중..."
    ssh "$REMOTE_HOST" "cd '$REMOTE_BASE' && rm -f benchmark_*.log mt5_training*.log *.tmp .synced_experiments 2>/dev/null || true"
    
    log_success "🎉 원격 서버 실험 결과 삭제 완료!"
}

# =================================================================
# 메인 실행 함수
# =================================================================

main() {
    echo "======================================================"
    echo "🗑️  AIStages 실험 결과 완전 삭제 도구"
    echo "======================================================"
    echo
    log_danger "⚠️  경고: 이 도구는 모든 실험 결과를 영구적으로 삭제합니다!"
    log_danger "⚠️  삭제된 데이터는 복구할 수 없습니다!"
    echo
    
    # 사용자 최종 확인
    echo -e "${YELLOW}삭제 대상:${NC}"
    echo "- 로컬: $LOCAL_BASE"
    echo "- 원격: $REMOTE_HOST:$REMOTE_BASE"
    echo "- 디렉토리: outputs, logs, wandb, models (데이터 파일 제외)"
    echo
    
    read -p "정말로 계속하시겠습니까? (yes 입력 필요): " -r
    echo
    
    if [[ "$REPLY" != "yes" ]]; then
        log_info "작업이 취소되었습니다."
        exit 0
    fi
    
    # 로컬 정리
    if [[ "${CLEAN_LOCAL:-true}" == "true" ]]; then
        cleanup_local_results
        echo
    fi
    
    # 원격 정리  
    if [[ "${CLEAN_REMOTE:-true}" == "true" ]]; then
        cleanup_remote_results
        echo
    fi
    
    # 완료 메시지
    echo "======================================================"
    log_success "🎉 모든 실험 결과 삭제 완료!"
    log_info "새로운 실험을 시작할 준비가 되었습니다."
    echo "======================================================"
}

# =================================================================
# 사용법 및 실행
# =================================================================

show_usage() {
    echo "사용법: $0 [옵션]"
    echo
    echo "옵션:"
    echo "  -h, --help          이 도움말 출력"
    echo "  -l, --local-only    로컬만 삭제"
    echo "  -r, --remote-only   원격 서버만 삭제"
    echo "  -s, --server HOST   서버 주소 지정"
    echo
    echo "예시:"
    echo "  $0                  # 로컬 + 원격 모두 삭제"
    echo "  $0 --local-only     # 로컬만 삭제"
    echo "  $0 --remote-only    # 원격 서버만 삭제"
    echo
    echo "⚠️  경고: 이 도구는 모든 실험 결과를 영구적으로 삭제합니다!"
    echo
}

# 명령행 인자 처리
CLEAN_LOCAL=true
CLEAN_REMOTE=true

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -l|--local-only)
            CLEAN_LOCAL=true
            CLEAN_REMOTE=false
            shift
            ;;
        -r|--remote-only)
            CLEAN_LOCAL=false
            CLEAN_REMOTE=true
            shift
            ;;
        -s|--server)
            REMOTE_HOST="$2"
            log_info "서버 주소를 $2로 변경했습니다"
            shift 2
            ;;
        *)
            log_error "알 수 없는 옵션: $1"
            show_usage
            exit 1
            ;;
    esac
done

# 스크립트 실행
main
