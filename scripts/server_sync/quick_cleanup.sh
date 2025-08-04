#!/bin/bash

#####################################################################
# 빠른 실험 결과 삭제 스크립트
# 
# 용도: 확인 절차를 최소화하여 빠르게 실험 결과 삭제
# 작성자: LYJ
# 날짜: 2025-08-01
#####################################################################

# 스크립트 디렉토리 경로
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/config.conf"

# 색상 코드
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 로깅 함수
log_info() { echo -e "${BLUE}[정보]${NC} $1"; }
log_success() { echo -e "${GREEN}[성공]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[경고]${NC} $1"; }
log_error() { echo -e "${RED}[에러]${NC} $1"; }

# 설정 파일 로드
if [[ ! -f "$CONFIG_FILE" ]]; then
    log_error "설정 파일을 찾을 수 없습니다: $CONFIG_FILE"
    exit 1
fi

source "$CONFIG_FILE"

# 필수 설정 검증
if [[ -z "$LOCAL_BASE" ]] || [[ -z "$REMOTE_BASE" ]] || [[ -z "$REMOTE_HOST" ]]; then
    log_error "필수 설정이 누락되었습니다. config.conf 파일을 확인하세요."
    exit 1
fi

# =================================================================
# 삭제 대상 경로 설정 (비어있으면 제외)
# =================================================================

# 삭제 대상 디렉토리들을 배열로 정의
DIRS_TO_CLEAN=()

# 경로가 비어있지 않은 디렉토리들만 삭제 대상에 추가
[[ -n "${OUTPUTS_PATH}" ]] && DIRS_TO_CLEAN+=("outputs:${OUTPUTS_PATH}")
[[ -n "${LOGS_PATH}" ]] && DIRS_TO_CLEAN+=("logs:${LOGS_PATH}")
[[ -n "${CHECKPOINTS_PATH}" ]] && DIRS_TO_CLEAN+=("checkpoints:${CHECKPOINTS_PATH}")
[[ -n "${MODELS_PATH}" ]] && DIRS_TO_CLEAN+=("models:${MODELS_PATH}")
[[ -n "${WANDB_PATH}" ]] && DIRS_TO_CLEAN+=("wandb:${WANDB_PATH}")
[[ -n "${VALIDATION_LOGS_PATH}" ]] && DIRS_TO_CLEAN+=("validation_logs:${VALIDATION_LOGS_PATH}")
[[ -n "${ANALYSIS_RESULTS_PATH}" ]] && DIRS_TO_CLEAN+=("analysis_results:${ANALYSIS_RESULTS_PATH}")
[[ -n "${FINAL_SUBMISSION_PATH}" ]] && DIRS_TO_CLEAN+=("final_submission:${FINAL_SUBMISSION_PATH}")
# DATA_PATH와 PREDICTION_PATH는 안전상 삭제 대상에서 제외 (중요한 데이터)

echo "🗑️  빠른 실험 결과 삭제 도구"
echo "=================================="

# 삭제 대상 디렉토리 표시
if [[ ${#DIRS_TO_CLEAN[@]} -eq 0 ]]; then
    log_info "삭제할 디렉토리가 설정되지 않았습니다."
    exit 0
fi

log_info "삭제 대상: ${#DIRS_TO_CLEAN[@]}개 디렉토리"
for dir_info in "${DIRS_TO_CLEAN[@]}"; do
    dir_type="${dir_info%%:*}"
    dir_path="${dir_info#*:}"
    log_info "  - $dir_type: $dir_path"
done
echo

# 로컬 삭제
log_info "로컬 실험 결과 삭제 중..."
for dir_info in "${DIRS_TO_CLEAN[@]}"; do
    dir_type="${dir_info%%:*}"
    dir_path="${dir_info#*:}"
    full_path="${LOCAL_BASE}/${dir_path}"
    
    if [[ -d "$full_path" ]] && [[ -n "$full_path" ]]; then
        rm -rf "${full_path:?}"/* 2>/dev/null || true
        log_success "$(basename "$full_path") 삭제 완료"
    fi
done

# 추가 파일 삭제
rm -f "$LOCAL_BASE"/benchmark_*.log "$LOCAL_BASE"/mt5_training*.log "$LOCAL_BASE"/sync_report_*.txt "$LOCAL_BASE"/.synced_experiments 2>/dev/null || true

# 원격 삭제
log_info "원격 서버 실험 결과 삭제 중..."
if ssh "$REMOTE_HOST" "echo '연결 확인'" >/dev/null 2>&1; then
    
    # 각 디렉토리별 삭제
    for dir_info in "${DIRS_TO_CLEAN[@]}"; do
        dir_type="${dir_info%%:*}"
        dir_path="${dir_info#*:}"
        full_path="${REMOTE_BASE}/${dir_path}"
        
        ssh "$REMOTE_HOST" "if [ -d \"$full_path\" ]; then rm -rf \"$full_path\"/* 2>/dev/null || true; fi" 2>/dev/null || true
    done
    
    # 추가 파일 삭제
    ssh "$REMOTE_HOST" "cd \"$REMOTE_BASE\" && rm -f benchmark_*.log mt5_training*.log *.tmp .synced_experiments 2>/dev/null || true"
    
    log_success "원격 서버 삭제 완료"
else
    log_warning "원격 서버에 연결할 수 없습니다"
fi

log_success "🎉 모든 실험 결과 삭제 완료!"
echo "새로운 실험을 시작할 수 있습니다."
