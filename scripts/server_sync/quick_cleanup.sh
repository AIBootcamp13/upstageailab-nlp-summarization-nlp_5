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

echo "🗑️  빠른 실험 결과 삭제 도구"
echo "=================================="

# 로컬 삭제 (데이터 파일 제외)
log_info "로컬 실험 결과 삭제 중..."
for dir in "$LOCAL_OUTPUTS_DIR" "$LOCAL_LOGS_DIR" "$LOCAL_WANDB_DIR" "$LOCAL_MODELS_DIR"; do
    if [[ -d "$dir" ]]; then
        rm -rf "$dir"/* 2>/dev/null || true
        log_success "$(basename "$dir") 삭제 완료"
    fi
done

# 추가 파일 삭제
rm -f "$LOCAL_BASE"/benchmark_*.log "$LOCAL_BASE"/mt5_training*.log "$LOCAL_BASE"/sync_report_*.txt "$LOCAL_BASE"/.synced_experiments 2>/dev/null || true
# 원격 삭제 (데이터 파일 제외)
log_info "원격 서버 실험 결과 삭제 중..."
if ssh "$REMOTE_HOST" "echo '연결 확인'" >/dev/null 2>&1; then
    ssh "$REMOTE_HOST" "
        cd '$REMOTE_BASE' || exit 1
        rm -rf '$REMOTE_OUTPUTS_DIR'/* '$REMOTE_LOGS_DIR'/* '$REMOTE_WANDB_DIR'/* '$REMOTE_MODELS_DIR'/* 2>/dev/null || true
        rm -f benchmark_*.log mt5_training*.log *.tmp .synced_experiments 2>/dev/null || true
        echo '원격 서버 삭제 완료'
    "
    log_success "원격 서버 삭제 완료"
else
    log_warning "원격 서버에 연결할 수 없습니다"
fi

log_success "🎉 모든 실험 결과 삭제 완료!"
echo "새로운 실험을 시작할 수 있습니다."
