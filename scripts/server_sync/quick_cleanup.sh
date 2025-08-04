#!/bin/bash

# =================================================================
# AIStages 실험 결과 빠른 정리 스크립트
# =================================================================
# 경고: 이 스크립트는 확인 절차 없이 즉시 모든 실험 결과를 삭제합니다!
# 
# 주요 기능:
# - 로컬 및 원격 서버의 실험 결과 즉시 삭제
# - 확인 절차 없는 빠른 실행
# - prediction과 data 디렉토리는 보호
# 
# 사용법:
#   ./scripts/server_sync/quick_cleanup.sh
# 
# 작성자: Claude MCP  
# 수정일: 2025-08-04
# =================================================================

set -euo pipefail

# =================================================================
# 설정 및 초기화
# =================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF_FILE="$SCRIPT_DIR/config.conf"

# 설정 파일 확인 및 로드
if [[ ! -f "$CONF_FILE" ]]; then
    echo "❌ 설정 파일을 찾을 수 없습니다: $CONF_FILE"
    echo "힌트: config.conf.template을 복사하여 config.conf로 이름을 변경하세요"
    exit 1
fi

# shellcheck source=scripts/server_sync/config.conf
source "$CONF_FILE"

# =================================================================
# 유틸리티 함수들
# =================================================================

log_info() {
    echo "ℹ️  $1"  
}

log_success() {
    echo "✅ $1"  
}

log_warning() {
    echo "⚠️  $1"
}

log_error() {
    echo "❌ $1"
}

# =================================================================
# 메인 삭제 로직
# =================================================================

# 삭제 대상 디렉토리 정의
DIRS_TO_CLEAN=(
    "실험출력:outputs"
    "학습로그:logs"
    "모델체크포인트:checkpoints"
    "저장모델:models"
    "검증로그:validation_logs"
    "분석결과:analysis_results"
    "최종제출:final_submission"
    "WandB로그:wandb"
)

echo ""
log_info "AIStages 실험 결과 빠른 정리 시작..."
echo ""

# 로컬 삭제
log_info "로컬 실험 결과 삭제 중..."

for dir_info in "${DIRS_TO_CLEAN[@]}"; do
    dir_path="${dir_info#*:}"
    full_path="${LOCAL_BASE}/${dir_path}"
    
    if [[ -d "$full_path" ]] && [[ -n "$full_path" ]]; then
        rm -rf "${full_path:?}"/* 2>/dev/null || true
        log_success "$(basename "$full_path") 삭제 완료"
    fi
done

# 추가 파일 삭제 (config.conf 기반)
# 벤치마크 로그 파일들
[[ -n "$BENCHMARK_LOGS_PATTERN" ]] && rm -f "$LOCAL_BASE"/$BENCHMARK_LOGS_PATTERN 2>/dev/null || true

# 학습 로그 파일들
[[ -n "$TRAINING_LOGS_PATTERN" ]] && rm -f "$LOCAL_BASE"/$TRAINING_LOGS_PATTERN 2>/dev/null || true

# 동기화 보고서 파일들
[[ -n "$SYNC_REPORT_PATTERN" ]] && rm -f "$LOCAL_BASE"/$SYNC_REPORT_PATTERN 2>/dev/null || true

# 동기화 상태 추적 파일들
if [[ -n "$SYNCED_EXPERIMENTS_FILE" ]]; then
    rm -f "$LOCAL_BASE"/$SYNCED_EXPERIMENTS_FILE "$LOCAL_BASE"/outputs/$SYNCED_EXPERIMENTS_FILE "$LOCAL_BASE"/logs/$SYNCED_EXPERIMENTS_FILE 2>/dev/null || true
fi

# Python 캐시 정리
if [[ "$CLEAN_PYTHON_CACHE" == "true" ]]; then
    log_info "Python 캐시 파일 정리 중..."
    find "$LOCAL_BASE" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find "$LOCAL_BASE" -name "*.pyc" -delete 2>/dev/null || true
    log_success "Python 캐시 정리 완료"
fi

# 원격 삭제
log_info "원격 서버 실험 결과 삭제 중..."
if ssh "$REMOTE_HOST" "echo '연결 확인'" >/dev/null 2>&1; then
    
    # 각 디렉토리별 삭제
    for dir_info in "${DIRS_TO_CLEAN[@]}"; do
        dir_path="${dir_info#*:}"
        full_path="${REMOTE_BASE}/${dir_path}"
        
        ssh "$REMOTE_HOST" "if [ -d '$full_path' ]; then rm -rf '$full_path'/* 2>/dev/null || true; fi" 2>/dev/null || true
    done
    
    # 추가 파일 삭제 (config.conf 기반)
    # 벤치마크 로그 파일들
    [[ -n "$BENCHMARK_LOGS_PATTERN" ]] && ssh "$REMOTE_HOST" "cd '$REMOTE_BASE' && rm -f $BENCHMARK_LOGS_PATTERN 2>/dev/null || true"
    
    # 학습 로그 파일들
    [[ -n "$TRAINING_LOGS_PATTERN" ]] && ssh "$REMOTE_HOST" "cd '$REMOTE_BASE' && rm -f $TRAINING_LOGS_PATTERN 2>/dev/null || true"
    
    # 동기화 보고서 파일들
    [[ -n "$SYNC_REPORT_PATTERN" ]] && ssh "$REMOTE_HOST" "cd '$REMOTE_BASE' && rm -f $SYNC_REPORT_PATTERN 2>/dev/null || true"
    
    # 동기화 상태 추적 파일들
    if [[ -n "$SYNCED_EXPERIMENTS_FILE" ]]; then
        ssh "$REMOTE_HOST" "cd '$REMOTE_BASE' && rm -f $SYNCED_EXPERIMENTS_FILE outputs/$SYNCED_EXPERIMENTS_FILE logs/$SYNCED_EXPERIMENTS_FILE 2>/dev/null || true"
    fi
    
    # 임시 파일들 (기본 정리)
    ssh "$REMOTE_HOST" "cd '$REMOTE_BASE' && rm -f *.tmp 2>/dev/null || true"
    
    # 원격 Python 캐시 정리
    if [[ "$CLEAN_PYTHON_CACHE" == "true" ]]; then
        log_info "원격 Python 캐시 파일 정리 중..."
        ssh "$REMOTE_HOST" "find '$REMOTE_BASE' -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true"
        ssh "$REMOTE_HOST" "find '$REMOTE_BASE' -name '*.pyc' -delete 2>/dev/null || true"
        log_success "원격 Python 캐시 정리 완료"
    fi
    
    log_success "원격 서버 삭제 완료"
else
    log_warning "원격 서버에 연결할 수 없습니다"
fi

log_success "🎉 모든 실험 결과 삭제 완료!"
echo "새로운 실험을 시작할 수 있습니다."
