#!/bin/bash

# =================================================================
# AIStages 실험 결과 정리 스크립트 (완전 삭제)
# =================================================================
# 경고: 이 스크립트는 모든 실험 결과를 영구적으로 삭제합니다!
# 
# 주요 기능:
# - 로컬 및 원격 서버의 실험 결과 완전 삭제
# - 3단계 안전 확인 절차
# - 상세한 삭제 전 분석 및 사용자 확인
# - prediction과 data 디렉토리는 보호
# 
# 사용법:
#   ./scripts/server_sync/cleanup_all_experiments.sh
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

log_separator() {
    echo "================================================================="
}

# 사용자 확인 함수
confirm_action() {
    local message="$1"
    local response
    
    echo ""
    log_warning "$message"
    echo ""
    read -rp "계속하시겠습니까? (yes/no): " response
    
    case "$response" in
        yes|YES|Y|y)
            return 0
            ;;
        *)
            log_info "작업이 취소되었습니다."
            exit 0
            ;;
    esac
}

# =================================================================
# 분석 함수들
# =================================================================

# 디렉토리 크기 확인 (원격)
get_remote_dir_size() {
    local dir_path="$1"
    if ssh "$REMOTE_HOST" "[ -d '$dir_path' ]" 2>/dev/null; then
        ssh "$REMOTE_HOST" "du -sh '$dir_path' 2>/dev/null | cut -f1" 2>/dev/null || echo "0B"
    else
        echo "없음"
    fi
}

# 디렉토리 크기 확인 (로컬)
get_local_dir_size() {
    local dir_path="$1"
    if [[ -d "$dir_path" ]]; then
        du -sh "$dir_path" 2>/dev/null | cut -f1 || echo "0B"
    else
        echo "없음"
    fi
}

# 파일 개수 확인 (원격)
get_remote_file_count() {
    local dir_path="$1"
    if ssh "$REMOTE_HOST" "[ -d '$dir_path' ]" 2>/dev/null; then
        ssh "$REMOTE_HOST" "find '$dir_path' -type f 2>/dev/null | wc -l" 2>/dev/null || echo "0"
    else
        echo "0"
    fi
}

# 파일 개수 확인 (로컬)
get_local_file_count() {
    local dir_path="$1"
    if [[ -d "$dir_path" ]]; then
        find "$dir_path" -type f 2>/dev/null | wc -l || echo "0"
    else
        echo "0"
    fi
}

# =================================================================
# 분석 및 확인 함수
# =================================================================

analyze_cleanup_impact() {
    log_separator
    log_info "삭제될 실험 결과 분석 중..."
    log_separator
    
    # 삭제 대상 디렉토리 정의
    local dirs_to_clean=(
        "실험출력:outputs"
        "학습로그:logs" 
        "모델체크포인트:checkpoints"
        "저장모델:models"
        "검증로그:validation_logs"
        "분석결과:analysis_results"
        "최종제출:final_submission"
        "WandB로그:wandb"
    )
    
    local total_files_local=0
    local total_files_remote=0
    
    printf "%-15s %-12s %-8s %-12s %-8s\n" "디렉토리" "로컬크기" "파일수" "원격크기" "파일수"
    printf "%-15s %-12s %-8s %-12s %-8s\n" "---------------" "------------" "--------" "------------" "--------"
    
    for dir_info in "${dirs_to_clean[@]}"; do
        local dir_type="${dir_info%%:*}"
        local dir_path="${dir_info#*:}"
        local local_full_path="${LOCAL_BASE}/${dir_path}"
        local remote_full_path="${REMOTE_BASE}/${dir_path}"
        
        local local_size
        local remote_size
        local local_files
        local remote_files
        
        local_size=$(get_local_dir_size "$local_full_path")
        remote_size=$(get_remote_dir_size "$remote_full_path")
        local_files=$(get_local_file_count "$local_full_path")
        remote_files=$(get_remote_file_count "$remote_full_path")
        
        printf "%-15s %-12s %-8s %-12s %-8s\n" "$dir_type" "$local_size" "$local_files" "$remote_size" "$remote_files"
        
        # 숫자로 변환 가능한 경우만 합계에 추가
        if [[ "$local_files" =~ ^[0-9]+$ ]]; then
            total_files_local=$((total_files_local + local_files))
        fi
        if [[ "$remote_files" =~ ^[0-9]+$ ]]; then
            total_files_remote=$((total_files_remote + remote_files))
        fi
    done
    
    log_separator
    printf "%-15s %-12s %-8s %-12s %-8s\n" "총합계" "---" "$total_files_local" "---" "$total_files_remote"
    log_separator
    
    # 추가 정리 파일들 확인
    log_info "추가 정리 파일들:"
    echo "  - benchmark_*.log, mt5_training*.log"
    echo "  - sync_report_*.txt"
    echo "  - .synced_experiments 파일들"
    echo "  - Python 캐시 파일들 (__pycache__, *.pyc)"
    
    log_separator
    log_warning "보호되는 디렉토리 (삭제되지 않음):"
    echo "  - prediction/ (채점용 결과 파일)"
    echo "  - data/ (데이터셋)"
    log_separator
}

# =================================================================
# 실제 삭제 함수들
# =================================================================

# 로컬 실험 결과 삭제
cleanup_local_results() {
    log_info "로컬 실험 결과 삭제 시작..."
    
    # 삭제 대상 디렉토리
    local DIRS_TO_CLEAN=(
        "실험출력:outputs"
        "학습로그:logs"
        "모델체크포인트:checkpoints"
        "저장모델:models"
        "검증로그:validation_logs"
        "분석결과:analysis_results"
        "최종제출:final_submission"
        "WandB로그:wandb"
    )
    
    # 각 디렉토리 삭제
    for dir_info in "${DIRS_TO_CLEAN[@]}"; do
        local dir_type="${dir_info%%:*}"
        local dir_path="${dir_info#*:}"
        local full_path="${LOCAL_BASE}/${dir_path}"
        
        if [[ -d "$full_path" ]] && [[ -n "$full_path" ]]; then
            log_info "$dir_type 디렉토리 삭제 중..."
            rm -rf "${full_path:?}"/* 2>/dev/null || true
            log_success "✅ $dir_type 삭제 완료"
        else
            log_info "$dir_type 디렉토리가 존재하지 않습니다"
        fi
    done
    
    # 추가 정리 파일들 (config.conf 기반)
    log_info "추가 정리 파일 처리 중..."
    
    # 벤치마크 로그 파일들
    if [[ -n "$BENCHMARK_LOGS_PATTERN" ]]; then
        if ls "$LOCAL_BASE/$BENCHMARK_LOGS_PATTERN" 1> /dev/null 2>&1; then
            log_info "벤치마크 로그 파일 삭제: $BENCHMARK_LOGS_PATTERN"
            rm -f "$LOCAL_BASE/$BENCHMARK_LOGS_PATTERN"
        fi
    fi
    
    # 학습 로그 파일들
    if [[ -n "$TRAINING_LOGS_PATTERN" ]]; then
        if ls "$LOCAL_BASE/$TRAINING_LOGS_PATTERN" 1> /dev/null 2>&1; then
            log_info "학습 로그 파일 삭제: $TRAINING_LOGS_PATTERN"
            rm -f "$LOCAL_BASE/$TRAINING_LOGS_PATTERN"
        fi
    fi
    
    # 동기화 보고서 파일들
    if [[ -n "$SYNC_REPORT_PATTERN" ]]; then
        if ls "$LOCAL_BASE/$SYNC_REPORT_PATTERN" 1> /dev/null 2>&1; then
            log_info "동기화 보고서 파일 삭제: $SYNC_REPORT_PATTERN"
            rm -f "$LOCAL_BASE/$SYNC_REPORT_PATTERN"
        fi
    fi
    
    # 동기화 상태 추적 파일듡
    if [[ -n "$SYNCED_EXPERIMENTS_FILE" ]]; then
        local synced_files=(
            "$LOCAL_BASE/$SYNCED_EXPERIMENTS_FILE"
            "$LOCAL_BASE/outputs/$SYNCED_EXPERIMENTS_FILE"
            "$LOCAL_BASE/logs/$SYNCED_EXPERIMENTS_FILE"
        )
        
        for file in "${synced_files[@]}"; do
            if [[ -f "$file" ]]; then
                log_info "동기화 상태 파일 삭제: $(basename "$file")"
                rm -f "$file"
            fi
        done
    fi
    
    # Python 캐시 정리
    if [[ "$CLEAN_PYTHON_CACHE" == "true" ]]; then
        log_info "Python 캐시 파일 정리 중..."
        find "$LOCAL_BASE" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
        find "$LOCAL_BASE" -name "*.pyc" -delete 2>/dev/null || true
        log_success "✅ Python 캐시 정리 완룼"
    fi
    
    log_success "🎉 로컬 실험 결과 삭제 완료!"
}

# 원격 실험 결과 삭제
cleanup_remote_results() {
    log_info "원격 서버 연결 테스트 중..."
    
    if ! ssh "$REMOTE_HOST" "echo '연결 확인'" >/dev/null 2>&1; then
        log_error "원격 서버($REMOTE_HOST)에 연결할 수 없습니다."
        log_info "SSH 키 설정을 확인하거나 수동으로 원격 서버를 정리해주세요."
        return 1
    fi
    
    log_success "원격 서버 연결 성공"
    log_info "원격 서버 실험 결과 삭제 시작..."
    
    # 삭제 대상 디렉토리
    local DIRS_TO_CLEAN=(
        "실험출력:outputs"
        "학습로그:logs"
        "모델체크포인트:checkpoints"
        "저장모델:models"
        "검증로그:validation_logs"
        "분석결과:analysis_results"
        "최종제출:final_submission"
        "WandB로그:wandb"
    )
    
    # 각 디렉토리 삭제
    for dir_info in "${DIRS_TO_CLEAN[@]}"; do
        local dir_type="${dir_info%%:*}"
        local dir_path="${dir_info#*:}"
        local full_path="${REMOTE_BASE}/${dir_path}"
        
        log_info "$dir_type 디렉토리 삭제 중..."
        ssh "$REMOTE_HOST" "if [ -d '$full_path' ]; then rm -rf '$full_path'/* 2>/dev/null || true; echo '$dir_type 삭제 완료'; else echo '$dir_type 디렉토리가 존재하지 않습니다'; fi"
    done
    
    # 추가 정리 파일들 (config.conf 기반)
    log_info "원격 추가 파일 처리 중..."
    
    # 벤치마크 로그 파일들
    if [[ -n "$BENCHMARK_LOGS_PATTERN" ]]; then
        ssh "$REMOTE_HOST" "cd '$REMOTE_BASE' && rm -f $BENCHMARK_LOGS_PATTERN 2>/dev/null || true"
        log_info "원격 벤치마크 로그 파일 삭제: $BENCHMARK_LOGS_PATTERN"
    fi
    
    # 학습 로그 파일들
    if [[ -n "$TRAINING_LOGS_PATTERN" ]]; then
        ssh "$REMOTE_HOST" "cd '$REMOTE_BASE' && rm -f $TRAINING_LOGS_PATTERN 2>/dev/null || true"
        log_info "원격 학습 로그 파일 삭제: $TRAINING_LOGS_PATTERN"
    fi
    
    # 동기화 보고서 파일듡
    if [[ -n "$SYNC_REPORT_PATTERN" ]]; then
        ssh "$REMOTE_HOST" "cd '$REMOTE_BASE' && rm -f $SYNC_REPORT_PATTERN 2>/dev/null || true"
        log_info "원격 동기화 보고서 파일 삭제: $SYNC_REPORT_PATTERN"
    fi
    
    # 동기화 상태 추적 파일듡
    if [[ -n "$SYNCED_EXPERIMENTS_FILE" ]]; then
        ssh "$REMOTE_HOST" "cd '$REMOTE_BASE' && rm -f $SYNCED_EXPERIMENTS_FILE outputs/$SYNCED_EXPERIMENTS_FILE logs/$SYNCED_EXPERIMENTS_FILE 2>/dev/null || true"
        log_info "원격 동기화 상태 파일 삭제: $SYNCED_EXPERIMENTS_FILE"
    fi
    
    # 임시 파일듡 (기본 정리)
    ssh "$REMOTE_HOST" "cd '$REMOTE_BASE' && rm -f *.tmp 2>/dev/null || true"
    
    # 원격 Python 캐시 정리  
    if [[ "$CLEAN_PYTHON_CACHE" == "true" ]]; then
        log_info "원격 Python 캐시 파일 정리 중..."
        ssh "$REMOTE_HOST" "find '$REMOTE_BASE' -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true"
        ssh "$REMOTE_HOST" "find '$REMOTE_BASE' -name '*.pyc' -delete 2>/dev/null || true"
        log_success "✅ 원격 Python 캐시 정리 완룼"
    fi
    
    log_success "🎉 원격 서버 실험 결과 삭제 완료!"
}

# =================================================================
# 메인 실행 함수
# =================================================================

main() {
    echo ""
    log_separator
    log_info "AIStages 실험 결과 완전 삭제 스크립트"
    log_separator
    echo ""
    
    log_warning "⚠️  중요: 이 스크립트는 모든 실험 결과를 영구적으로 삭제합니다!"
    log_warning "⚠️  prediction/ 및 data/ 디렉토리는 보호됩니다."
    echo ""
    
    # 1단계: 현재 상태 분석
    analyze_cleanup_impact
    
    # 2단계: 첫 번째 확인
    confirm_action "위의 모든 실험 결과가 영구적으로 삭제됩니다."
    
    # 3단계: 두 번째 확인 (더 강한 경고)
    echo ""
    log_warning "🔥 마지막 경고: 삭제된 파일은 복구할 수 없습니다!"
    log_warning "🔥 정말로 모든 실험 결과를 삭제하시겠습니까?"
    confirm_action "최종 확인: 모든 실험 결과를 영구적으로 삭제합니다."
    
    # 4단계: 실제 삭제 실행
    echo ""
    log_separator
    log_info "삭제 작업 시작..."
    log_separator
    
    # 로컬 삭제
    cleanup_local_results
    echo ""
    
    # 원격 삭제
    cleanup_remote_results
    echo ""
    
    # 완료 메시지
    log_separator
    log_success "🎉 모든 실험 결과 삭제 완료!"
    log_separator
    echo ""
    log_info "새로운 실험을 시작할 수 있습니다."
    echo "  실험 시작: ./run_main_5_experiments.sh"
    echo "  개별 실험: python -m code.main --config config.yaml"
    echo ""
}

# 스크립트 실행
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
