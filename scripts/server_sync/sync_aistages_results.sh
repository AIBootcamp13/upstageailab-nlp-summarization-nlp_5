#!/bin/bash

#####################################################################
# AIStages 서버 실험 결과 동기화 및 정리 스크립트
# 
# 용도: AIStages 서버에서 실험 결과를 로컬 맥으로 복사하고 검증 후 서버에서 삭제
# 작성자: LYJ
# 날짜: 2025-07-29
#####################################################################

set -e  # 에러 발생 시 스크립트 중단

# =================================================================
# 설정 파일 로드
# =================================================================

# 스크립트 디렉토리 경로
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/config.conf"

# 색상 코드 (설정 파일 로드 전에 정의)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 로깅 함수들 (설정 파일 로드 전에 정의)
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

# 설정 파일 존재 여부 확인
if [[ ! -f "$CONFIG_FILE" ]]; then
    log_error "설정 파일을 찾을 수 없습니다: $CONFIG_FILE"
    log_error "다음 명령으로 설정 파일을 생성하세요:"
    log_error "cp ${SCRIPT_DIR}/config.conf.template ${SCRIPT_DIR}/config.conf"
    log_error "그 후 설정 파일을 수정하여 사용하세요."
    exit 1
fi

# 설정 파일 로드
source "$CONFIG_FILE"

# 필수 설정 검증
if [[ -z "$LOCAL_BASE" ]] || [[ -z "$REMOTE_BASE" ]] || [[ -z "$REMOTE_HOST" ]]; then
    log_error "필수 설정이 누락되었습니다. config.conf 파일을 확인하세요."
    log_error "LOCAL_BASE, REMOTE_BASE, REMOTE_HOST가 모두 설정되어야 합니다."
    exit 1
fi

# 경로 설정 (설정 파일 기반)
LOCAL_OUTPUTS_DIR="${LOCAL_BASE}/${OUTPUTS_SUBDIR:-outputs}"
LOCAL_LOGS_DIR="${LOCAL_BASE}/${LOGS_SUBDIR:-logs}"
LOCAL_WANDB_DIR="${LOCAL_BASE}/${WANDB_SUBDIR:-wandb}"
LOCAL_MODELS_DIR="${LOCAL_BASE}/${MODELS_SUBDIR:-models}"
LOCAL_DATA_DIR="${LOCAL_BASE}/${DATA_SUBDIR:-data}"

# 원격 경로 설정
REMOTE_OUTPUTS_DIR="${REMOTE_BASE}/${OUTPUTS_SUBDIR:-outputs}"
REMOTE_LOGS_DIR="${REMOTE_BASE}/${LOGS_SUBDIR:-logs}"
REMOTE_WANDB_DIR="${REMOTE_BASE}/${WANDB_SUBDIR:-wandb}"
REMOTE_MODELS_DIR="${REMOTE_BASE}/${MODELS_SUBDIR:-models}"
REMOTE_DATA_DIR="${REMOTE_BASE}/${DATA_SUBDIR:-data}"

# =================================================================
# 유틸리티 함수들
# =================================================================

# 파일 크기 비교 함수
compare_file_sizes() {
    local local_file="$1"
    local remote_path="$2"
    
    if [[ ! -f "$local_file" ]]; then
        return 1  # 로컬 파일이 없음
    fi
    
    local_size=$(stat -f%z "$local_file" 2>/dev/null || echo "0")
    remote_size=$(ssh "$REMOTE_HOST" "LC_ALL=C stat -c%s '$remote_path' 2>/dev/null || echo '0'")
    
    if [[ "$local_size" -eq "$remote_size" ]] && [[ "$local_size" -gt 0 ]]; then
        return 0  # 크기가 같고 0이 아님
    else
        return 1  # 크기가 다르거나 0임
    fi
}

# 디렉토리 구조 비교 함수
verify_directory_sync() {
    local local_dir="$1"
    local remote_dir="$2"
    local dir_name="$3"
    
    log_info "$dir_name 동기화 검증 중..."
    
    # 로컬 파일 수 계산
    local_file_count=$(find "$local_dir" -type f 2>/dev/null | wc -l)
    
    # 원격 파일 수 계산
    remote_file_count=$(ssh "$REMOTE_HOST" "LC_ALL=C find '$remote_dir' -type f 2>/dev/null | wc -l")
    
    log_info "$dir_name - 로컬 파일: $local_file_count개, 원격 파일: $remote_file_count개"
    
    if [[ "$local_file_count" -eq "$remote_file_count" ]] && [[ "$local_file_count" -gt 0 ]]; then
        return 0
    else
        return 1
    fi
}

# =================================================================
# 메인 동기화 함수들
# =================================================================

# 1. 초기 설정
setup_directories() {
    log_info "로컬 디렉토리 설정 중..."
    
    mkdir -p "$LOCAL_OUTPUTS_DIR"
    mkdir -p "$LOCAL_LOGS_DIR"
    mkdir -p "$LOCAL_WANDB_DIR"
    mkdir -p "$LOCAL_MODELS_DIR"
    mkdir -p "$LOCAL_DATA_DIR"
    
    log_success "로컬 디렉토리 생성 완료"
    log_info "로컬 경로: $LOCAL_BASE"
    log_info "원격 경로: $REMOTE_BASE"
}

# 2. 서버 연결 테스트
test_server_connection() {
    log_info "서버 연결 테스트 중..."
    
    if ssh "$REMOTE_HOST" "LC_ALL=C echo '연결 성공'" >/dev/null 2>&1; then
        log_success "서버 연결 확인됨"
        return 0
    else
        log_error "서버에 연결할 수 없습니다: $REMOTE_HOST"
        log_error "SSH 설정과 서버 주소를 확인해주세요"
        return 1
    fi
}

# 3. 원격 실험 결과 디렉토리 목록 가져오기
get_remote_experiment_list() {
    local experiment_dirs=()
    
    # outputs 디렉토리의 실험 폴더들 찾기
    local outputs_dirs
    outputs_dirs=$(ssh "$REMOTE_HOST" "LC_ALL=C find '$REMOTE_OUTPUTS_DIR' -maxdepth 1 -type d -name '*_*' 2>/dev/null" | \
        grep -E '(dialogue_summarization|auto_experiments|.*_[0-9]{8}_[0-9]{6})' | \
        sort)
    
    # logs 디렉토리의 실험 로그 폴더들도 찾기 (실패한 실험 포함)
    local log_dirs
    log_dirs=$(ssh "$REMOTE_HOST" "LC_ALL=C find '$REMOTE_LOGS_DIR' -maxdepth 1 -type d -name '*experiments*' 2>/dev/null" | \
        sort)
    
    # outputs 결과를 배열에 추가
    if [[ -n "$outputs_dirs" ]]; then
        while IFS= read -r dir; do
            experiment_dirs+=("$dir")
        done <<< "$outputs_dirs"
    fi
    
    # logs 결과를 배열에 추가 (중복 제거)
    if [[ -n "$log_dirs" ]]; then
        while IFS= read -r dir; do
            # 이미 존재하지 않는 경우에만 추가
            local already_exists=false
            for existing_dir in "${experiment_dirs[@]}"; do
                if [[ "$(basename "$dir")" == "$(basename "$existing_dir")" ]]; then
                    already_exists=true
                    break
                fi
            done
            if [[ "$already_exists" == "false" ]]; then
                experiment_dirs+=("$dir")
            fi
        done <<< "$log_dirs"
    fi
    
    # 결과 출력
    printf '%s\n' "${experiment_dirs[@]}"
}

# 4. 개별 실험 디렉토리 동기화
sync_experiment_directory() {
    local remote_exp_dir="$1"
    local exp_name=$(basename "$remote_exp_dir")
    
    # 로그 디렉토리인지 판단
    local is_log_dir=false
    local local_exp_dir
    
    if [[ "$remote_exp_dir" == *"/logs/"* ]]; then
        is_log_dir=true
        local_exp_dir="${LOCAL_LOGS_DIR}/${exp_name}"
        log_info "실험 로그 동기화 중: $exp_name (로그 디렉토리)"
    else
        local_exp_dir="${LOCAL_OUTPUTS_DIR}/${exp_name}"
        log_info "실험 동기화 중: $exp_name"
    fi
    
    # 디렉토리 이름 유효성 검증 (영문, 숫자, 언더스코어, 하이픈만 허용)
    if [[ -z "$exp_name" ]] || [[ ! "$exp_name" =~ ^[a-zA-Z0-9_-]+$ ]]; then
        log_error "잘못된 디렉토리 이름: $exp_name"
        return 1
    fi
    
    # 로컬 디렉토리 생성
    mkdir -p "$local_exp_dir"
    
    # rsync로 동기화 (기존 파일은 건너뛰기)
    # 로케일 설정을 통해 한글 경로 문제 방지
    LC_ALL=C rsync -avz --progress --ignore-existing \
        "$REMOTE_HOST:$remote_exp_dir/" \
        "$local_exp_dir/" || {
        log_error "$exp_name 동기화 실패"
        return 1
    }
    
    # 동기화 검증
    if verify_directory_sync "$local_exp_dir" "$remote_exp_dir" "$exp_name"; then
        log_success "✅ $exp_name 동기화 및 검증 완료"
        if [[ "$is_log_dir" == "true" ]]; then
            echo "$remote_exp_dir" >> "${LOCAL_LOGS_DIR}/.synced_experiments"
        else
            echo "$remote_exp_dir" >> "${LOCAL_OUTPUTS_DIR}/.synced_experiments"
        fi
        return 0
    else
        log_warning "⚠️ $exp_name 동기화가 불완전할 수 있습니다"
        return 1
    fi
}

# 5. 추가 로그 파일들 동기화
sync_additional_logs() {
    log_info "추가 로그 파일 동기화 중..."
    
    if ssh "$REMOTE_HOST" "LC_ALL=C [ -d '$REMOTE_LOGS_DIR' ]" 2>/dev/null; then
        log_info "로그 디렉토리 동기화: logs"
        LC_ALL=C rsync -avz --progress --ignore-existing \
            "$REMOTE_HOST:$REMOTE_LOGS_DIR/" \
            "$LOCAL_LOGS_DIR/" || true
    fi
    
    # wandb 디렉토리 동기화
    if ssh "$REMOTE_HOST" "LC_ALL=C [ -d '$REMOTE_WANDB_DIR' ]" 2>/dev/null; then
        log_info "WandB 로그 동기화: wandb"
        LC_ALL=C rsync -avz --progress --ignore-existing \
            --include="*/" --include="*.log" --include="*.json" --include="*.csv" \
            --exclude="*" \
            "$REMOTE_HOST:$REMOTE_WANDB_DIR/" \
            "$LOCAL_WANDB_DIR/" || true
    fi
    
    # models 디렉토리 동기화
    if ssh "$REMOTE_HOST" "LC_ALL=C [ -d '$REMOTE_MODELS_DIR' ]" 2>/dev/null; then
        log_info "모델 디렉토리 동기화: models"
        LC_ALL=C rsync -avz --progress --ignore-existing \
            "$REMOTE_HOST:$REMOTE_MODELS_DIR/" \
            "$LOCAL_MODELS_DIR/" || true
    
    fi
    
    # data 디렉토리 동기화 (CSV 및 데이터 파일들)
    if ssh "$REMOTE_HOST" "LC_ALL=C [ -d '$REMOTE_DATA_DIR' ]" 2>/dev/null; then
        log_info "데이터 디렉토리 동기화: data"
        LC_ALL=C rsync -avz --progress --ignore-existing \
            "$REMOTE_HOST:$REMOTE_DATA_DIR/" \
            "$LOCAL_DATA_DIR/" || true
    fi
}

# 6. 동기화 완료된 실험 결과 서버에서 삭제
cleanup_remote_experiments() {
    local synced_file="${LOCAL_OUTPUTS_DIR}/.synced_experiments"
    
    if [[ ! -f "$synced_file" ]]; then
        log_warning "정리할 실험이 없습니다"
        return 0
    fi
    
    log_info "서버에서 동기화된 실험 결과 정리 중..."
    
    # 사용자 확인
    echo -e "${YELLOW}다음 실험들이 서버에서 삭제됩니다:${NC}"
    cat "$synced_file"
    echo
    read -p "정말 서버에서 삭제하시겠습니까? (y/N): " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "사용자가 정리를 취소했습니다"
        return 0
    fi
    
    # 각 동기화된 실험 삭제
    while IFS= read -r remote_dir; do
        if [[ -n "$remote_dir" ]]; then
            local exp_name=$(basename "$remote_dir")
            log_info "서버에서 $exp_name 삭제 중..."
            
            if ssh "$REMOTE_HOST" "LC_ALL=C rm -rf '$remote_dir'" 2>/dev/null; then
                log_success "✅ 서버에서 $exp_name 삭제 완료"
            else
                log_error "❌ 서버에서 $exp_name 삭제 실패"
            fi
        fi
    done < "$synced_file"
    
    # 정리 완료 후 추적 파일 삭제
    rm -f "$synced_file"
    log_success "서버 정리 완료"
}

# 7. 동기화 결과 요약 생성
generate_sync_report() {
    local report_file="${LOCAL_BASE}/sync_report_$(date +%Y%m%d_%H%M%S).txt"
    
    log_info "동기화 보고서 생성 중..."
    
    cat > "$report_file" << EOF
AIStages 실험 결과 동기화 보고서
================================================
날짜: $(date)
로컬 기본 경로: $LOCAL_BASE
원격 호스트: $REMOTE_HOST
원격 기본 경로: $REMOTE_BASE

동기화된 디렉토리들:
EOF
    
    find "$LOCAL_OUTPUTS_DIR" -maxdepth 1 -type d -name '*_*' 2>/dev/null | while read -r local_dir; do
        if [[ -d "$local_dir" ]]; then
            local exp_name=$(basename "$local_dir")
            local file_count=$(find "$local_dir" -type f | wc -l)
            local total_size=$(du -sh "$local_dir" | cut -f1)
            
            echo "- $exp_name ($file_count 파일, $total_size)" >> "$report_file"
        fi
    done
    
    cat >> "$report_file" << EOF

주요 파일 위치:
- 학습 결과: outputs/*/results/training_results.json
- 요약 정보: outputs/*/results/summary.txt
- 실험 정보: outputs/*/experiments/*/experiment_info.json
- 모델 레지스트리: outputs/*/models/models.json
- 자동 실험 결과: outputs/auto_experiments/experiment_results_*.json
- 실험 결과 CSV: outputs/auto_experiments/csv_results/*.csv
- 로그 파일: logs/
- WandB 로그: wandb/ (*.log, *.json, *.csv)
- 모델 파일: models/
- 데이터 파일: data/ (CSV 및 기타 데이터 파일)

보고서 저장 위치: $report_file
EOF
    
    log_success "동기화 보고서 저장 완료: $report_file"
    
    # 요약 표시
    echo
    log_info "=== 동기화 요약 ==="
    local total_dirs=$(find "$LOCAL_OUTPUTS_DIR" -maxdepth 1 -type d -name '*_*' 2>/dev/null | wc -l)
    local outputs_size=$(du -sh "$LOCAL_OUTPUTS_DIR" 2>/dev/null | cut -f1 || echo "0B")
    local logs_size=$(du -sh "$LOCAL_LOGS_DIR" 2>/dev/null | cut -f1 || echo "0B")
    local wandb_size=$(du -sh "$LOCAL_WANDB_DIR" 2>/dev/null | cut -f1 || echo "0B")
    local models_size=$(du -sh "$LOCAL_MODELS_DIR" 2>/dev/null | cut -f1 || echo "0B")
    local data_size=$(du -sh "$LOCAL_DATA_DIR" 2>/dev/null | cut -f1 || echo "0B")
    echo "총 실험 디렉토리: $total_dirs개"
    echo "Outputs 크기: $outputs_size"
    echo "Logs 크기: $logs_size"
    echo "WandB 크기: $wandb_size"
    echo "Models 크기: $models_size"
    echo "Data 크기: $data_size"
    echo "보고서: $report_file"
}

# =================================================================
# 메인 실행 함수
# =================================================================

main() {
    echo "======================================================"
    echo "AIStages 서버 실험 결과 동기화"
    echo "======================================================"
    echo
    
    # 1. 초기 설정
    setup_directories
    
    # 2. 서버 연결 테스트
    if ! test_server_connection; then
        exit 1
    fi
    
    # 3. 실험 목록 가져오기
    log_info "서버에서 실험 디렉토리 검색 중..."
    local experiments=($(get_remote_experiment_list))
    if [[ ${#experiments[@]} -eq 0 ]]; then
        log_warning "서버에서 실험을 찾을 수 없습니다"
        exit 0
    fi
    
    log_info "${#experiments[@]}개의 실험 디렉토리를 찾았습니다"
    
    # 4. 각 실험 동기화
    local success_count=0
    for exp_dir in "${experiments[@]}"; do
        # 유효한 디렉토리 경로인지 검증 (로그 디렉토리도 포함)
        if [[ "$exp_dir" =~ ^/.*/(dialogue_summarization|auto_experiments|.*_[0-9]{8}_[0-9]{6}|.*experiments.*)$ ]]; then
            if sync_experiment_directory "$exp_dir"; then
                ((success_count++))
            fi
        else
            log_warning "잘못된 디렉토리 경로 건너뛰기: $exp_dir"
        fi
    done
    
    # 5. 추가 로그 동기화
    sync_additional_logs
    
    # 6. 동기화 결과 보고
    generate_sync_report
    
    # 7. 서버 정리 (선택사항)
    if [[ $success_count -gt 0 ]]; then
        echo
        cleanup_remote_experiments
    fi
    
    log_success "동기화 프로세스 완료!"
    log_info "$success_count개의 실험이 성공적으로 동기화되었습니다"
}

# =================================================================
# 스크립트 실행
# =================================================================

# 사용법 출력 함수
show_usage() {
    echo "사용법: $0 [옵션]"
    echo
    echo "옵션:"
    echo "  -h, --help     이 도움말 출력"
    echo "  -d, --dry-run  실제 작업 없이 미리보기만 실행"
    echo "  -s, --server   서버 주소 지정 (기본값: config.conf에서 설정)"
    echo
    echo "예시:"
    echo "  $0                                    # 기본 동기화 실행"
    echo "  $0 --dry-run                         # 미리보기 모드"
    echo "  $0 --server aistages                 # 다른 서버 지정"
    echo
    echo "설정:"
    echo "  - 먼저 config.conf.template을 복사하여 config.conf를 만드세요"
    echo "  - config.conf에서 LOCAL_BASE와 REMOTE_BASE를 설정하세요"
    echo
}

# 명령행 인자 처리
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -d|--dry-run)
            DRY_RUN=true
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

# 스크립트 시작
if [[ "${DRY_RUN:-false}" == "true" ]]; then
    log_info "미리보기 모드로 실행 중 - 실제 작업은 수행되지 않습니다"
    # 미리보기 로직 구현 (생략)
    exit 0
else
    main
fi
