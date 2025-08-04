#!/bin/bash

#####################################################################
# AIStages 서버 실험 결과 동기화 스크립트 (수정된 버전)
# 
# 수정사항:
# - 대화형 입력 제거 (환경변수로 제어)
# - SSH 명령어 단순화
# - 더 나은 에러 처리
# - 타임아웃 방지
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

# 환경변수로 동작 제어
AUTO_CLEANUP=${AUTO_CLEANUP:-false}
DRY_RUN=${DRY_RUN:-false}
SKIP_CONFIRMATION=${SKIP_CONFIRMATION:-true}  # 기본적으로 확인 건너뛰기

# =================================================================
# 동기화 대상 경로 설정 (비어있으면 제외)
# =================================================================

# 동기화 대상 디렉토리들을 배열로 정의
DIRS_TO_SYNC=()

# 경로가 비어있지 않은 디렉토리들만 동기화 대상에 추가
[[ -n "${OUTPUTS_PATH}" ]] && DIRS_TO_SYNC+=("outputs:${OUTPUTS_PATH}")
[[ -n "${LOGS_PATH}" ]] && DIRS_TO_SYNC+=("logs:${LOGS_PATH}")
[[ -n "${PREDICTION_PATH}" ]] && DIRS_TO_SYNC+=("prediction:${PREDICTION_PATH}")
[[ -n "${CHECKPOINTS_PATH}" ]] && DIRS_TO_SYNC+=("checkpoints:${CHECKPOINTS_PATH}")
[[ -n "${MODELS_PATH}" ]] && DIRS_TO_SYNC+=("models:${MODELS_PATH}")
[[ -n "${WANDB_PATH}" ]] && DIRS_TO_SYNC+=("wandb:${WANDB_PATH}")
[[ -n "${VALIDATION_LOGS_PATH}" ]] && DIRS_TO_SYNC+=("validation_logs:${VALIDATION_LOGS_PATH}")
[[ -n "${ANALYSIS_RESULTS_PATH}" ]] && DIRS_TO_SYNC+=("analysis_results:${ANALYSIS_RESULTS_PATH}")
[[ -n "${FINAL_SUBMISSION_PATH}" ]] && DIRS_TO_SYNC+=("final_submission:${FINAL_SUBMISSION_PATH}")
[[ -n "${DATA_PATH}" ]] && DIRS_TO_SYNC+=("data:${DATA_PATH}")

# 동기화 대상 디렉토리 로그 출력
log_info "동기화 대상 디렉토리: ${#DIRS_TO_SYNC[@]}개"
for dir_info in "${DIRS_TO_SYNC[@]}"; do
    dir_type="${dir_info%%:*}"
    dir_path="${dir_info#*:}"
    log_info "  - $dir_type: $dir_path"
done

# =================================================================
# 유틸리티 함수들
# =================================================================

# 서버 연결 테스트
test_server_connection() {
    log_info "서버 연결 테스트 중..."
    
    if ssh "$REMOTE_HOST" "echo '연결 성공'" >/dev/null 2>&1; then
        log_success "서버 연결 확인됨"
        return 0
    else
        log_error "서버에 연결할 수 없습니다: $REMOTE_HOST"
        return 1
    fi
}

# 디렉토리 생성
setup_directories() {
    log_info "로컬 디렉토리 설정 중..."
    
    # LOCAL_BASE 디렉토리 존재 확인
    if [[ ! -d "$LOCAL_BASE" ]]; then
        log_error "로컬 베이스 디렉토리가 존재하지 않습니다: $LOCAL_BASE"
        exit 1
    fi
    
    # 동기화 대상 디렉토리들을 LOCAL_BASE 기준으로 생성
    for dir_info in "${DIRS_TO_SYNC[@]}"; do
        local dir_type="${dir_info%%:*}"
        local dir_path="${dir_info#*:}"
        local full_path="$LOCAL_BASE/$dir_path"
        
        if [[ -n "$dir_path" ]]; then
            mkdir -p "$full_path"
            log_info "디렉토리 생성: $full_path"
        fi
    done
    
    log_success "로컬 디렉토리 생성 완료"
}

# 원격 실험 목록 가져오기
get_remote_experiment_list() {
    log_info "서버에서 동기화 대상 디렉토리 검색 중..."
    
    # 동기화 대상 디렉토리별로 검색
    for dir_info in "${DIRS_TO_SYNC[@]}"; do
        local dir_type="${dir_info%%:*}"
        local dir_path="${dir_info#*:}"
        local remote_full_path="${REMOTE_BASE}/${dir_path}"
        
        # 원격 디렉토리 존재 여부 확인
        if ssh "$REMOTE_HOST" "[ -d '$remote_full_path' ]" 2>/dev/null; then
            # 디렉토리 내에 하위 디렉토리가 있는지 확인
            local subdirs=$(ssh "$REMOTE_HOST" "find '$remote_full_path' -maxdepth 1 -type d -name '*_*' 2>/dev/null" | xargs -I {} basename {} 2>/dev/null || true)
            
            if [[ -n "$subdirs" ]]; then
                # 하위 디렉토리들 출력
                echo "$subdirs" | while read -r subdir; do
                    [[ -n "$subdir" ]] && echo "$dir_type:$subdir"
                done
            fi
            
            # 전체 디렉토리도 동기화 (루트 파일들 포함)
            echo "$dir_type:."
        else
            log_warning "⚠️  원격 $dir_type 디렉토리가 존재하지 않습니다: $remote_full_path"
        fi
    done
}

# 단일 디렉토리 동기화
sync_single_directory() {
    local dir_type="$1"  # 디렉토리 타입
    local dir_name="$2"  # 디렉토리 이름 (. 이면 전체)
    
    # 동기화 대상에서 해당 디렉토리 정보 찾기
    local dir_path=""
    for dir_info in "${DIRS_TO_SYNC[@]}"; do
        if [[ "${dir_info%%:*}" == "$dir_type" ]]; then
            dir_path="${dir_info#*:}"
            break
        fi
    done
    
    if [[ -z "$dir_path" ]]; then
        log_error "잘못된 디렉토리 타입 또는 비활성화된 디렉토리: $dir_type"
        return 1
    fi
    
    # 로컬 및 원격 경로 설정
    if [[ "$dir_name" == "." ]]; then
        # 전체 디렉토리 동기화
        local_dir="${LOCAL_BASE}/${dir_path}"
        remote_dir="${REMOTE_BASE}/${dir_path}"
        log_info "동기화 중: $dir_type 디렉토리 전체"
    else
        # 하위 디렉토리 동기화
        local_dir="${LOCAL_BASE}/${dir_path}/${dir_name}"
        remote_dir="${REMOTE_BASE}/${dir_path}/${dir_name}"
        log_info "동기화 중: $dir_type/$dir_name"
    fi
    
    # 로컬 디렉토리 생성
    mkdir -p "$local_dir"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] $remote_dir -> $local_dir"
        return 0
    fi
    
    # 원격 디렉토리 존재 여부 확인
    if ! ssh "$REMOTE_HOST" "[ -d '$remote_dir' ]" 2>/dev/null; then
        if [[ "$dir_name" == "." ]]; then
            log_warning "⚠️  원격 $dir_type 디렉토리가 존재하지 않습니다: $remote_dir"
        else
            log_warning "⚠️  원격 디렉토리가 존재하지 않습니다: $dir_type/$dir_name"
        fi
        return 1
    fi
    
    # rsync로 동기화 (개선된 옵션 - 기존 파일 유지)
    if rsync -avz --progress --update --ignore-existing \
        "$REMOTE_HOST:$remote_dir/" \
        "$local_dir/" 2>/dev/null; then
        if [[ "$dir_name" == "." ]]; then
            log_success "✅ $dir_type 디렉토리 동기화 완료"
        else
            log_success "✅ $dir_type/$dir_name 동기화 완료"
        fi
        return 0
    else
        # 더 상세한 에러 정보 수집
        local rsync_error_code=$?
        log_error "❌ rsync 에러 코드: $rsync_error_code"
        
        if [[ "$dir_name" == "." ]]; then
            log_error "❌ $dir_type 디렉토리 동기화 실패"
        else
            log_error "❌ $dir_type/$dir_name 동기화 실패"
        fi
        return 1
    fi
}

# 동기화 결과 보고서 생성
generate_sync_report() {
    local report_file="$LOCAL_BASE/sync_report_$(date +%Y%m%d_%H%M%S).txt"
    
    log_info "동기화 보고서 생성 중..."
    
    cat > "$report_file" << EOF
AIStages 실험 결과 동기화 보고서
================================================
날짜: $(date)
로컬 기본 경로: $LOCAL_BASE
원격 호스트: $REMOTE_HOST
원격 기본 경로: $REMOTE_BASE

동기화된 outputs 디렉토리들:
EOF

    # Outputs 디렉토리 정보
    if [[ -d "$LOCAL_BASE/outputs" ]]; then
        find "$LOCAL_BASE/outputs" -maxdepth 1 -type d -name '*_*' 2>/dev/null | while read -r local_dir; do
            if [[ -d "$local_dir" ]]; then
                local exp_name=$(basename "$local_dir")
                local file_count=$(find "$local_dir" -type f 2>/dev/null | wc -l)
                local total_size=$(du -sh "$local_dir" 2>/dev/null | cut -f1)
                echo "- $exp_name ($file_count 파일, $total_size)" >> "$report_file"
            fi
        done
    fi
    
    cat >> "$report_file" << EOF

동기화된 logs 디렉토리들:
EOF

    # Logs 디렉토리 정보
    if [[ -d "$LOCAL_BASE/logs" ]]; then
        find "$LOCAL_BASE/logs" -maxdepth 1 -type d -name '*_*' 2>/dev/null | while read -r local_dir; do
            if [[ -d "$local_dir" ]]; then
                local exp_name=$(basename "$local_dir")
                local file_count=$(find "$local_dir" -type f 2>/dev/null | wc -l)
                local total_size=$(du -sh "$local_dir" 2>/dev/null | cut -f1)
                echo "- $exp_name ($file_count 파일, $total_size)" >> "$report_file"
            fi
        done
    fi
    
    cat >> "$report_file" << EOF

동기화된 prediction 디렉토리 (채점용 파일들):
EOF
    
    # Prediction 디렉토리 정보
    if [[ -d "$LOCAL_BASE/prediction" ]]; then
        local csv_count=$(find "$LOCAL_BASE/prediction" -name '*.csv' 2>/dev/null | wc -l)
        local total_size=$(du -sh "$LOCAL_BASE/prediction" 2>/dev/null | cut -f1)
        echo "- 채점용 CSV 파일: $csv_count개, 전체 크기: $total_size" >> "$report_file"
    fi
    
    cat >> "$report_file" << EOF

보고서 저장 위치: $report_file
EOF
    
    log_success "동기화 보고서 저장 완료: $report_file"
    
    # 요약 표시
    log_info "=== 동기화 요약 ==="
    
    local outputs_count=$(find "$LOCAL_BASE/outputs" -maxdepth 1 -type d -name '*_*' 2>/dev/null | wc -l)
    local logs_count=$(find "$LOCAL_BASE/logs" -maxdepth 1 -type d -name '*_*' 2>/dev/null | wc -l)
    local prediction_csv_count=$(find "$LOCAL_BASE/prediction" -name '*.csv' 2>/dev/null | wc -l)
    local outputs_size=$(du -sh "$LOCAL_BASE/outputs" 2>/dev/null | cut -f1 || echo "0B")
    local logs_size=$(du -sh "$LOCAL_BASE/logs" 2>/dev/null | cut -f1 || echo "0B")
    local prediction_size=$(du -sh "$LOCAL_BASE/prediction" 2>/dev/null | cut -f1 || echo "0B")
    
    echo "Outputs 디렉토리: $outputs_count개, 크기: $outputs_size"
    echo "Logs 디렉토리: $logs_count개, 크기: $logs_size"
    echo "Prediction 디렉토리: $prediction_csv_count개 CSV 파일, 크기: $prediction_size (채점용 - 중요!)"
    echo "보고서: $report_file"
}

# =================================================================
# 메인 실행 함수
# =================================================================

main() {
    echo "======================================================"
    echo "AIStages 서버 실험 결과 동기화 (수정된 버전)"
    echo "======================================================"
    echo
    
    # 1. 초기 설정
    setup_directories
    
    # 2. 서버 연결 테스트
    if ! test_server_connection; then
        exit 1
    fi
    
    # 3. 실험 목록 가져오기
    log_info "원격 서버에서 동기화할 디렉토리 검색 중..."
    local experiments=($(get_remote_experiment_list))
    if [[ ${#experiments[@]} -eq 0 ]]; then
        log_warning "서버에서 동기화할 실험 디렉토리를 찾을 수 없습니다"
        exit 0
    fi
    
    log_info "${#experiments[@]}개의 디렉토리를 찾았습니다"
    
    # 찾은 디렉토리 목록 출력
    log_info "동기화 대상 목록:"
    for exp_entry in "${experiments[@]}"; do
        local dir_type=$(echo "$exp_entry" | cut -d: -f1)
        local dir_name=$(echo "$exp_entry" | cut -d: -f2)
        echo "  - $dir_type: $dir_name"
    done
    echo
    
    # 4. 각 실험 동기화
    local success_count=0
    for exp_entry in "${experiments[@]}"; do
        local dir_type=$(echo "$exp_entry" | cut -d: -f1)
        local dir_name=$(echo "$exp_entry" | cut -d: -f2)
        
        if sync_single_directory "$dir_type" "$dir_name"; then
            ((success_count++))
        fi
    done
    
    # 5. 동기화 결과 보고
    generate_sync_report
    
    log_success "동기화 프로세스 완료!"
    log_info "$success_count개의 디렉토리가 성공적으로 동기화되었습니다"
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
    echo "  -y, --yes      모든 확인을 자동으로 승인"
    echo
    echo "환경변수:"
    echo "  DRY_RUN=true           미리보기 모드"
    echo "  SKIP_CONFIRMATION=true 확인 건너뛰기 (기본값)"
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
        -y|--yes)
            SKIP_CONFIRMATION=true
            shift
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
fi

main
