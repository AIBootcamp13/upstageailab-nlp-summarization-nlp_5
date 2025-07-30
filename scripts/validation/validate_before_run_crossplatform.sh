#!/bin/bash
# 크로스 플랫폼 (macOS/Ubuntu) 실험 환경 검증 스크립트
# 이 스크립트는 실험 실행 전에 모든 환경을 점검하고 문제를 사전에 감지합니다.

set -e  # 오류 발생 시 즉시 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# OS 감지
OS_TYPE="unknown"
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS_TYPE="macos"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS_TYPE="linux"
fi

# 로그 파일 설정
LOG_DIR="./validation_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/validation_$(date +%Y%m%d_%H%M%S).log"

# 로깅 함수
log() {
    echo "$1" | tee -a "$LOG_FILE"
}

print_header() {
    echo -e "\n${BOLD}${BLUE}============================================================${NC}" | tee -a "$LOG_FILE"
    echo -e "${BOLD}${BLUE}$1${NC}" | tee -a "$LOG_FILE"
    echo -e "${BOLD}${BLUE}============================================================${NC}" | tee -a "$LOG_FILE"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1" | tee -a "$LOG_FILE"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1" | tee -a "$LOG_FILE"
}

print_error() {
    echo -e "${RED}✗${NC} $1" | tee -a "$LOG_FILE"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1" | tee -a "$LOG_FILE"
}

# 전역 변수
ERRORS=0
WARNINGS=0

# 1. 시스템 정보 확인
check_system_info() {
    print_header "시스템 정보 확인"
    
    # OS 정보
    if [[ "$OS_TYPE" == "macos" ]]; then
        OS_VERSION=$(sw_vers -productVersion)
        print_info "OS: macOS $OS_VERSION"
        print_success "macOS 시스템 확인됨"
    elif [[ "$OS_TYPE" == "linux" ]]; then
        if [[ -f /etc/os-release ]]; then
            . /etc/os-release
            print_info "OS: $NAME $VERSION"
            
            if [[ "$ID" == "ubuntu" ]]; then
                print_success "Ubuntu 시스템 확인됨"
            else
                print_warning "Ubuntu가 아닌 Linux: $ID"
                ((WARNINGS++))
            fi
        fi
    else
        print_error "지원하지 않는 OS: $OSTYPE"
        ((ERRORS++))
    fi
    
    # 하드웨어 정보
    if [[ "$OS_TYPE" == "macos" ]]; then
        # macOS
        CPU_COUNT=$(sysctl -n hw.physicalcpu)
        print_info "CPU: $CPU_COUNT cores"
        
        # 메모리 (macOS)
        MEM_TOTAL=$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))
        print_info "메모리: ${MEM_TOTAL}GB total"
        
        # 디스크 (macOS)
        DISK_INFO=$(df -h . | tail -1 | awk '{print $4}')
        print_info "디스크: $DISK_INFO available"
    else
        # Linux
        print_info "CPU: $(nproc) cores"
        print_info "메모리: $(free -h | grep '^Mem:' | awk '{print $2}') total, $(free -h | grep '^Mem:' | awk '{print $7}') available"
        print_info "디스크: $(df -h . | tail -1 | awk '{print $4}') available"
    fi
    
    # GPU 정보
    if [[ "$OS_TYPE" == "macos" ]]; then
        # macOS - MPS 지원 확인
        if python3 -c "import torch; print(torch.backends.mps.is_available())" 2>/dev/null | grep -q "True"; then
            print_success "Apple Silicon GPU (MPS) 사용 가능"
        else
            print_info "MPS를 사용할 수 없음 (Intel Mac 또는 PyTorch 버전 확인 필요)"
        fi
    else
        # Linux - NVIDIA GPU
        if command -v nvidia-smi &> /dev/null; then
            GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
            print_success "GPU 감지됨: $GPU_COUNT 개"
            nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | while read line; do
                print_info "  - $line"
            done
        else
            print_warning "nvidia-smi를 찾을 수 없음 (GPU 없거나 드라이버 미설치)"
            ((WARNINGS++))
        fi
    fi
}

# 2. Python 환경 확인
check_python_env() {
    print_header "Python 환경 확인"
    
    # Python 버전
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
        print_info "Python 버전: $PYTHON_VERSION"
        
        # 버전 확인 (3.8 이상)
        MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
        
        if [[ $MAJOR -ge 3 && $MINOR -ge 8 ]]; then
            print_success "Python 버전 적합 (3.8+)"
        else
            print_error "Python 3.8 이상이 필요합니다"
            ((ERRORS++))
        fi
    else
        print_error "Python3를 찾을 수 없습니다"
        ((ERRORS++))
        return 1
    fi
    
    # 가상환경 확인
    if [[ -n "$VIRTUAL_ENV" ]]; then
        print_success "가상환경 활성화됨: $VIRTUAL_ENV"
    elif [[ -n "$CONDA_DEFAULT_ENV" ]]; then
        print_success "Conda 환경 활성화됨: $CONDA_DEFAULT_ENV"
    else
        print_warning "가상환경이 활성화되지 않음"
        ((WARNINGS++))
    fi
    
    # uv 패키지 매니저 확인 (필수)
    if ! command -v uv &> /dev/null; then
        print_error "uv 패키지 매니저가 설치되지 않았습니다"
        print_error "uv를 먼저 설치해주세요: https://github.com/astral-sh/uv"
        ((ERRORS++))
    else
        print_success "uv 패키지 매니저 사용 가능"
        
        # uv pip 확인
        if uv pip list &> /dev/null; then
            print_success "uv pip 사용 가능"
        else
            print_error "uv pip를 사용할 수 없습니다"
            print_warning "가상환경이 활성화되었는지 확인해주세요"
            ((ERRORS++))
        fi
    fi
}

# 3. 프로젝트 구조 확인
check_project_structure() {
    print_header "프로젝트 구조 확인"
    
    # 필수 디렉토리
    REQUIRED_DIRS=("code" "config" "data" "models" "outputs" "logs" "scripts")
    
    for dir in "${REQUIRED_DIRS[@]}"; do
        if [[ -d "$dir" ]]; then
            print_success "디렉토리 존재: $dir"
        else
            print_error "디렉토리 누락: $dir"
            ((ERRORS++))
        fi
    done
    
    # 필수 파일
    REQUIRED_FILES=(
        "requirements.txt"
        "config.yaml"
        "scripts/experiments/run_auto_experiments.sh"
        "code/trainer.py"
        "code/auto_experiment_runner.py"
    )
    
    for file in "${REQUIRED_FILES[@]}"; do
        if [[ -f "$file" ]]; then
            print_success "파일 존재: $file"
        else
            print_error "파일 누락: $file"
            ((ERRORS++))
        fi
    done
}

# 4. Python 패키지 확인
check_python_packages() {
    print_header "Python 패키지 확인"
    
    # requirements.txt 존재 확인
    if [[ ! -f "requirements.txt" ]]; then
        print_error "requirements.txt 파일이 없습니다"
        ((ERRORS++))
        return 1
    fi
    
    # 핵심 패키지 확인
    CORE_PACKAGES=(
        "torch"
        "transformers"
        "datasets"
        "pandas"
        "numpy"
        "wandb"
        "yaml:pyyaml"  # import명:표시명
        "tqdm"
    )
    
    for package_info in "${CORE_PACKAGES[@]}"; do
        if [[ "$package_info" == *":"* ]]; then
            # import명과 표시명이 다른 경우
            import_name="${package_info%%:*}"
            display_name="${package_info##*:}"
        else
            import_name="$package_info"
            display_name="$package_info"
        fi
        
        if python3 -c "import $import_name" 2>/dev/null; then
            VERSION=$(python3 -c "import $import_name; print(getattr($import_name, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")
            print_success "$display_name: $VERSION"
        else
            print_error "$display_name: 설치되지 않음"
            ((ERRORS++))
        fi
    done
    
    # PyTorch 디바이스 확인
    print_info "PyTorch 디바이스 지원 확인..."
    
    if [[ "$OS_TYPE" == "macos" ]]; then
        # macOS - MPS 확인
        MPS_AVAILABLE=$(python3 -c "import torch; print(torch.backends.mps.is_available())" 2>/dev/null || echo "False")
        if [[ "$MPS_AVAILABLE" == "True" ]]; then
            print_success "PyTorch MPS (Metal Performance Shaders) 사용 가능"
        else
            print_warning "PyTorch MPS를 사용할 수 없음 (CPU 모드로 실행됨)"
            ((WARNINGS++))
        fi
    else
        # Linux - CUDA 확인
        CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
        if [[ "$CUDA_AVAILABLE" == "True" ]]; then
            CUDA_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
            print_success "PyTorch CUDA 사용 가능 (GPU: $CUDA_COUNT개)"
        else
            print_warning "PyTorch CUDA를 사용할 수 없음"
            ((WARNINGS++))
        fi
    fi
}

# 5. 데이터 파일 확인
check_data_files() {
    print_header "데이터 파일 확인"
    
    DATA_DIR="./data"
    REQUIRED_DATA=("train.csv" "dev.csv" "test.csv")
    
    for datafile in "${REQUIRED_DATA[@]}"; do
        filepath="$DATA_DIR/$datafile"
        if [[ -f "$filepath" ]]; then
            # 파일 크기 확인
            size=$(du -h "$filepath" | cut -f1)
            lines=$(wc -l < "$filepath")
            print_success "$datafile: $size, $lines lines"
            
            # CSV 헤더 확인
            if [[ -f "$filepath" ]]; then
                header=$(head -n1 "$filepath")
                if [[ "$datafile" == "test.csv" ]]; then
                    # test.csv는 fname과 dialogue만 필요
                    if [[ "$header" == *"fname"* && "$header" == *"dialogue"* ]]; then
                        print_success "  → 필수 컬럼 확인됨"
                    else
                        print_error "  → 필수 컬럼 누락 (fname, dialogue 필요)"
                        ((ERRORS++))
                    fi
                else
                    # train.csv와 dev.csv는 summary도 필요
                    if [[ "$header" == *"fname"* && "$header" == *"dialogue"* && "$header" == *"summary"* ]]; then
                        print_success "  → 필수 컬럼 확인됨"
                    else
                        print_error "  → 필수 컬럼 누락 (fname, dialogue, summary 필요)"
                        ((ERRORS++))
                    fi
                fi
            fi
        else
            print_error "$datafile: 파일 없음"
            ((ERRORS++))
        fi
    done
}

# 6. 설정 파일 검증
check_config_files() {
    print_header "설정 파일 검증"
    
    # 기본 설정 파일
    if [[ -f "config.yaml" ]]; then
        print_info "config.yaml 파싱 중..."
        if python3 -c "import yaml; yaml.safe_load(open('config.yaml'))" 2>/dev/null; then
            print_success "config.yaml: 유효함"
        else
            print_error "config.yaml: 파싱 실패"
            ((ERRORS++))
        fi
    fi
    
    # base_config.yaml
    if [[ -f "config/base_config.yaml" ]]; then
        if python3 -c "import yaml; yaml.safe_load(open('config/base_config.yaml'))" 2>/dev/null; then
            print_success "base_config.yaml: 유효함"
        else
            print_error "base_config.yaml: 파싱 실패"
            ((ERRORS++))
        fi
    fi
    
    # 실험 설정 파일들
    if [[ -d "config/experiments" ]]; then
        exp_count=$(find config/experiments -name "*.yaml" -o -name "*.yml" | wc -l)
        print_info "실험 설정 파일: $exp_count개 발견"
        
        # 각 파일 검증
        find config/experiments -name "*.yaml" -o -name "*.yml" | while read exp_file; do
            if python3 -c "import yaml; yaml.safe_load(open('$exp_file'))" 2>/dev/null; then
                echo -n "  ✓ $(basename $exp_file)" >> "$LOG_FILE"
            else
                print_error "  $(basename $exp_file): 파싱 실패"
                ((ERRORS++))
            fi
        done
        echo "" >> "$LOG_FILE"
    else
        print_warning "config/experiments 디렉토리가 없습니다"
        ((WARNINGS++))
    fi
}

# 7. 메모리 및 디스크 공간 확인
check_resources() {
    print_header "시스템 리소스 확인"
    
    # 메모리 확인
    if [[ "$OS_TYPE" == "macos" ]]; then
        # macOS - vm_stat 사용
        MEM_PAGE_SIZE=$(vm_stat | grep "page size" | awk '{print $8}')
        MEM_FREE_PAGES=$(vm_stat | grep "Pages free" | awk '{print $3}' | sed 's/\.//')
        MEM_INACTIVE_PAGES=$(vm_stat | grep "Pages inactive" | awk '{print $3}' | sed 's/\.//')
        MEM_AVAILABLE_GB=$(( (MEM_FREE_PAGES + MEM_INACTIVE_PAGES) * MEM_PAGE_SIZE / 1024 / 1024 / 1024 ))
    else
        # Linux - free 명령어 사용
        MEM_AVAILABLE_GB=$(free -g | grep '^Mem:' | awk '{print $7}')
    fi
    
    if [[ $MEM_AVAILABLE_GB -lt 16 ]]; then
        print_warning "사용 가능한 메모리가 16GB 미만입니다 ($MEM_AVAILABLE_GB GB)"
        ((WARNINGS++))
    else
        print_success "메모리 충분: $MEM_AVAILABLE_GB GB 사용 가능"
    fi
    
    # 디스크 공간 확인
    DISK_AVAILABLE=$(df -BG . 2>/dev/null | tail -1 | awk '{print $4}' | sed 's/G//' || df -g . | tail -1 | awk '{print $4}')
    if [[ $DISK_AVAILABLE -lt 50 ]]; then
        print_warning "디스크 여유 공간이 50GB 미만입니다 ($DISK_AVAILABLE GB)"
        ((WARNINGS++))
    else
        print_success "디스크 공간 충분: $DISK_AVAILABLE GB 여유"
    fi
    
    # CPU 사용률 확인
    if [[ "$OS_TYPE" == "macos" ]]; then
        # macOS - top 명령어가 다르게 동작
        CPU_USAGE=$(ps aux | awk '{sum+=$3} END {print sum}')
        CPU_USAGE=${CPU_USAGE%.*}  # 소수점 제거
    else
        # Linux
        CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')
    fi
    print_info "현재 CPU 사용률: ${CPU_USAGE}%"
}

# 8. 실행 권한 확인
check_permissions() {
    print_header "파일 권한 확인"
    
    # 실행 스크립트 권한
    SCRIPTS=("scripts/experiments/run_auto_experiments.sh" "setup_env.sh" "check_env.sh")
    
    for script in "${SCRIPTS[@]}"; do
        if [[ -f "$script" ]]; then
            if [[ -x "$script" ]]; then
                print_success "$script: 실행 권한 있음"
            else
                print_warning "$script: 실행 권한 없음 (수정 중...)"
                chmod +x "$script"
                if [[ -x "$script" ]]; then
                    print_success "  → 실행 권한 부여됨"
                else
                    print_error "  → 권한 수정 실패"
                    ((ERRORS++))
                fi
            fi
        fi
    done
    
    # 쓰기 권한 확인
    WRITE_DIRS=("outputs" "logs" "models")
    
    for dir in "${WRITE_DIRS[@]}"; do
        if [[ -d "$dir" ]]; then
            if touch "$dir/.write_test" 2>/dev/null; then
                rm -f "$dir/.write_test"
                print_success "$dir: 쓰기 권한 있음"
            else
                print_error "$dir: 쓰기 권한 없음"
                ((ERRORS++))
            fi
        fi
    done
}

# 9. 빠른 코드 테스트
quick_code_test() {
    print_header "빠른 코드 테스트"
    
    # Python 스크립트로 빠른 테스트 실행
    cat > /tmp/quick_test.py << 'EOF'
import sys
sys.path.insert(0, '.')

try:
    # 주요 모듈 import 테스트
    from code.utils import load_config
    from code.utils.device_utils import get_optimal_device
    print("✓ 유틸리티 모듈 import 성공")
    
    # 디바이스 확인
    device, device_info = get_optimal_device()
    print(f"✓ 최적 디바이스: {device}")
    
    # 설정 파일 로딩 테스트
    config = load_config("config/base_config.yaml")
    print("✓ 설정 파일 로딩 성공")
    
    print("\n모든 빠른 테스트 통과!")
    sys.exit(0)
    
except Exception as e:
    print(f"✗ 테스트 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF
    
    if python3 /tmp/quick_test.py 2>&1 | tee -a "$LOG_FILE"; then
        print_success "코드 무결성 확인 완료"
    else
        print_error "코드 테스트 실패"
        ((ERRORS++))
    fi
    
    rm -f /tmp/quick_test.py
}

# 10. 사전 실행 테스트
run_prerun_test() {
    print_header "사전 실행 테스트"
    
    if [[ -f "scripts/validation/prerun_test.py" ]]; then
        print_info "사전 실행 테스트 스크립트 실행 중..."
        if python3 scripts/validation/prerun_test.py --quick 2>&1 | tee -a "$LOG_FILE"; then
            print_success "사전 실행 테스트 통과"
        else
            print_error "사전 실행 테스트 실패"
            ((ERRORS++))
        fi
    else
        print_warning "prerun_test.py 파일이 없어 테스트 생략"
        ((WARNINGS++))
    fi
}

# 11. 다양한 모델 지원 검증 (송규헌님 요청사항)
check_multi_model_support() {
    print_header "다양한 모델 지원 검증 (송규헌님 요청사항)"
    
    if [[ -f "scripts/validation/validate_multi_model_support.py" ]]; then
        print_info "다양한 모델 지원 검증 중..."
        if python3 scripts/validation/validate_multi_model_support.py 2>&1 | tee -a "$LOG_FILE"; then
            print_success "다양한 모델 지원 검증 통과"
        else
            print_error "다양한 모델 지원 검증 실패"
            ((ERRORS++))
        fi
    else
        print_warning "validate_multi_model_support.py 파일이 없음"
        print_info "다음 항목들을 수동으로 확인하세요:"
        print_info "  - trainer.py의 AutoModelForSeq2SeqLM/AutoModelForCausalLM 지원"
        print_info "  - config/model_configs/ 디렉토리의 모델별 설정 파일"
        print_info "  - unsloth 라이브러리 지원 (선택사항)"
        ((WARNINGS++))
    fi
}

# 최종 요약
print_summary() {
    print_header "검증 결과 요약"
    
    echo -e "${BOLD}실행 환경: ${OS_TYPE}${NC}" | tee -a "$LOG_FILE"
    echo -e "${BOLD}총 오류: ${RED}$ERRORS${NC}" | tee -a "$LOG_FILE"
    echo -e "${BOLD}총 경고: ${YELLOW}$WARNINGS${NC}" | tee -a "$LOG_FILE"
    
    if [[ $ERRORS -eq 0 ]]; then
        echo -e "\n${GREEN}${BOLD}✓ 모든 검증을 통과했습니다!${NC}" | tee -a "$LOG_FILE"
        echo -e "${GREEN}실험을 안전하게 실행할 수 있습니다.${NC}" | tee -a "$LOG_FILE"
        
        if [[ $WARNINGS -gt 0 ]]; then
            echo -e "\n${YELLOW}경고 사항이 있지만 실행에는 문제가 없습니다.${NC}" | tee -a "$LOG_FILE"
        fi
        
        echo -e "\n${BLUE}실험 실행 명령어:${NC}" | tee -a "$LOG_FILE"
        echo -e "  ${BOLD}./scripts/experiments/run_auto_experiments.sh${NC}" | tee -a "$LOG_FILE"
        
        if [[ "$OS_TYPE" == "macos" ]]; then
            echo -e "\n${YELLOW}macOS 참고사항:${NC}" | tee -a "$LOG_FILE"
            echo -e "  - MPS를 사용할 수 없다면 CPU로 실행됩니다" | tee -a "$LOG_FILE"
            echo -e "  - 학습 속도가 느릴 수 있으므로 작은 배치 크기 권장" | tee -a "$LOG_FILE"
        fi
        
        return 0
    else
        echo -e "\n${RED}${BOLD}✗ 검증 실패!${NC}" | tee -a "$LOG_FILE"
        echo -e "${RED}위의 오류를 먼저 해결해주세요.${NC}" | tee -a "$LOG_FILE"
        
        echo -e "\n${YELLOW}권장 조치사항:${NC}" | tee -a "$LOG_FILE"
        echo -e "  1. pip install -r requirements.txt" | tee -a "$LOG_FILE"
        echo -e "  2. 누락된 파일/디렉토리 생성" | tee -a "$LOG_FILE"
        echo -e "  3. 설정 파일 오류 수정" | tee -a "$LOG_FILE"
        
        return 1
    fi
}

# 메인 실행
main() {
    echo -e "${BOLD}${BLUE}NLP 대화 요약 실험 환경 검증 (크로스 플랫폼)${NC}"
    echo -e "시작 시간: $(date)"
    echo -e "실행 환경: $OS_TYPE"
    echo -e "로그 파일: $LOG_FILE\n"
    
    # 각 검증 단계 실행
    check_system_info
    check_python_env
    check_project_structure
    check_python_packages
    check_data_files
    check_config_files
    check_resources
    check_permissions
    quick_code_test
    run_prerun_test
    check_multi_model_support
    
    # 최종 요약
    print_summary
    exit_code=$?
    
    echo -e "\n종료 시간: $(date)" | tee -a "$LOG_FILE"
    echo -e "전체 로그는 $LOG_FILE 에서 확인하세요." | tee -a "$LOG_FILE"
    
    exit $exit_code
}

# 스크립트 실행
main
