#!/bin/bash

# 환경 복구 스크립트
# 마이그레이션 중 문제 발생 시 이전 환경으로 롤백

set -e

echo "🔄 환경 복구 스크립트 시작"
echo "========================"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 백업 디렉토리 선택
echo "📁 사용 가능한 백업 디렉토리:"
ls -la | grep "^d.*backup_" || {
    log_error "백업 디렉토리를 찾을 수 없습니다."
    echo "백업이 없으면 복구할 수 없습니다."
    exit 1
}

echo ""
read -p "복구할 백업 디렉토리 이름을 입력하세요 (예: backup_20241128_143000): " BACKUP_DIR

if [ ! -d "$BACKUP_DIR" ]; then
    log_error "지정된 백업 디렉토리가 존재하지 않습니다: $BACKUP_DIR"
    exit 1
fi

log_info "백업 디렉토리 확인: $BACKUP_DIR"

# 복구 확인
echo ""
log_warning "⚠️  복구를 진행하면 현재 환경이 완전히 삭제되고 백업 상태로 되돌아갑니다."
read -p "계속 진행하시겠습니까? (y/N): " CONFIRM

if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
    log_info "복구가 취소되었습니다."
    exit 0
fi

echo ""
log_info "복구를 시작합니다..."

# 1. 현재 가상환경 제거
if [ -d ".venv" ]; then
    log_info "현재 가상환경 제거 중..."
    rm -rf .venv
    log_success "가상환경 제거 완료"
fi

# 2. 백업된 설정 파일들 복구
log_info "설정 파일 복구 중..."

if [ -f "$BACKUP_DIR/requirements.txt" ]; then
    cp "$BACKUP_DIR/requirements.txt" .
    log_success "requirements.txt 복구 완료"
fi

if [ -f "$BACKUP_DIR/requirements.txt" ] && [ -d "code" ]; then
    cp "$BACKUP_DIR/requirements.txt" code/
    log_success "code/requirements.txt 복구 완료"
fi

if [ -f "$BACKUP_DIR/config.yaml" ]; then
    cp "$BACKUP_DIR/config.yaml" .
    log_success "config.yaml 복구 완료"
fi

# 3. 백업된 가상환경 복구
if [ -f "$BACKUP_DIR/venv_backup.tar.gz" ]; then
    log_info "백업된 가상환경 복구 중... (시간이 걸릴 수 있습니다)"
    tar -xzf "$BACKUP_DIR/venv_backup.tar.gz"
    log_success "가상환경 복구 완료"
else
    log_warning "백업된 가상환경이 없습니다. 새로 생성합니다."
    
    # Python 버전 확인
    if [ -f ".python-version" ]; then
        PYTHON_VERSION=$(cat .python-version)
    else
        PYTHON_VERSION="3.11"
    fi
    
    log_info "Python $PYTHON_VERSION 가상환경 생성 중..."
    python3 -m venv .venv
    
    # requirements.txt가 있으면 설치
    if [ -f "requirements.txt" ]; then
        log_info "requirements.txt에서 패키지 설치 중..."
        .venv/bin/pip install -r requirements.txt
        log_success "패키지 설치 완료"
    fi
fi

# 4. 환경 검증
log_info "복구된 환경 검증 중..."

if [ -d ".venv" ]; then
    log_success "가상환경 복구 확인"
    
    # Python 및 주요 패키지 확인
    .venv/bin/python -c "
import sys
print(f'Python version: {sys.version}')

# 주요 패키지 확인
packages = ['torch', 'transformers', 'pytorch_lightning']
for pkg in packages:
    try:
        module = __import__(pkg)
        version = getattr(module, '__version__', 'unknown')
        print(f'{pkg}: {version}')
    except ImportError:
        print(f'{pkg}: 설치되지 않음')
" || log_warning "일부 패키지에 문제가 있을 수 있습니다."
else
    log_error "가상환경 복구 실패"
    exit 1
fi

# 5. 마이그레이션 관련 파일 정리
log_info "마이그레이션 관련 임시 파일 정리 중..."

# .env.local 제거 (마이그레이션에서 생성된 파일)
[ -f ".env.local" ] && rm .env.local && log_info ".env.local 제거"

# 백업된 설정 파일들 제거
[ -f "config.yaml.backup" ] && rm config.yaml.backup && log_info "config.yaml.backup 제거"
[ -f "requirements.txt.backup" ] && rm requirements.txt.backup && log_info "requirements.txt.backup 제거"

log_success "정리 완료"

# 6. 복구 완료 메시지
echo ""
echo "🎉 환경 복구가 성공적으로 완료되었습니다!"
echo ""
echo "📋 복구된 내용:"
echo "• 설정 파일: 백업 시점으로 복구"
echo "• 가상환경: 이전 상태로 복구"
echo "• 패키지: 백업 시점의 버전으로 복구"
echo ""
echo "🔍 환경 확인:"
echo "• ./check_env.sh 로 환경 상태 확인"
echo "• 기존 학습 스크립트가 정상 작동하는지 테스트"
echo ""
echo "📁 백업 보관:"
echo "• 백업 디렉토리는 보관됩니다: $BACKUP_DIR"
echo "• 필요시 언제든 다시 복구 가능"
echo ""

log_success "복구 완료! 이전 환경으로 성공적으로 되돌렸습니다. 🔄"
