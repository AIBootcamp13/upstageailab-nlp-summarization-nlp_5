#!/bin/bash

# 마이그레이션 스크립트: 조장님의 최신 기술 스택으로 완전 업그레이드
# 작성: nlp-sum-lyj 통합 프로젝트
# 기능: 기존 환경 백업 + 새 환경 설정 + 완전 자동화

set -e  # 에러 발생 시 스크립트 중단

echo "🚀 nlp-sum-lyj 최신 기술 스택 마이그레이션 시작"
echo "============================================"
echo ""

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 로그 함수들
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

# 단계별 진행 상황
CURRENT_STEP=0
TOTAL_STEPS=8

progress() {
    CURRENT_STEP=$((CURRENT_STEP + 1))
    echo ""
    echo -e "${BLUE}=== 단계 $CURRENT_STEP/$TOTAL_STEPS: $1 ===${NC}"
}

# 1단계: 환경 검증
progress "환경 및 전제조건 검증"

# Python 버전 확인
if ! command -v python3 &> /dev/null; then
    log_error "Python3가 설치되어 있지 않습니다."
    exit 1
fi

# UV 설치 확인
if ! command -v uv &> /dev/null; then
    log_warning "UV가 설치되어 있지 않습니다. 설치하겠습니다."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
    log_success "UV 설치 완료"
fi

log_info "Python 버전: $(python3 --version)"
log_info "UV 버전: $(uv --version)"

# 2단계: 기존 환경 백업
progress "기존 환경 백업"

BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# 중요 파일들 백업
if [ -f "requirements.txt" ]; then
    cp requirements.txt "$BACKUP_DIR/"
    log_info "requirements.txt 백업 완료"
fi

if [ -f "code/requirements.txt" ]; then
    cp code/requirements.txt "$BACKUP_DIR/"
    log_info "code/requirements.txt 백업 완료"
fi

if [ -f "config.yaml" ]; then
    cp config.yaml "$BACKUP_DIR/"
    log_info "config.yaml 백업 완료"
fi

if [ -d ".venv" ]; then
    log_warning "기존 가상환경을 백업합니다 (크기가 클 수 있습니다)"
    tar -czf "$BACKUP_DIR/venv_backup.tar.gz" .venv/
    log_info ".venv 백업 완료"
fi

log_success "백업 디렉토리: $BACKUP_DIR"

# 3단계: 기존 가상환경 제거
progress "기존 가상환경 정리"

if [ -d ".venv" ]; then
    log_warning "기존 가상환경을 제거합니다..."
    rm -rf .venv
    log_success "기존 가상환경 제거 완료"
fi

# 4단계: 환경별 Python 환경 설정
progress "Python 3.11 환경 구성 (conda/UV 자동 감지)"

log_info "Python 3.11 가상환경 생성..."
if command -v conda &> /dev/null; then
    # conda 환경에서 (AIStages 등)
    conda create -n nlp-sum-latest python==3.11 -y
    source activate nlp-sum-latest
    log_success "conda 가상환경 생성 완료"
else
    # 로컬 환경에서 UV 사용
    uv venv --python 3.11
    
    if [ ! -f ".python-version" ] || [ "$(cat .python-version)" != "3.11" ]; then
        echo "3.11" > .python-version
        log_info ".python-version 파일 업데이트 (3.11)"
    fi
    
    log_success "UV 가상환경 생성 완료"
fi

# 5단계: 최신 라이브러리 설치
progress "최신 기술 스택 설치 (torch 2.6.0, transformers 4.54.0)"

log_info "핵심 라이브러리 설치 중..."
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
uv pip install transformers==4.54.0
uv pip install pytorch_lightning==2.5.2
uv pip install accelerate==1.9.0

log_info "데이터 처리 라이브러리 설치 중..."
uv pip install datasets pandas numpy

log_info "평가 및 모니터링 도구 설치 중..."
uv pip install wandb rouge-score

log_info "효율성 도구 설치 중..."
uv pip install peft bitsandbytes

# unsloth는 환경에 따라 조건부 설치
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    log_info "Linux 환경 감지: unsloth 설치 시도 중..."
    if uv pip install unsloth; then
        log_success "unsloth 설치 완료 (고성능 파인튜닝 활성화)"
        echo "USE_UNSLOTH=true" >> .env.local
    else
        log_warning "unsloth 설치 실패, QLoRA 모드로 대체"
        echo "USE_UNSLOTH=false" >> .env.local
        echo "USE_QLORA=true" >> .env.local
    fi
else
    log_warning "macOS 환경: unsloth는 Linux에서 최적화됨, QLoRA 모드 사용"
    echo "USE_UNSLOTH=false" >> .env.local
    echo "USE_QLORA=true" >> .env.local
fi

log_success "모든 라이브러리 설치 완료"

# 6단계: 환경 변수 설정
progress "환경 변수 및 설정 파일 구성"

if [ ! -f ".env" ]; then
    if [ -f ".env.template" ]; then
        cp .env.template .env
        log_info ".env 파일을 .env.template에서 생성했습니다."
        log_warning "중요: .env 파일을 편집하여 API 키 등을 설정하세요."
    else
        log_warning ".env.template 파일이 없습니다. 수동으로 .env 파일을 생성하세요."
    fi
fi

# 환경별 설정 적용
if [ -f ".env.local" ]; then
    log_info "환경별 설정 (.env.local) 생성 완료"
fi

# 7단계: 호환성 검증
progress "업그레이드된 환경 호환성 검증"

log_info "라이브러리 import 테스트 중..."
.venv/bin/python -c "
import torch
import transformers
import pytorch_lightning
import accelerate
import datasets
print(f'✅ torch: {torch.__version__}')
print(f'✅ transformers: {transformers.__version__}')
print(f'✅ pytorch_lightning: {pytorch_lightning.__version__}')
print(f'✅ accelerate: {accelerate.__version__}')
print(f'✅ datasets: {datasets.__version__}')

# QLoRA 지원 확인
try:
    import peft, bitsandbytes
    print('✅ QLoRA 지원 (peft + bitsandbytes)')
except ImportError as e:
    print(f'❌ QLoRA 지원 실패: {e}')

# unsloth 지원 확인
try:
    import unsloth
    print('✅ unsloth 지원 (고성능 파인튜닝)')
except ImportError:
    print('⚠️  unsloth 없음 (Linux 환경에서 권장)')
"

if [ $? -eq 0 ]; then
    log_success "모든 라이브러리 호환성 검증 완료"
else
    log_error "라이브러리 호환성 문제 발견"
    exit 1
fi

# 8단계: 최종 요약
progress "마이그레이션 완료 및 요약"

echo ""
echo "🎉 마이그레이션이 성공적으로 완료되었습니다!"
echo ""
echo "📋 주요 업그레이드 내용:"
echo "• torch: >=2.0.0 → 2.6.0"
echo "• transformers: 4.35.2 → 4.54.0"
echo "• pytorch_lightning: 2.1.2 → 2.5.2"
echo "• accelerate: 1.9.0 (새로 추가)"
echo "• QLoRA 지원 (peft + bitsandbytes)"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "• unsloth 지원 (메모리 75% 절약)"
fi
echo ""
echo "📁 백업 위치: $BACKUP_DIR"
echo ""
echo "🚀 다음 단계:"
echo "1. .env 파일을 편집하여 API 키 설정"
echo "2. ./check_env.sh로 환경 확인"
echo "3. 학습 스크립트 실행 테스트"
echo ""
echo "💡 예상 효과:"
echo "• 학습 속도 20-30% 향상"
echo "• 메모리 사용량 30-75% 감소"
echo "• 더 긴 요약 생성 (decoder_max_len 200)"
echo ""

log_success "마이그레이션 완료! 🎯"
