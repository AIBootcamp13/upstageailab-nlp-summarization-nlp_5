#!/bin/bash
# 원격 서버 환경 자동 설정 스크립트

set -e  # 오류 시 중단

echo "🌟 원격 서버 환경 설정 시작"
echo "==============================="

# 1. 시스템 정보 확인
echo "📊 시스템 정보 확인"
echo "OS: $(uname -s)"
echo "Architecture: $(uname -m)"
echo "Hostname: $(hostname)"

# 2. GPU 확인
echo ""
echo "🔍 GPU 상태 확인"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
    GPU_COUNT=$(nvidia-smi -L | wc -l)
    echo "감지된 GPU 수: $GPU_COUNT"
else
    echo "⚠️ NVIDIA GPU를 찾을 수 없습니다. CPU 모드로 설정됩니다."
    GPU_COUNT=0
fi

# 3. Python 환경 설정
echo ""
echo "🐍 Python 환경 설정"

# Python 버전 확인
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "Python 버전: $PYTHON_VERSION"

if [[ "$PYTHON_VERSION" < "3.8" ]]; then
    echo "❌ Python 3.8 이상이 필요합니다. 현재: $PYTHON_VERSION"
    exit 1
fi

# 4. 가상환경 생성
echo ""
echo "📦 가상환경 설정"

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "✅ 가상환경 생성 완료"
else
    echo "✅ 기존 가상환경 발견"
fi

# 가상환경 활성화
source .venv/bin/activate

# 5. 패키지 설치
echo ""
echo "📚 의존성 설치"

# pip 업그레이드
pip install --upgrade pip

# GPU 환경에 따른 PyTorch 설치
if [ $GPU_COUNT -gt 0 ]; then
    echo "🚀 CUDA 환경용 PyTorch 설치"
    
    # CUDA 버전 확인
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d, -f1 | cut -dV -f2)
        echo "CUDA 버전: $CUDA_VERSION"
        
        if [[ "$CUDA_VERSION" == "12."* ]]; then
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl