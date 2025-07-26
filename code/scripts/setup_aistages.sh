#!/bin/bash

# AIStages 환경 자동 설정 스크립트
# 사용법: bash scripts/setup_aistages.sh

echo "========================================"
echo "AIStages 환경 설정 시작"
echo "========================================"

# 1. Base 환경 정리
echo "1. Base 가상환경 초기화 중..."
conda activate base

# UV가 설치되어 있는지 확인
if ! command -v uv &> /dev/null; then
    echo "UV가 설치되어 있지 않습니다. 먼저 UV를 설치합니다..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="/data/ephemeral/home/.local/bin:$PATH"
fi

# 기존 패키지 제거
echo "기존 패키지 제거 중..."
uv pip freeze | xargs -n 1 uv pip uninstall -y 2>/dev/null || true

# 필수 패키지 설치
echo "필수 패키지 설치 중..."
uv pip install -U pip setuptools wheel --system

# 2. 시스템 패키지 설치
echo "2. 시스템 패키지 설치 중..."

# Git 설치 확인
if ! command -v git &> /dev/null; then
    echo "Git 설치 중..."
    apt update
    apt install -y git
fi

# 한국어 폰트 설치
echo "한국어 폰트 설치 중..."
apt-get install -y fonts-nanum* >/dev/null 2>&1

# OpenCV 의존성
echo "OpenCV 의존성 설치 중..."
apt-get install -y libgl1-mesa-glx libglib2.0-0 >/dev/null 2>&1

# 3. Git 설정
echo "3. Git 설정 중..."
echo "Git 사용자 이름을 입력하세요:"
read git_username
echo "Git 이메일을 입력하세요:"
read git_email

git config --global user.name "$git_username"
git config --global user.email "$git_email"
git config --global credential.helper store
git config --global core.pager "cat"
git config --global core.editor "vim"

# 4. 프로젝트 requirements 설치
echo "4. 프로젝트 의존성 설치 중..."
if [ -f "requirements.txt" ]; then
    echo "requirements.txt 설치 중..."
    uv pip install -r requirements.txt --system
else
    echo "requirements.txt 파일을 찾을 수 없습니다."
fi

if [ -f "pyproject.toml" ]; then
    echo "pyproject.toml 설치 중..."
    uv pip install -r pyproject.toml --system
fi

# 5. 환경 변수 설정
echo "5. 환경 변수 설정 중..."
if ! grep -q "/data/ephemeral/home/.local/bin" ~/.bashrc; then
    echo 'export PATH="/data/ephemeral/home/.local/bin:$PATH"' >> ~/.bashrc
fi

# 6. 완료 메시지
echo "========================================"
echo "환경 설정 완료!"
echo "========================================"
echo ""
echo "다음 명령어로 설정을 확인하세요:"
echo "  uv --version"
echo "  git config --list"
echo "  python -c 'import torch; print(torch.__version__)'"
echo ""
echo "환경 변수 적용을 위해 다음 명령어를 실행하세요:"
echo "  source ~/.bashrc"
