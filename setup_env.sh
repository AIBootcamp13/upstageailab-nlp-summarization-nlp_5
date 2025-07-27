#!/bin/bash

# UV를 사용한 Python 환경 설정 스크립트

echo "=== NLP 요약 프로젝트 환경 설정 시작 ==="

# UV 설치 확인
if ! command -v uv &> /dev/null; then
    echo "UV가 설치되어 있지 않습니다. 설치를 진행합니다..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# 프로젝트 디렉토리로 이동
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "프로젝트 디렉토리: $PROJECT_DIR"

# 기존 가상환경이 있다면 삭제
if [ -d ".venv" ]; then
    echo "기존 가상환경을 삭제합니다..."
    rm -rf .venv
fi

# UV로 Python 3.11 가상환경 생성
echo "Python 3.11 가상환경을 생성합니다..."
uv venv --python 3.11

# 가상환경 활성화
echo "가상환경을 활성화합니다..."
source .venv/bin/activate

# 기본 의존성 설치
echo "기본 의존성을 설치합니다..."
uv pip install -r code/requirements.txt

# 플랫폼 감지 및 PyTorch 설치
echo "플랫폼을 감지하고 적절한 PyTorch를 설치합니다..."

# macOS 감지 (M1/M2)
if [[ "$OSTYPE" == "darwin"* ]]; then
    # Apple Silicon 확인
    if [[ $(uname -m) == "arm64" ]]; then
        echo "Apple Silicon (M1/M2) 감지됨. MPS 지원 PyTorch 설치..."
        uv pip install torch torchvision torchaudio
    else
        echo "Intel Mac 감지됨. CPU 버전 PyTorch 설치..."
        uv pip install torch torchvision torchaudio
    fi
# Linux 감지
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # CUDA 사용 가능 여부 확인
    if command -v nvidia-smi &> /dev/null; then
        echo "NVIDIA GPU 감지됨. CUDA 11.8 지원 PyTorch 설치..."
        uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        echo "GPU가 감지되지 않음. CPU 버전 PyTorch 설치..."
        uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
# Windows (Git Bash, WSL 등)
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    echo "Windows 감지됨. CUDA 11.8 지원 PyTorch 설치..."
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "알 수 없는 플랫폼: $OSTYPE"
    echo "기본 PyTorch 설치..."
    uv pip install torch torchvision torchaudio
fi

# 개발 도구 설치 (선택사항)
read -p "개발 도구(pytest, black, flake8 등)를 설치하시겠습니까? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "개발 도구를 설치합니다..."
    uv pip install -e ".[dev]"
fi

# Jupyter 커널 업데이트
echo "Jupyter 커널을 업데이트합니다..."
.venv/bin/python -m ipykernel install --user --name nlp-sum --display-name "NLP-Sum (UV)"

# 설치 확인
echo ""
echo "=== 설치 확인 ==="
echo "Python 버전:"
.venv/bin/python --version
echo ""
echo "PyTorch 버전:"
.venv/bin/python -c "import torch; print(f'PyTorch: {torch.__version__}')"
.venv/bin/python -c "import torch; print(f'CUDA 사용 가능: {torch.cuda.is_available()}')"
if [[ "$OSTYPE" == "darwin"* ]] && [[ $(uname -m) == "arm64" ]]; then
    .venv/bin/python -c "import torch; print(f'MPS 사용 가능: {torch.backends.mps.is_available()}')"
fi

echo ""
echo "=== 환경 설정 완료 ==="
echo "가상환경 활성화: source .venv/bin/activate"
echo "Jupyter 실행: jupyter notebook"
