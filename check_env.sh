#!/bin/bash

# UV 환경 정보 출력 스크립트

echo "=== NLP 요약 프로젝트 UV 환경 정보 ==="
echo ""

# UV 버전
echo "UV 버전:"
uv --version
echo ""

# Python 버전
echo "Python 버전:"
.venv/bin/python --version
echo ""

# 가상환경 위치
echo "가상환경 위치:"
echo "$(pwd)/.venv"
echo ""

# 설치된 주요 패키지
echo "주요 패키지 버전:"
.venv/bin/python -c "
import importlib
packages = {
    'torch': 'PyTorch',
    'transformers': 'Transformers',
    'pytorch_lightning': 'PyTorch Lightning',
    'pandas': 'Pandas',
    'numpy': 'NumPy',
    'wandb': 'Weights & Biases',
    'rouge': 'ROUGE',
    'accelerate': 'Accelerate',
    'datasets': 'Datasets',
    'peft': 'PEFT',
    'bitsandbytes': 'BitsAndBytes',
    'unsloth': 'Unsloth (High-Performance Fine-tuning)',
}

for pkg, name in packages.items():
    try:
        module = importlib.import_module(pkg)
        version = getattr(module, '__version__', 'unknown')
        print(f'{name}: {version}')
    except ImportError:
        print(f'{name}: 설치되지 않음')
"
echo ""

# 디바이스 정보
echo "디바이스 정보:"
.venv/bin/python -c "
import torch
import platform

print(f'운영체제: {platform.system()} {platform.release()}')
print(f'프로세서: {platform.processor()}')
print(f'CUDA 사용 가능: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA 버전: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# Mac M1/M2 확인
if platform.system() == 'Darwin' and platform.processor() == 'arm':
    print(f'MPS 사용 가능: {torch.backends.mps.is_available()}')
"
echo ""

# 환경 변수 확인
echo "환경 변수 설정:"
if [ -f .env ]; then
    echo ".env 파일이 존재합니다."
    echo "주요 환경 변수:"
    grep -E "^(WANDB_PROJECT|CUDA_VISIBLE_DEVICES|USE_UNSLOTH|USE_QLORA)=" .env 2>/dev/null || echo "환경 변수가 설정되지 않았습니다."
else
    echo ".env 파일이 없습니다. .env.template를 복사하여 .env를 생성하세요."
fi
echo ""

# QLoRA/unsloth 지원 확인
echo "고급 기능 지원:"
.venv/bin/python -c "
import importlib

# QLoRA 지원 확인
try:
    importlib.import_module('peft')
    importlib.import_module('bitsandbytes')
    print('QLoRA 지원: ✅ (peft + bitsandbytes)')
except ImportError:
    print('QLoRA 지원: ❌ (peft 또는 bitsandbytes 없음)')

# unsloth 지원 확인
try:
    importlib.import_module('unsloth')
    print('unsloth 지원: ✅ (고성능 파인튜닝 가능)')
except ImportError:
    print('unsloth 지원: ❌ (Ubuntu/Linux 환경에서 설치 권장)')

# Gradient Checkpointing 지원
try:
    import torch
    if hasattr(torch.utils.checkpoint, 'checkpoint'):
        print('Gradient Checkpointing: ✅')
    else:
        print('Gradient Checkpointing: ❌')
except:
    print('Gradient Checkpointing: ❌')
"
echo ""

# Jupyter 커널 정보
echo "Jupyter 커널:"
.venv/bin/python -m jupyter kernelspec list 2>/dev/null | grep nlp || echo "Jupyter 커널이 설치되지 않았습니다."
