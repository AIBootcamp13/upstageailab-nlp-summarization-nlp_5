#!/bin/bash

echo "=== 새로운 conda 환경 생성 ==="

# 1. 기존 환경 비활성화
echo "1. 기존 환경에서 나가기..."
conda deactivate 2>/dev/null || true

# 2. 새 환경 생성
echo "2. 새로운 환경 생성..."
conda create -n venv_new python=3.11 -y

# 3. 새 환경 활성화
echo "3. 새 환경 활성화..."
conda activate venv_new

# 4. uv 설치
echo "4. uv 설치..."
pip install uv

# 5. 검증된 패키지 조합 설치
echo "5. 패키지 설치..."
uv pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118
uv pip install transformers==4.36.0 datasets==2.14.0 accelerate==0.25.0
uv pip install bitsandbytes==0.41.1
uv pip install unsloth==2024.8
uv pip install gradio wandb

# 6. 환경 변수 설정
echo "6. 환경 변수 설정..."
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH

# 7. 테스트
echo "7. 최종 테스트..."
python -c "
print('=== 새 환경 테스트 ===')

import torch
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')

import transformers
print(f'✅ Transformers: {transformers.__version__}')

import bitsandbytes as bnb
print(f'✅ bitsandbytes')

from unsloth import FastLanguageModel
print('✅ unsloth: FastLanguageModel 임포트 성공')

print('\n🎉 새 환경 설정 완료!')
"

echo "=== 환경 생성 완료 ==="
echo ""
echo "앞으로 사용법:"
echo "conda activate venv_new"
echo ""
echo "기존 환경 제거 (선택사항):"
echo "conda env remove -n venv"
