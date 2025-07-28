#!/bin/bash

echo "=== 간단한 CUDA 호환성 수정 ==="

# PyTorch와 bitsandbytes 호환 버전으로 일괄 설치
echo "1. 호환 버전 일괄 설치..."

# 모든 관련 패키지 제거
uv pip uninstall torch torchvision torchaudio bitsandbytes unsloth -q || true

# 검증된 호환 조합으로 설치
echo "PyTorch 2.0.1 + CUDA 11.8 + bitsandbytes 0.41.1 조합 설치..."
uv pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118
uv pip install bitsandbytes==0.41.1
uv pip install unsloth

# 환경 변수 설정
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
echo 'export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH' >> ~/.bashrc

echo "2. 테스트..."
python -c "
import torch
print(f'PyTorch: {torch.__version__} (CUDA: {torch.version.cuda})')
print(f'GPU: {torch.cuda.is_available()}')

try:
    import bitsandbytes as bnb
    print('✅ bitsandbytes 성공')
except Exception as e:
    print(f'❌ bitsandbytes: {e}')

try:
    from unsloth import FastLanguageModel
    print('✅ unsloth 성공')
except Exception as e:
    print(f'❌ unsloth: {e}')
"

echo "=== 완료 ==="
