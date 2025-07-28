#!/bin/bash

echo "=== 환경 완전 리셋 및 재구축 ==="

# 1. Python 환경 완전 정리
echo "1. Python 환경 정리..."
find /opt/conda/envs/venv -name "*.pyc" -delete 2>/dev/null || true
find /opt/conda/envs/venv -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
rm -rf /opt/conda/envs/venv/lib/python3.11/site-packages/torch* 2>/dev/null || true

# 2. 모든 패키지 제거
echo "2. 관련 패키지 완전 제거..."
uv pip uninstall torch torchvision torchaudio transformers datasets gradio wandb unsloth bitsandbytes || true

# 3. UV 캐시 정리 (가능한 명령어 사용)
echo "3. 캐시 정리..."
rm -rf ~/.cache/uv 2>/dev/null || true

# 4. 기본 패키지부터 차례대로 설치
echo "4. 기본 패키지 설치..."
uv pip install numpy packaging

# 5. PyTorch CUDA 버전 설치
echo "5. PyTorch CUDA 설치..."
uv pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# 6. 다른 패키지들 설치
echo "6. 기타 패키지 설치..."
uv pip install transformers==4.36.0 datasets==2.14.0 gradio==4.7.1 wandb

# 7. bitsandbytes 안정 버전 설치
echo "7. bitsandbytes 설치..."
uv pip install bitsandbytes==0.41.1

# 8. unsloth 설치
echo "8. unsloth 설치..."
uv pip install unsloth

# 9. Python 재시작을 위한 환경 확인
echo "9. 설치 확인..."
python -c "
import sys
print('Python 경로:', sys.executable)

print('\n=== 패키지 확인 ===')
try:
    import torch
    print(f'✅ torch: {torch.__version__}')
    print(f'   CUDA 사용가능: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'   GPU: {torch.cuda.get_device_name(0)}')
except Exception as e:
    print(f'❌ torch: {e}')

try:
    import transformers
    print(f'✅ transformers: {transformers.__version__}')
except Exception as e:
    print(f'❌ transformers: {e}')

try:
    import bitsandbytes
    print(f'✅ bitsandbytes')
except Exception as e:
    print(f'❌ bitsandbytes: {e}')

try:
    import unsloth
    print(f'✅ unsloth')
    from unsloth import FastLanguageModel
    print('✅ FastLanguageModel 임포트 성공')
except Exception as e:
    print(f'❌ unsloth: {e}')
"

echo "=== 환경 리셋 완료 ==="
echo "문제가 계속되면 conda 환경을 새로 만드는 것을 고려하세요:"
echo "conda create -n new_env python=3.11"
echo "conda activate new_env"
