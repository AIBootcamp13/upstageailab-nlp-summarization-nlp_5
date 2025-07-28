#!/bin/bash

echo "=== 작동하는 환경 설치 스크립트 ==="
echo "현재 성공적으로 작동하는 패키지 버전들로 새로운 환경 설치"

# 1. 환경 활성화
eval "$(/opt/conda/bin/conda shell.bash hook)"
conda activate venv

# 2. 핵심 패키지 설치 (uv 사용)
echo "2. 핵심 ML 라이브러리 설치..."

# PyTorch 계열
uv pip install torch==2.7.1 torchaudio==2.5.1
uv pip install torchvision==0.22.1+cu126 --index-url https://download.pytorch.org/whl/cu126

# Transformers 계열
uv pip install transformers==4.54.0 datasets==3.6.0 tokenizers==0.21.4

# 최적화 라이브러리
uv pip install bitsandbytes==0.46.1 accelerate==1.9.0 peft==0.16.0

# Unsloth (가장 중요)
uv pip install unsloth==2025.7.8

# 3. 추가 라이브러리
echo "3. 추가 라이브러리 설치..."
uv pip install gradio==5.38.2 wandb==0.21.0 xformers==0.0.31.post1 trl==0.19.1

# 4. 유틸리티
echo "4. 유틸리티 설치..."
uv pip install pandas==2.3.1 numpy==2.1.2 tqdm==4.67.1 requests==2.32.4

# 5. 설치 확인
echo "5. 설치 확인..."
python -c "
print('=== 설치 확인 ===')

# 핵심 패키지 확인
packages = {
    'torch': '2.7.1',
    'torchvision': '0.22.1+cu126', 
    'transformers': '4.54.0',
    'bitsandbytes': '0.46.1',
    'unsloth': '2025.7.8'
}

for pkg, expected in packages.items():
    try:
        module = __import__(pkg)
        version = getattr(module, '__version__', 'N/A')
        status = '✅' if expected in version else '⚠️'
        print(f'{status} {pkg}: {version} (예상: {expected})')
    except ImportError as e:
        print(f'❌ {pkg}: {e}')

# GPU 테스트
import torch
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
else:
    print('❌ GPU 사용 불가')

# Unsloth 최종 테스트
try:
    from unsloth import FastLanguageModel
    print('🎉 unsloth 완벽 작동!')
except Exception as e:
    print(f'❌ unsloth 오류: {e}')
"

echo ""
echo "=== 설치 완료 ==="
echo "requirements_working.txt에 작동하는 버전들이 저장되었습니다."
