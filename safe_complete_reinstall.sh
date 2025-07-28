#!/bin/bash

echo "=== 안전한 전체 재설치 ==="

# 1. Python 프로세스 완전 종료 및 캐시 정리
echo "1. 환경 완전 정리..."
pkill -f python || true
sleep 2

# Python 캐시 완전 삭제
rm -rf /opt/conda/envs/venv/lib/python3.11/site-packages/torch*
rm -rf /opt/conda/envs/venv/lib/python3.11/site-packages/transformers*
rm -rf /opt/conda/envs/venv/lib/python3.11/site-packages/unsloth*
rm -rf /opt/conda/envs/venv/lib/python3.11/site-packages/bitsandbytes*
find /opt/conda/envs/venv -name "*.pyc" -delete 2>/dev/null || true
find /opt/conda/envs/venv -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# 2. uv 자체도 리셋
echo "2. uv 캐시 정리..."
rm -rf ~/.cache/uv 2>/dev/null || true

# 3. 검증된 순서대로 하나씩 설치
echo "3. 단계별 설치..."

# 3-1. 기본 패키지
echo "  3-1. 기본 패키지 설치..."
uv pip install packaging wheel setuptools

# 3-2. NumPy 먼저
echo "  3-2. NumPy 설치..."
uv pip install "numpy<2.0"

# 3-3. PyTorch (CUDA 11.8)
echo "  3-3. PyTorch CUDA 11.8 설치..."
uv pip install torch==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118
uv pip install torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118  
uv pip install torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# 3-4. PyTorch 설치 확인
echo "  3-4. PyTorch 확인..."
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" || {
    echo "❌ PyTorch 설치 실패. 다시 시도..."
    sleep 5
    uv pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118 --force-reinstall
}

# 3-5. bitsandbytes
echo "  3-5. bitsandbytes 설치..."
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
uv pip install bitsandbytes==0.41.1

# 3-6. 기타 필수 패키지
echo "  3-6. 기타 패키지 설치..."
uv pip install transformers==4.36.0
uv pip install datasets==2.14.0
uv pip install accelerate==0.25.0

# 3-7. unsloth (마지막)
echo "  3-7. unsloth 설치..."
uv pip install unsloth==2024.8

# 4. 환경 변수 설정
echo "4. 환경 변수 설정..."
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
echo 'export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH' >> ~/.bashrc

# 5. 최종 검증
echo "5. 최종 검증..."
python -c "
print('=== 설치 검증 ===')

# torch 먼저
try:
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
    print(f'   CUDA: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'   GPU: {torch.cuda.get_device_name(0)}')
except Exception as e:
    print(f'❌ PyTorch: {e}')
    exit(1)

# transformers
try:
    import transformers
    print(f'✅ Transformers: {transformers.__version__}')
except Exception as e:
    print(f'❌ Transformers: {e}')

# bitsandbytes  
try:
    import bitsandbytes as bnb
    print(f'✅ bitsandbytes: GPU={torch.cuda.is_available()}')
except Exception as e:
    print(f'❌ bitsandbytes: {e}')

# unsloth (가장 마지막)
try:
    from unsloth import FastLanguageModel
    print('✅ unsloth: FastLanguageModel 임포트 성공')
    print('\n🎉 모든 패키지 정상 작동!')
except Exception as e:
    print(f'❌ unsloth: {e}')
"

echo "=== 재설치 완료 ==="
echo "성공했다면 이제 'python -c \"from unsloth import FastLanguageModel; print('✅ Ready')\"' 로 확인하세요"
