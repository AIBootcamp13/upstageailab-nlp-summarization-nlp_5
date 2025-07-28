#!/bin/bash

echo "=== CUDA 버전 맞춤 수정 ==="

# 1. 현재 상황 확인
echo "1. 현재 PyTorch CUDA 버전 확인..."
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"

# 2. CUDA 라이브러리 찾기
echo "2. CUDA 라이브러리 검색..."
find /opt/conda -name "libcudart.so*" 2>/dev/null | head -5
find /usr/local -name "libcudart.so*" 2>/dev/null | head -5

# 3. bitsandbytes 호환 버전으로 재설치
echo "3. bitsandbytes 호환 버전 설치..."
uv pip uninstall bitsandbytes -q
# CUDA 12.6 호환 또는 강제로 CUDA 버전 지정
CUDA_VERSION=126 uv pip install bitsandbytes --no-cache-dir

# 4. 환경 변수 강제 설정
echo "4. 환경 변수 설정..."
export CUDA_HOME=/opt/conda
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
export BNB_CUDA_VERSION=126

# .bashrc에 추가
cat >> ~/.bashrc << 'EOF'
# bitsandbytes CUDA 설정
export CUDA_HOME=/opt/conda
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
export BNB_CUDA_VERSION=126
EOF

# 5. 대안: PyTorch 다운그레이드로 호환성 확보
echo "5. 대안: PyTorch CUDA 11.8로 다운그레이드..."
read -p "PyTorch를 CUDA 11.8로 다운그레이드하시겠습니까? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    uv pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118 --force-reinstall
    uv pip install bitsandbytes==0.41.1 --force-reinstall
fi

# 6. 최종 테스트
echo "6. 최종 테스트..."
source ~/.bashrc

python -c "
print('=== 환경 정보 ===')
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'CUDA 사용가능: {torch.cuda.is_available()}')

import os
print(f'CUDA_HOME: {os.environ.get(\"CUDA_HOME\", \"없음\")}')
print(f'BNB_CUDA_VERSION: {os.environ.get(\"BNB_CUDA_VERSION\", \"없음\")}')

print('\n=== bitsandbytes 테스트 ===')
try:
    import bitsandbytes as bnb
    print('✅ bitsandbytes 성공')
    
    # GPU 테스트
    x = bnb.nn.Int8Params(torch.randn(10, 10), requires_grad=False).cuda()
    print('✅ bitsandbytes GPU 연산 성공')
except Exception as e:
    print(f'❌ bitsandbytes 실패: {e}')

print('\n=== unsloth 테스트 ===')
try:
    from unsloth import FastLanguageModel
    print('✅ unsloth 성공')
except Exception as e:
    print(f'❌ unsloth 실패: {e}')
"

echo "=== 수정 완료 ==="
