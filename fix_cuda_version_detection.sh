#!/bin/bash

echo "=== CUDA 버전 감지 오류 수정 ==="

# 1. 환경 활성화
eval "$(/opt/conda/bin/conda shell.bash hook)"
conda activate venv

# 2. 현재 CUDA 버전 확인
echo "2. CUDA 버전 확인..."
echo "=== conda list | grep cuda ==="
conda list | grep cuda

echo -e "\n=== nvcc 버전 ==="
nvcc --version 2>/dev/null || echo "nvcc 없음"

echo -e "\n=== PyTorch CUDA 버전 ==="
python -c "import torch; print(f'PyTorch CUDA: {torch.version.cuda}')"

# 3. 잘못된 bitsandbytes 제거
echo -e "\n3. 잘못된 bitsandbytes 제거..."
pip uninstall bitsandbytes -y
conda remove bitsandbytes -y 2>/dev/null || true

# 4. CUDA 환경 변수 강제 설정
echo "4. CUDA 환경 변수 강제 설정..."
export CUDA_VERSION=121  # 12.1
export BNB_CUDA_VERSION=121
export CUDA_HOME=/opt/conda
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH

# 5. 올바른 CUDA 버전으로 bitsandbytes 설치
echo "5. 올바른 CUDA 버전으로 bitsandbytes 설치..."

# 5-1. 먼저 CUDA 12.1 버전 시도
echo "  5-1. CUDA 12.1 버전 bitsandbytes 시도..."
CUDA_VERSION=121 pip install bitsandbytes --no-cache-dir --force-reinstall

# 5-2. 설치 확인
echo "6. 설치 확인..."
python -c "
import os
os.environ['CUDA_VERSION'] = '121'
os.environ['BNB_CUDA_VERSION'] = '121'

try:
    import bitsandbytes as bnb
    print('✅ bitsandbytes 성공')
except Exception as e:
    print(f'❌ 첫 번째 시도 실패: {e}')
    print('CPU 버전으로 재시도...')
    exit(1)
" || {
    echo "7. CPU 버전으로 백업 설치..."
    pip install bitsandbytes-cpu --force-reinstall
}

# 8. unsloth 재설치
echo "8. unsloth 재설치..."
pip uninstall unsloth -y
pip install unsloth --no-cache-dir

# 9. 환경 변수를 .bashrc에 영구 저장
echo "9. 환경 변수 영구 저장..."
cat >> ~/.bashrc << 'EOF'

# CUDA 버전 강제 설정 (bitsandbytes용)
export CUDA_VERSION=121
export BNB_CUDA_VERSION=121
export CUDA_HOME=/opt/conda
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
EOF

# 10. 최종 테스트
echo "10. 최종 테스트..."
source ~/.bashrc

python -c "
import os
print('=== 환경 변수 확인 ===')
print(f'CUDA_VERSION: {os.environ.get(\"CUDA_VERSION\", \"없음\")}')
print(f'BNB_CUDA_VERSION: {os.environ.get(\"BNB_CUDA_VERSION\", \"없음\")}')

print('\n=== PyTorch 테스트 ===')
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA 사용가능: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

print('\n=== bitsandbytes 테스트 ===')
try:
    import bitsandbytes as bnb
    print('✅ bitsandbytes 임포트 성공')
    
    # 간단한 양자화 테스트
    import torch
    if torch.cuda.is_available():
        x = torch.randn(10, 10).cuda()
        linear = bnb.nn.Linear8bitLt(10, 10).cuda()
        print('✅ bitsandbytes GPU 연산 성공')
    else:
        print('⚠️ CPU 모드에서 bitsandbytes 사용')
        
except Exception as e:
    print(f'❌ bitsandbytes 오류: {e}')

print('\n=== unsloth 테스트 ===')
try:
    from unsloth import FastLanguageModel
    print('✅ unsloth 임포트 성공')
    print('\n🎉 모든 패키지 정상 작동!')
except Exception as e:
    print(f'❌ unsloth 오류: {e}')
    print('하지만 bitsandbytes가 작동하면 unsloth도 곧 작동할 것입니다.')
"

echo ""
echo "=== 수정 완료 ==="
echo ""
echo "다음 세션부터는 conda activate venv만 하면 됩니다."
echo ""
echo "최종 확인:"
echo "python -c \"from unsloth import FastLanguageModel; print('✅ 완료')\""
