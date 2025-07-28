#!/bin/bash

echo "=== CUDA 환경 수정 (대안 방법) ==="

# 1. 기존 설치 제거
echo "1. 기존 패키지 제거..."
uv pip uninstall bitsandbytes unsloth -y 2>/dev/null || true

# 2. 특정 버전으로 재설치
echo "2. 호환성 검증된 버전으로 설치..."

# PyTorch 1.13+ 호환 bitsandbytes
uv pip install bitsandbytes==0.41.3 --no-cache-dir

# CPU 백업용 (CUDA 실패시 사용)
uv pip install bitsandbytes-cpu --no-cache-dir

# unsloth 최신 안정 버전
uv pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" --no-deps

# 3. 필수 의존성 재설치
echo "3. 의존성 재설치..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --force-reinstall

# 4. 환경변수 강제 설정
echo "4. 환경변수 설정..."
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/conda/envs/venv/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH

# .bashrc에 추가
cat >> ~/.bashrc << 'EOF'
# CUDA 환경 설정
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/conda/envs/venv/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH
EOF

# 5. 테스트
echo "5. 최종 테스트..."
source ~/.bashrc

python -c "
packages = ['torch', 'transformers', 'datasets', 'gradio', 'wandb']
print('=== 기본 패키지 테스트 ===')
for pkg in packages:
    try:
        __import__(pkg)
        print(f'✅ {pkg}')
    except ImportError as e:
        print(f'❌ {pkg}: {e}')

print('\n=== GPU 테스트 ===')
import torch
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
    print(f'✅ GPU 메모리: {torch.cuda.get_device_properties(0).total_memory // (1024**3)}GB')
else:
    print('⚠️ CUDA 미사용 - CPU 모드')

print('\n=== bitsandbytes 테스트 ===')
try:
    import bitsandbytes as bnb
    print('✅ bitsandbytes 성공')
except Exception as e:
    print(f'❌ bitsandbytes 실패: {e}')
    try:
        import bitsandbytes_cpu as bnb
        print('✅ bitsandbytes-cpu 백업 사용')
    except:
        print('❌ bitsandbytes 완전 실패')

print('\n=== unsloth 테스트 ===')
try:
    from unsloth import FastLanguageModel
    print('✅ unsloth 성공')
except Exception as e:
    print(f'❌ unsloth 실패: {e}')
"

echo "=== 수정 완료 ==="
