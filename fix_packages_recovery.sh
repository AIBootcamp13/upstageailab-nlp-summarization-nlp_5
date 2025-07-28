#!/bin/bash

echo "=== 패키지 복구 및 CUDA 수정 ==="

# 1. 기본 패키지들 재설치
echo "1. 기본 패키지 재설치..."
uv pip install transformers datasets gradio wandb

# 2. PyTorch 완전 재설치 (CPU+CUDA 혼합 문제 해결)
echo "2. PyTorch 완전 재설치..."
uv pip uninstall torch torchvision torchaudio -y
uv pip cache clean
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --force-reinstall

# 3. 환경 새로고침
echo "3. 환경 새로고침..."
source ~/.bashrc

# 4. Python 캐시 정리
echo "4. Python 캐시 정리..."
find /opt/conda/envs/venv -name "*.pyc" -delete
find /opt/conda/envs/venv -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# 5. 테스트
echo "5. 최종 테스트..."
python -c "
print('=== 패키지 테스트 ===')
packages = ['torch', 'transformers', 'datasets', 'gradio', 'wandb']
for pkg in packages:
    try:
        module = __import__(pkg)
        print(f'✅ {pkg} - 버전: {getattr(module, \"__version__\", \"N/A\")}')
    except ImportError as e:
        print(f'❌ {pkg}: {e}')

print('\n=== PyTorch CUDA 테스트 ===')
import torch
print(f'PyTorch 버전: {torch.__version__}')
print(f'CUDA 빌드: {torch.version.cuda}')
print(f'CUDA 사용 가능: {hasattr(torch, \"cuda\") and torch.cuda.is_available()}')
if hasattr(torch, 'cuda') and torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU 메모리: {torch.cuda.get_device_properties(0).total_memory // (1024**3)}GB')

print('\n=== unsloth 테스트 ===')
try:
    import unsloth
    print(f'✅ unsloth 버전: {unsloth.__version__}')
    
    # FastLanguageModel 테스트
    from unsloth import FastLanguageModel
    print('✅ FastLanguageModel 임포트 성공')
except Exception as e:
    print(f'❌ unsloth 오류: {e}')
"

echo "=== 복구 완료 ==="
