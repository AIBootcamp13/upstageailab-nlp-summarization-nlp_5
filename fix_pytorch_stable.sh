#!/bin/bash

echo "=== PyTorch 안정 버전으로 수정 ==="

# 1. 환경 활성화 (수동으로)
echo "1. 환경 설정..."
export PATH="/opt/conda/envs/venv/bin:$PATH"
export CONDA_DEFAULT_ENV=venv

# 2. 현재 PyTorch 제거
echo "2. 기존 PyTorch 제거..."
/opt/conda/envs/venv/bin/python -m pip uninstall torch torchvision torchaudio -y

# 3. 더 안정적인 PyTorch 버전 설치 시도
echo "3. PyTorch 1.13.1 + CUDA 11.8 설치..."
/opt/conda/envs/venv/bin/uv pip install torch==1.13.1+cu118 torchvision==0.14.1+cu118 torchaudio==0.13.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# 4. 테스트
echo "4. PyTorch 1.13.1 테스트..."
/opt/conda/envs/venv/bin/python -c "
try:
    import torch
    print(f'✅ PyTorch 1.13.1: {torch.__version__}')
    print(f'   CUDA: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'   GPU: {torch.cuda.get_device_name(0)}')
        x = torch.randn(10, 10).cuda()
        print('   ✅ GPU 연산 성공')
    print('✅ PyTorch 1.13.1 정상 작동')
except Exception as e:
    print(f'❌ PyTorch 1.13.1 실패: {e}')
    print('PyTorch 2.1.0으로 재시도...')
    exit(1)
"

# 5. 1.13.1이 실패하면 2.1.0으로 재시도
if [ $? -ne 0 ]; then
    echo "5. PyTorch 2.1.0 + CUDA 11.8로 재시도..."
    /opt/conda/envs/venv/bin/uv pip uninstall torch torchvision torchaudio -y
    /opt/conda/envs/venv/bin/uv pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118
    
    echo "6. PyTorch 2.1.0 테스트..."
    /opt/conda/envs/venv/bin/python -c "
    try:
        import torch
        print(f'✅ PyTorch 2.1.0: {torch.__version__}')
        print(f'   CUDA: {torch.cuda.is_available()}')
        if torch.cuda.is_available():
            print(f'   GPU: {torch.cuda.get_device_name(0)}')
            x = torch.randn(10, 10).cuda()
            print('   ✅ GPU 연산 성공')
        print('✅ PyTorch 2.1.0 정상 작동')
    except Exception as e:
        print(f'❌ PyTorch 2.1.0도 실패: {e}')
        exit(1)
    "
fi

# 7. bitsandbytes 호환 버전으로 재설치
echo "7. bitsandbytes 호환 버전 재설치..."
/opt/conda/envs/venv/bin/uv pip uninstall bitsandbytes -y
/opt/conda/envs/venv/bin/uv pip install bitsandbytes==0.39.1

# 8. unsloth 재설치
echo "8. unsloth 재설치..."
/opt/conda/envs/venv/bin/uv pip uninstall unsloth -y
/opt/conda/envs/venv/bin/uv pip install unsloth==2024.8

# 9. 최종 통합 테스트
echo "9. 최종 통합 테스트..."
/opt/conda/envs/venv/bin/python -c "
print('=== 최종 통합 테스트 ===')

# PyTorch
try:
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
    print(f'   CUDA: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'   GPU: {torch.cuda.get_device_name(0)}')
except Exception as e:
    print(f'❌ PyTorch: {e}')
    exit(1)

# Transformers
try:
    import transformers
    print(f'✅ Transformers: {transformers.__version__}')
except Exception as e:
    print(f'❌ Transformers: {e}')

# bitsandbytes
try:
    import bitsandbytes as bnb
    print('✅ bitsandbytes: 임포트 성공')
except Exception as e:
    print(f'❌ bitsandbytes: {e}')

# unsloth
try:
    from unsloth import FastLanguageModel
    print('✅ unsloth: FastLanguageModel 임포트 성공')
    print('\n🎉 모든 패키지 정상 작동!')
except Exception as e:
    print(f'❌ unsloth: {e}')
"

echo ""
echo "=== 수정 완료 ==="
echo ""
echo "사용법:"
echo "conda activate venv"
echo "또는 직접: export PATH=\"/opt/conda/envs/venv/bin:\$PATH\""
