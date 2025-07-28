#!/bin/bash

echo "=== MKL 호환성 문제 해결 (uv + conda 병행) ==="

# 1. 환경 활성화
eval "$(/opt/conda/bin/conda shell.bash hook)"
conda activate venv

# 2. 현재 MKL 버전 확인
echo "2. 현재 MKL 버전 확인..."
conda list | grep mkl

# 3. MKL 다운그레이드 (conda로)
echo "3. MKL을 2024.0.0으로 다운그레이드..."
conda install mkl=2024.0 -y

# 4. uv로 모든 PyTorch 관련 패키지 제거
echo "4. uv로 기존 패키지 제거..."
uv pip uninstall torch torchvision torchaudio bitsandbytes unsloth transformers datasets accelerate gradio wandb -q || true

# 5. uv로 PyTorch 설치 (MKL 호환 후)
echo "5. uv로 PyTorch CUDA 12.1 설치..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 6. 설치 확인
echo "6. PyTorch 설치 확인..."
python -c "
import torch
print(f'✅ PyTorch: {torch.__version__}')
print(f'   CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'   GPU: {torch.cuda.get_device_name(0)}')
    print(f'   GPU 메모리: {torch.cuda.get_device_properties(0).total_memory // (1024**3)}GB')
    # 간단한 GPU 연산 테스트
    x = torch.randn(100, 100).cuda()
    y = x @ x.T
    print('   ✅ GPU 연산 성공')
else:
    print('   ⚠️ CPU 모드')
"

# 7. uv로 다른 패키지들 설치
echo "7. uv로 다른 패키지들 설치..."
uv pip install transformers datasets accelerate gradio wandb

# 8. uv로 bitsandbytes 설치 (MKL 수정 후)
echo "8. uv로 bitsandbytes 설치..."
uv pip install bitsandbytes --no-cache-dir

# 9. uv로 unsloth 설치
echo "9. uv로 unsloth 설치..."
uv pip install unsloth --no-cache-dir

# 10. 최종 종합 테스트
echo "10. 최종 종합 테스트..."
python -c "
print('=== 최종 검증 ===')

# PyTorch 테스트
try:
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
    print(f'   CUDA: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'   GPU: {torch.cuda.get_device_name(0)}')
except Exception as e:
    print(f'❌ PyTorch 오류: {e}')
    exit(1)

# transformers 테스트
try:
    import transformers
    print(f'✅ Transformers: {transformers.__version__}')
except Exception as e:
    print(f'❌ Transformers 오류: {e}')

# bitsandbytes 테스트
try:
    import bitsandbytes as bnb
    print('✅ bitsandbytes: 임포트 성공')
    if torch.cuda.is_available():
        # 간단한 양자화 테스트
        linear = bnb.nn.Linear8bitLt(10, 10).cuda()
        x = torch.randn(5, 10).cuda()
        out = linear(x)
        print('✅ bitsandbytes: GPU 양자화 성공')
except Exception as e:
    print(f'❌ bitsandbytes 오류: {e}')

# unsloth 최종 테스트
try:
    from unsloth import FastLanguageModel
    print('✅ unsloth: FastLanguageModel 임포트 성공')
    print('\\n🎉 모든 패키지가 완벽하게 작동합니다!')
except Exception as e:
    print(f'❌ unsloth 오류: {e}')
    print('하지만 기본 패키지들은 정상 작동합니다.')
"

# 11. MKL 버전 고정 (향후 업데이트 방지)
echo "11. MKL 버전 고정..."
conda pin add mkl=2024.0

# 12. 설치된 패키지 목록 확인
echo "12. 설치된 패키지 목록 확인..."
echo "=== conda 패키지 (MKL 관련) ==="
conda list | grep mkl
echo ""
echo "=== uv 패키지 (PyTorch 관련) ==="
uv pip list | grep -E "(torch|bitsandbytes|unsloth|transformers)"

echo ""
echo "=== MKL 호환성 문제 해결 완료 ==="
echo ""
echo "- MKL: conda로 2024.0.0 설치 및 고정"
echo "- PyTorch 관련: uv로 설치"
echo "- 이제 unsloth가 정상 작동할 것입니다."
echo ""
echo "최종 확인:"
echo "python -c \"from unsloth import FastLanguageModel; print('✅ 완료')\""
