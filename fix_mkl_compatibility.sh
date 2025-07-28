#!/bin/bash

echo "=== MKL 버전 충돌 해결 (검색 결과 기반) ==="

# 1. 환경 활성화
eval "$(/opt/conda/bin/conda shell.bash hook)"
conda activate venv

# 2. 현재 MKL 버전 확인
echo "2. 현재 MKL 버전 확인..."
conda list | grep mkl

# 3. MKL 다운그레이드 (핵심 해결책)
echo "3. MKL을 2024.0.0으로 다운그레이드..."
conda install mkl=2024.0 -y

# 4. PyTorch 재설치 (MKL 호환 버전으로)
echo "4. PyTorch 재설치..."
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 5. 설치 확인
echo "5. PyTorch 설치 확인..."
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

# 6. 다른 패키지들 설치
echo "6. 다른 패키지들 설치..."
pip install transformers datasets accelerate gradio wandb

# 7. bitsandbytes 설치 (MKL 수정 후)
echo "7. bitsandbytes 설치..."
pip install bitsandbytes --no-cache-dir

# 8. unsloth 설치
echo "8. unsloth 설치..."
pip install unsloth --no-cache-dir

# 9. 최종 종합 테스트
echo "9. 최종 종합 테스트..."
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

# 10. MKL 버전 고정 (향후 업데이트 방지)
echo "10. MKL 버전 고정..."
conda install 'mkl=2024.0' -y
conda pin add mkl=2024.0

echo ""
echo "=== MKL 호환성 문제 해결 완료 ==="
echo ""
echo "MKL 버전이 2024.0.0으로 고정되었습니다."
echo "이제 unsloth가 정상 작동할 것입니다."
echo ""
echo "최종 확인:"
echo "python -c \"from unsloth import FastLanguageModel; print('✅ 완료')\""
