#!/bin/bash

echo "=== 기존 venv 삭제 후 재생성 ==="

# 1. 현재 환경에서 나가기
echo "1. 현재 환경 비활성화..."
conda deactivate 2>/dev/null || true

# 2. 기존 venv 환경 완전 삭제
echo "2. 기존 venv 환경 삭제..."
conda env remove -n venv -y 2>/dev/null || true

# 3. 물리적 디렉토리도 삭제 (확실하게)
echo "3. 물리적 디렉토리 삭제..."
rm -rf /opt/conda/envs/venv 2>/dev/null || true

# 4. 새로운 venv 환경 생성
echo "4. 새로운 venv 환경 생성..."
conda create -n venv python=3.11 -y

# 5. 새 환경 활성화
echo "5. venv 환경 활성화..."
conda activate venv

# 6. uv 설치
echo "6. uv 설치..."
pip install uv

# 7. 기본 패키지 설치
echo "7. 기본 패키지 설치..."
uv pip install packaging wheel setuptools "numpy<2.0"

# 8. PyTorch CUDA 11.8 설치
echo "8. PyTorch CUDA 11.8 설치..."
uv pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# 9. ML 패키지들 설치
echo "9. ML 패키지들 설치..."
uv pip install transformers==4.36.0
uv pip install datasets==2.14.0
uv pip install accelerate==0.25.0

# 10. bitsandbytes 설치
echo "10. bitsandbytes 설치..."
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
uv pip install bitsandbytes==0.41.1

# 11. unsloth 설치
echo "11. unsloth 설치..."
uv pip install unsloth==2024.8

# 12. 추가 도구들 설치
echo "12. 추가 패키지 설치..."
uv pip install gradio wandb

# 13. 환경 변수 설정
echo "13. 환경 변수 설정..."
echo 'export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH' >> ~/.bashrc

# 14. 최종 검증
echo "14. 최종 검증..."
python -c "
print('=== 새로운 venv 환경 검증 ===')

# PyTorch 테스트
try:
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
    print(f'   CUDA 사용가능: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'   GPU: {torch.cuda.get_device_name(0)}')
        print(f'   GPU 메모리: {torch.cuda.get_device_properties(0).total_memory // (1024**3)}GB')
        
        # 간단한 GPU 연산 테스트
        x = torch.randn(100, 100).cuda()
        y = x @ x.T
        print('   ✅ GPU 연산 테스트 성공')
except Exception as e:
    print(f'❌ PyTorch 오류: {e}')
    exit(1)

# Transformers 테스트
try:
    import transformers
    print(f'✅ Transformers: {transformers.__version__}')
except Exception as e:
    print(f'❌ Transformers 오류: {e}')

# bitsandbytes 테스트
try:
    import bitsandbytes as bnb
    print('✅ bitsandbytes: 임포트 성공')
except Exception as e:
    print(f'❌ bitsandbytes 오류: {e}')

# unsloth 테스트 (최종)
try:
    from unsloth import FastLanguageModel
    print('✅ unsloth: FastLanguageModel 임포트 성공')
    print('\n🎉 모든 패키지 정상 작동! 환경 설정 완료!')
except Exception as e:
    print(f'❌ unsloth 오류: {e}')
"

echo ""
echo "=== venv 환경 재생성 완료 ==="
echo ""
echo "사용법:"
echo "conda activate venv"
echo ""
echo "테스트 명령어:"
echo "python -c \"from unsloth import FastLanguageModel; print('✅ Ready')\""
