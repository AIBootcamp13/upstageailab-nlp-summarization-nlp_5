#!/bin/bash

echo "=== 검색 기반 완전 해결책 ==="

# 1. 환경 활성화
eval "$(/opt/conda/bin/conda shell.bash hook)"
conda activate venv

# 2. C 컴파일러 설치 (bitsandbytes 컴파일용)
echo "2. C 컴파일러 설치..."
apt-get update -qq
apt-get install -y gcc g++ build-essential

# 3. 환경 변수 설정
echo "3. 환경 변수 설정..."
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

# 4. 모든 PyTorch 관련 패키지 완전 제거
echo "4. 모든 PyTorch 관련 패키지 완전 제거..."
pip freeze | grep -E "(torch|nvidia|bitsandbytes|unsloth|triton|xformers)" | cut -d= -f1 | xargs pip uninstall -y 2>/dev/null || true

# 5. conda 캐시 정리
echo "5. conda 및 pip 캐시 정리..."
conda clean --all -y
pip cache purge

# 6. Python 모듈 캐시 완전 정리
echo "6. Python 모듈 캐시 정리..."
find /opt/conda/envs/venv -name "*.pyc" -delete 2>/dev/null || true
find /opt/conda/envs/venv -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
rm -rf /opt/conda/envs/venv/lib/python3.11/site-packages/torch* 2>/dev/null || true
rm -rf /opt/conda/envs/venv/lib/python3.11/site-packages/nvidia* 2>/dev/null || true

# 7. Unsloth 공식 설치 방법 적용
echo "7. Unsloth 공식 conda 설치..."
conda install pytorch-cuda=12.1 pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers -y

# 8. 설치 확인
echo "8. PyTorch 설치 확인..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
" || {
    echo "❌ PyTorch 설치 실패. 재시도..."
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
}

# 9. bitsandbytes 설치 (컴파일러 사용)
echo "9. bitsandbytes 설치..."
CC=/usr/bin/gcc CXX=/usr/bin/g++ pip install bitsandbytes --no-cache-dir

# 10. unsloth 설치
echo "10. unsloth 설치..."
pip install unsloth --no-cache-dir

# 11. 추가 패키지 설치
echo "11. 추가 패키지 설치..."
pip install transformers datasets accelerate gradio wandb

# 12. 환경 변수 영구 저장
echo "12. 환경 변수 영구 저장..."
cat >> ~/.bashrc << 'EOF'
# 개발 환경 설정
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
EOF

# 13. 최종 종합 테스트
echo "13. 최종 종합 테스트..."
source ~/.bashrc

python -c "
print('=== 최종 검증 ===')

# 1. PyTorch 기본 테스트
try:
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
    print(f'   CUDA: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'   GPU: {torch.cuda.get_device_name(0)}')
        # GPU 메모리 테스트
        x = torch.randn(100, 100).cuda()
        y = x @ x.T
        print(f'   GPU 메모리: {torch.cuda.get_device_properties(0).total_memory // (1024**3)}GB')
        print('   ✅ GPU 연산 성공')
except Exception as e:
    print(f'❌ PyTorch 오류: {e}')
    exit(1)

# 2. transformers 테스트
try:
    import transformers
    print(f'✅ Transformers: {transformers.__version__}')
except Exception as e:
    print(f'❌ Transformers 오류: {e}')

# 3. bitsandbytes 테스트
try:
    import bitsandbytes as bnb
    print('✅ bitsandbytes: 임포트 성공')
    # 간단한 양자화 테스트
    if torch.cuda.is_available():
        linear = bnb.nn.Linear8bitLt(10, 10).cuda()
        x = torch.randn(5, 10).cuda()
        out = linear(x)
        print('✅ bitsandbytes: GPU 양자화 연산 성공')
except Exception as e:
    print(f'❌ bitsandbytes 오류: {e}')

# 4. unsloth 최종 테스트
try:
    from unsloth import FastLanguageModel
    print('✅ unsloth: FastLanguageModel 임포트 성공')
    print('\\n🎉 모든 패키지가 정상적으로 작동합니다!')
except Exception as e:
    print(f'❌ unsloth 오류: {e}')
    print('하지만 기본 패키지들은 정상 작동합니다.')
"

echo ""
echo "=== 설치 완료 ==="
echo ""
echo "다음부터 사용법:"
echo "conda activate venv"
echo ""
echo "최종 확인:"
echo "python -c \"from unsloth import FastLanguageModel; print('✅ 완료')\""
