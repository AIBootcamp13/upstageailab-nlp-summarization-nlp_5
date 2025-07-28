#!/bin/bash

echo "=== Unsloth 공식 설치 방법으로 수정 ==="

# 1. 현재 환경에서 나가기
echo "1. 현재 환경 비활성화..."
conda deactivate 2>/dev/null || true

# 2. 기존 venv 환경 완전 삭제
echo "2. 기존 venv 환경 삭제..."
conda env remove -n venv -y 2>/dev/null || true
rm -rf /opt/conda/envs/venv 2>/dev/null || true

# 3. Unsloth 공식 설치 방법 적용
echo "3. Unsloth 공식 방법으로 환경 생성..."
conda create --name venv \
    python=3.11 \
    pytorch-cuda=12.1 \
    pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
    -y

# 4. 환경 활성화 (conda init 실행 후)
echo "4. conda 초기화 및 환경 활성화..."
eval "$(/opt/conda/bin/conda shell.bash hook)"
conda activate venv

# 5. CUDA 링크 수정 (중요!)
echo "5. CUDA 링크 수정..."
sudo ldconfig /usr/local/cuda*/lib64 2>/dev/null || true
sudo ldconfig /opt/conda/lib 2>/dev/null || true

# 6. 환경 변수 설정
echo "6. 환경 변수 설정..."
export LD_LIBRARY_PATH=/opt/conda/lib:/usr/local/cuda*/lib64:$LD_LIBRARY_PATH
export BNB_CUDA_VERSION=""  # 자동 감지하도록

# 7. bitsandbytes 설치 (conda로)
echo "7. bitsandbytes conda 설치..."
conda install -c conda-forge bitsandbytes -y || {
    echo "conda 설치 실패, pip으로 재시도..."
    pip install bitsandbytes
}

# 8. unsloth 설치
echo "8. unsloth 설치..."
pip install unsloth

# 9. 추가 패키지 설치
echo "9. 추가 패키지 설치..."
pip install transformers datasets accelerate gradio wandb

# 10. 환경 설정을 bashrc에 저장
echo "10. 환경 설정 저장..."
cat >> ~/.bashrc << 'EOF'

# Unsloth 환경 설정
export LD_LIBRARY_PATH=/opt/conda/lib:/usr/local/cuda*/lib64:$LD_LIBRARY_PATH
export BNB_CUDA_VERSION=""
EOF

# 11. 최종 테스트
echo "11. 최종 테스트..."
python -c "
print('=== Unsloth 환경 테스트 ===')

# PyTorch CUDA 테스트
import torch
print(f'✅ PyTorch: {torch.__version__}')
print(f'   CUDA 사용가능: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'   GPU: {torch.cuda.get_device_name(0)}')
    print(f'   GPU 메모리: {torch.cuda.get_device_properties(0).total_memory // (1024**3)}GB')

# bitsandbytes 테스트
try:
    import bitsandbytes as bnb
    print('✅ bitsandbytes: 정상 임포트')
except Exception as e:
    print(f'❌ bitsandbytes 오류: {e}')

# unsloth 테스트
try:
    from unsloth import FastLanguageModel
    print('✅ unsloth: FastLanguageModel 임포트 성공')
    print('\n🎉 모든 패키지 정상 작동!')
except Exception as e:
    print(f'❌ unsloth 오류: {e}')
    print('\n📋 디버그 정보:')
    print('python -m bitsandbytes 실행 결과:')
    import subprocess
    try:
        result = subprocess.run(['python', '-m', 'bitsandbytes'], 
                              capture_output=True, text=True, timeout=10)
        print(result.stdout)
        if result.stderr:
            print('STDERR:', result.stderr)
    except Exception as debug_e:
        print(f'디버그 실행 실패: {debug_e}')
"

echo ""
echo "=== 설치 완료 ==="
echo ""
echo "사용법:"
echo "conda activate venv"
echo ""
echo "문제 발생시:"
echo "sudo ldconfig /opt/conda/lib"
echo "python -m bitsandbytes"
