#!/bin/bash

echo "=== CUDA 환경 수정 스크립트 ==="

# 1. CUDA 라이브러리 경로 찾기 및 설정
echo "1. CUDA 라이브러리 검색 및 설정..."

# 일반적인 CUDA 설치 경로들 확인
CUDA_PATHS=(
    "/usr/local/cuda/lib64"
    "/usr/local/cuda-11.8/lib64"
    "/usr/local/cuda-12.0/lib64"
    "/opt/conda/envs/venv/lib"
    "/opt/conda/lib"
)

FOUND_CUDA=""
for path in "${CUDA_PATHS[@]}"; do
    if [ -f "$path/libcudart.so" ] || [ -f "$path/libcudart.so.11.0" ] || [ -f "$path/libcudart.so.12.0" ]; then
        FOUND_CUDA="$path"
        echo "✅ CUDA 라이브러리 발견: $path"
        break
    fi
done

if [ -z "$FOUND_CUDA" ]; then
    echo "⚠️ 표준 경로에서 CUDA 라이브러리를 찾을 수 없습니다. 전체 시스템 검색 시작..."
    FOUND_CUDA=$(find /usr /opt -name "libcudart.so*" 2>/dev/null | head -1 | xargs dirname 2>/dev/null)
    if [ -n "$FOUND_CUDA" ]; then
        echo "✅ CUDA 라이브러리 발견: $FOUND_CUDA"
    fi
fi

# 2. LD_LIBRARY_PATH 설정
if [ -n "$FOUND_CUDA" ]; then
    echo "2. LD_LIBRARY_PATH 환경변수 설정..."
    export LD_LIBRARY_PATH="$FOUND_CUDA:$LD_LIBRARY_PATH"
    
    # .bashrc에 영구 설정 추가
    if ! grep -q "export LD_LIBRARY_PATH.*$FOUND_CUDA" ~/.bashrc; then
        echo "export LD_LIBRARY_PATH=\"$FOUND_CUDA:\$LD_LIBRARY_PATH\"" >> ~/.bashrc
        echo "✅ ~/.bashrc에 LD_LIBRARY_PATH 설정 추가"
    fi
    
    # 현재 세션에서도 적용
    source ~/.bashrc
else
    echo "❌ CUDA 라이브러리를 찾을 수 없습니다."
    echo "   conda 환경에서 CUDA 도구 설치를 시도합니다..."
    
    # conda로 CUDA 도구 설치
    conda install -y -c conda-forge cudatoolkit-dev
fi

# 3. bitsandbytes 재설치
echo "3. bitsandbytes 재설치..."
uv pip uninstall bitsandbytes -y 2>/dev/null || true
uv pip install bitsandbytes --force-reinstall --no-cache-dir

# 4. 환경 테스트
echo "4. 환경 테스트..."
python -c "
import torch
print(f'PyTorch CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

try:
    import bitsandbytes as bnb
    print('✅ bitsandbytes 임포트 성공')
except Exception as e:
    print(f'❌ bitsandbytes 오류: {e}')

try:
    import unsloth
    print('✅ unsloth 임포트 성공')
except Exception as e:
    print(f'❌ unsloth 오류: {e}')
"

echo "=== 수정 완료 ==="
echo "성공하면 'python -c \"import unsloth; print(\'✅ unsloth 정상 작동\')\"' 실행하여 확인하세요"
