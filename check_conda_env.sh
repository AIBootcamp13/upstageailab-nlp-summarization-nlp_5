#!/bin/bash
# Conda 환경 검증 스크립트

echo "🔍 Conda 환경 검증 시작"
echo "=================================="

# 1. 현재 환경 확인
echo "📋 현재 활성화된 환경:"
echo "CONDA_DEFAULT_ENV: $CONDA_DEFAULT_ENV"
echo "CONDA_PREFIX: $CONDA_PREFIX"

# 2. Conda 환경 목록
echo ""
echo "📝 사용 가능한 Conda 환경들:"
conda env list

# 3. Python 경로 및 버전 확인
echo ""
echo "🐍 Python 정보:"
echo "Current python: $(which python)"
echo "Current python3: $(which python3)"
echo "Python version: $(python --version 2>&1)"
echo "Python3 version: $(python3 --version 2>&1)"

# 4. 실험 스크립트에서 사용하는 Python 확인
echo ""
echo "🧪 실험용 Python 확인:"
if [ -f "/opt/conda/envs/python311/bin/python3.11" ]; then
    echo "✅ 실험용 Python 존재: /opt/conda/envs/python311/bin/python3.11"
    echo "   버전: $(/opt/conda/envs/python311/bin/python3.11 --version)"
else
    echo "❌ 실험용 Python 없음: /opt/conda/envs/python311/bin/python3.11"
fi

# 5. UV 환경 확인
echo ""
echo "🔄 UV 환경 확인:"
if conda env list | grep -q "uv"; then
    echo "✅ UV 환경 존재"
    if [ "$CONDA_DEFAULT_ENV" = "uv" ]; then
        echo "✅ UV 환경 활성화됨"
    else
        echo "⚠️  UV 환경이 활성화되지 않음"
        echo "   활성화 명령: conda activate uv"
    fi
else
    echo "❌ UV 환경 없음"
fi

# 6. 필수 패키지 확인
echo ""
echo "📦 필수 패키지 확인:"
python3 -c "
import sys
packages = ['torch', 'transformers', 'datasets', 'wandb', 'numpy', 'pandas']
missing = []
for pkg in packages:
    try:
        __import__(pkg)
        print(f'✅ {pkg}')
    except ImportError:
        print(f'❌ {pkg}')
        missing.append(pkg)

if missing:
    print(f'누락된 패키지: {missing}')
    sys.exit(1)
else:
    print('✅ 모든 필수 패키지 설치됨')
"

# 7. GPU 확인
echo ""
echo "🖥️  GPU 상태:"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
else
    echo "❌ nvidia-smi 없음"
fi

echo ""
echo "🎯 권장사항:"
if [ "$CONDA_DEFAULT_ENV" != "uv" ] && conda env list | grep -q "uv"; then
    echo "⚠️  UV 환경으로 전환하세요: conda activate uv"
elif [ "$CONDA_DEFAULT_ENV" != "python311" ]; then
    echo "⚠️  python311 환경으로 전환하세요: conda activate python311"
else
    echo "✅ 환경 설정이 올바릅니다!"
fi

echo ""
echo "🚀 실험 실행 명령 예시:"
echo "conda activate uv  # 또는 conda activate python311"
echo "/opt/conda/envs/python311/bin/python3.11 code/auto_experiment_runner.py --configs config/experiments/mt5_xlsum_ultimate_korean_qlora.yaml --one-epoch"
