#!/bin/bash
# unsloth 설치 스크립트
# PyTorch 2.4+ 및 CUDA 환경에서 실행

set -e

echo "=================================="
echo "unsloth Installation Script"
echo "=================================="

# Python 버전 확인
echo "Checking Python version..."
python --version

# PyTorch 버전 확인
echo -e "\nChecking PyTorch version..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# PyTorch 2.4+ 확인
PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
MAJOR_VERSION=$(echo $PYTORCH_VERSION | cut -d. -f1)
MINOR_VERSION=$(echo $PYTORCH_VERSION | cut -d. -f2)

if [ "$MAJOR_VERSION" -lt 2 ] || ([ "$MAJOR_VERSION" -eq 2 ] && [ "$MINOR_VERSION" -lt 4 ]); then
    echo "⚠️  Warning: PyTorch 2.4+ is required for unsloth"
    echo "Current version: $PYTORCH_VERSION"
    echo ""
    echo "Please upgrade PyTorch first:"
    echo "pip install torch>=2.4.0"
    exit 1
fi

echo -e "\n✅ PyTorch version is compatible"

# unsloth 설치
echo -e "\nInstalling unsloth..."
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# 필요한 추가 패키지 설치
echo -e "\nInstalling additional dependencies..."
pip install xformers trl peft accelerate bitsandbytes

# 설치 확인
echo -e "\nVerifying installation..."
python -c "
try:
    from unsloth import FastLanguageModel
    print('✅ unsloth successfully installed!')
except ImportError as e:
    print('❌ unsloth installation failed:', e)
"

python -c "
try:
    import peft
    print('✅ peft successfully installed!')
except ImportError as e:
    print('❌ peft installation failed:', e)
"

python -c "
try:
    import bitsandbytes
    print('✅ bitsandbytes successfully installed!')
except ImportError as e:
    print('❌ bitsandbytes installation failed:', e)
"

echo -e "\n=================================="
echo "Installation complete!"
echo "=================================="
echo ""
echo "To use unsloth in your experiments:"
echo "1. Set 'use_unsloth: true' in your config file"
echo "2. Set 'use_qlora: true' for QLoRA optimization"
echo "3. Run experiments with kobart_unsloth.yaml config"
echo ""
echo "Example:"
echo "python code/trainer.py --config config/model_configs/kobart_unsloth.yaml"
