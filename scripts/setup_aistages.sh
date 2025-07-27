#!/bin/bash

# AIStages 최신 기술 스택 자동 설정 스크립트
# conda 가상환경에서 nlp-sum-lyj 최신 버전을 실행하기 위한 자동 설정

set -e

echo "🏢 AIStages conda 환경 자동 설정 시작"
echo "==============================="

# conda 환경 감지
if ! command -v conda &> /dev/null; then
    echo "❌ conda가 설치되어 있지 않습니다. AIStages 환경이 아닙니다."
    exit 1
fi

echo "✅ conda 환경 감지됨"

# Python 3.11 가상환경 생성
echo "🐍 Python 3.11 가상환경 생성..."
conda create -n nlp-sum-latest python==3.11 -y
source activate nlp-sum-latest

echo "✅ Python 3.11 가상환경 생성 완료"

# 최신 라이브러리 설치
echo "📦 최신 기술 스택 설치..."
uv pip install -r requirements.txt
echo "✅ 모든 라이브러리 설치 완료"

# AIStages용 .env 파일 생성
# AIStages용 .env 파일 생성
cat > .env << EOF
# AIStages Environment Configuration
AISTAGES_WORKSPACE=/opt/ml
AISTAGES_INPUT=/opt/ml/input/data
AISTAGES_OUTPUT=/opt/ml/output

# CUDA Configuration (AIStages GPU)
CUDA_VISIBLE_DEVICES=0
CUDA_DEVICE_ORDER=PCI_BUS_ID

# Training Configuration for AIStages
OUTPUT_DIR=/opt/ml/output
MODEL_CACHE_DIR=/opt/ml/models
LOG_LEVEL=INFO

# Memory Optimization for AIStages
TORCH_COMPILE=false
GRADIENT_CHECKPOINTING=true
DATALOADER_NUM_WORKERS=4

# Enable unsloth in Linux environment
USE_UNSLOTH=true
USE_QLORA=true

# Weights & Biases (비활성화, AIStages에서는 로컬 로깅 사용)
WANDB_DISABLED=true
WANDB_MODE=offline

# System Optimization
OMP_NUM_THREADS=4
TOKENIZERS_PARALLELISM=false
EOF

echo "✅ AIStages용 .env 파일 생성 완료"

# config.yaml을 AIStages 환경에 맞게 조정
if [ -f "config.yaml" ]; then
    # 백업 생성
    cp config.yaml config.yaml.aistages_backup
    
    # AIStages용 설정 적용 (Python으로 YAML 수정)
    python3 << EOF
import yaml

with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# AIStages 환경에 맞는 설정 조정
config['data']['input_dir'] = '/opt/ml/input/data'
config['model']['output_dir'] = '/opt/ml/output'
config['model']['cache_dir'] = '/opt/ml/models'

# 로깅 설정 조정
config['logging']['tensorboard_log_dir'] = '/opt/ml/output/logs'
config['logging']['output_dir'] = '/opt/ml/output'

# 성능 최적화 (AIStages GPU 환경)
config['training']['dataloader_num_workers'] = 4
config['training']['save_steps'] = 200
config['training']['eval_steps'] = 200

# unsloth 활성화 (Linux 환경)
if 'qlora' in config:
    config['qlora']['use_unsloth'] = True
    config['qlora']['use_qlora'] = True

with open('config.yaml', 'w', encoding='utf-8') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)

print("✅ config.yaml AIStages 환경 설정 완료")
EOF

fi

# 라이브러리 설치 및 검증
echo "📦 라이브러리 설치 및 검증..."

# 라이브러리 버전 확인
python3 << EOF
import torch
import transformers
import pytorch_lightning
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Transformers: {transformers.__version__}')
print(f'PyTorch Lightning: {pytorch_lightning.__version__}')

# unsloth 확인
try:
    import unsloth
    print('✅ unsloth 사용 가능')
except ImportError:
    print('⚠️  unsloth 없음, QLoRA 모드 사용')
EOF

# 권한 설정
chmod +x scripts/*.sh 2>/dev/null || true

echo ""
echo "🎉 AIStages conda 환경 설정이 완료되었습니다!"
echo ""
echo "📋 설정 내용:"
echo "• Python 3.11 conda 가상환경"
echo "• 조장님 최신 기술 스택 적용"
echo "• 업그레이드된 최신 기술 스택 적용"
echo "• requirements.txt 기반 설치"
echo "• conda 환경으로 안전한 격리"
echo ""
echo "🚀 학습 시작 준비 완료!"
echo "python main_base.py 또는 해당 학습 스크립트를 실행하세요."
