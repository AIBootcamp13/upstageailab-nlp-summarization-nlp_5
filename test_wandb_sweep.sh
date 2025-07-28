#!/bin/bash
# WandB Sweep 빠른 테스트 스크립트

set -e  # 오류 시 중단

# .env 파일에서 환경 변수 로드
if [ -f .env ]; then
    echo "🔑 .env 파일에서 환경 변수 로드 중..."
    set -a
    source .env
    set +a
    echo "✅ WandB API 키 로드 완료"
else
    echo "⚠️  .env 파일을 찾을 수 없습니다!"
    exit 1
fi

echo "🧪 WandB Sweep 빠른 테스트"
echo "======================================="
echo "📋 테스트 목적:"
echo "  - WandB 연결 확인"
echo "  - Sweep 생성 및 실행 테스트"
echo "  - 1-2개 실험만 빠르게 실행"
echo ""

# 간단한 테스트 sweep 설정 생성
cat > sweep_test.yaml << EOF
# WandB Sweep 테스트 설정
program: code/trainer.py
method: random  # 빠른 테스트를 위해 random 사용
entity: $WANDB_ENTITY
project: $WANDB_PROJECT
metric:
  goal: maximize
  name: eval_rouge_combined_f1
parameters:
  # 최소한의 파라미터만 테스트
  learning_rate:
    values: [3e-5, 5e-5]
  
  per_device_train_batch_size:
    value: 8
  
  num_train_epochs:
    value: 1  # 빠른 테스트를 위해 1 에포크만
  
  # 고정 파라미터
  config:
    value: config/experiments/00_baseline_mini_test.yaml
  
  sweep:
    value: true
  
  # 빠른 테스트를 위한 설정
  logging_steps:
    value: 10
  
  eval_steps:
    value: 50
  
  save_steps:
    value: 50
EOF

echo "📝 테스트 Sweep 설정 파일 생성 완료"
echo ""

# Sweep 생성
echo "🚀 Sweep 생성 중..."
SWEEP_ID=$(wandb sweep sweep_test.yaml 2>&1 | grep -oP 'wandb: Created sweep with ID: \K[a-zA-Z0-9]+')

if [ -z "$SWEEP_ID" ]; then
    echo "❌ Sweep ID를 가져올 수 없습니다."
    echo "🔍 디버깅을 위해 다시 실행:"
    wandb sweep sweep_test.yaml
    exit 1
fi

echo "✅ Sweep ID: $SWEEP_ID"
echo "🔗 Sweep URL: https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT/sweeps/$SWEEP_ID"
echo ""

# Agent 실행
echo "🏃 WandB Agent 실행 (2개 실험만)..."
wandb agent "$WANDB_ENTITY/$WANDB_PROJECT/$SWEEP_ID" --count 2

echo ""
echo "✅ 테스트 완료!"
echo "📊 결과 확인: https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT/sweeps/$SWEEP_ID"

# 테스트 파일 정리
rm -f sweep_test.yaml
echo "🧹 임시 파일 정리 완료"
