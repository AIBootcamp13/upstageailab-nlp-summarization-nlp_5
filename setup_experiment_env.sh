#!/bin/bash
# WandB 오프라인 모드 및 필요한 환경 설정 스크립트

echo "🔧 실험 환경 설정 중..."

# WandB 오프라인 모드 설정
export WANDB_MODE=offline
echo "✅ WandB 오프라인 모드 설정 완료"

# 기타 환경 변수 설정
export TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS="ignore"

echo "✅ 환경 설정 완료!"
echo ""
echo "설정된 환경 변수:"
echo "  WANDB_MODE=$WANDB_MODE"
echo "  TOKENIZERS_PARALLELISM=$TOKENIZERS_PARALLELISM"
echo "  PYTHONWARNINGS=$PYTHONWARNINGS"
echo ""
echo "이제 실험을 실행할 수 있습니다:"
echo "  bash run_main_5_experiments.sh"
