#!/bin/bash
# WandB Sweep 실행 스크립트

set -e  # 오류 시 중단

# .env 파일에서 환경 변수 로드
if [ -f .env ]; then
    echo "🔑 .env 파일에서 환경 변수 로드 중..."
    set -a  # 모든 변수를 export
    source .env
    set +a  # export 모드 해제
    echo "✅ WandB API 키 로드 완료"
    echo "   Entity: $WANDB_ENTITY"
    echo "   Project: $WANDB_PROJECT"
else
    echo "⚠️  .env 파일을 찾을 수 없습니다!"
    echo "👉 .env.template을 복사하여 .env 파일을 생성하세요."
    exit 1
fi

# 추가 환경 변수 설정
export TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS="ignore"

echo "🚀 WandB Sweep 실행"
echo "======================================="
echo ""

# 사용법 함수
usage() {
    echo "사용법: $0 [sweep_type] [options]"
    echo ""
    echo "Sweep 타입:"
    echo "  mt5        - mT5 XL-Sum 모델 하이퍼파라미터 최적화"
    echo "  eenzeenee  - eenzeenee T5 모델 하이퍼파라미터 최적화"
    echo "  baseline   - 기본 모델 하이퍼파라미터 최적화"
    echo ""
    echo "옵션:"
    echo "  --count N  - 실행할 실험 수 (기본값: 20)"
    echo "  --direct   - sweep_runner.py 대신 직접 wandb sweep 사용"
    echo ""
    echo "예시:"
    echo "  $0 mt5 --count 50"
    echo "  $0 eenzeenee --direct"
    exit 1
}

# 인자 확인
if [ $# -lt 1 ]; then
    usage
fi

SWEEP_TYPE=$1
COUNT=20
USE_DIRECT=false

# 옵션 파싱
shift
while [[ $# -gt 0 ]]; do
    case $1 in
        --count)
            COUNT=$2
            shift 2
            ;;
        --direct)
            USE_DIRECT=true
            shift
            ;;
        *)
            echo "❌ 알 수 없는 옵션: $1"
            usage
            ;;
    esac
done

# Sweep 설정 파일 확인
case $SWEEP_TYPE in
    mt5)
        SWEEP_CONFIG="sweep_mt5.yaml"
        DESCRIPTION="mT5 XL-Sum 모델 하이퍼파라미터 최적화"
        ;;
    eenzeenee)
        SWEEP_CONFIG="sweep_eenzeenee.yaml"
        DESCRIPTION="eenzeenee T5 모델 하이퍼파라미터 최적화"
        ;;
    baseline)
        # baseline sweep 설정이 없으면 생성
        if [ ! -f "sweep_baseline.yaml" ]; then
            echo "📝 baseline sweep 설정 파일 생성 중..."
            cat > sweep_baseline.yaml << EOF
# WandB Sweep 설정 - Baseline 모델 하이퍼파라미터 튜닝
program: code/trainer.py
method: bayes
project: nlp-5
entity: lyjune37-juneictlab
metric:
  goal: maximize
  name: eval_rouge_combined_f1
parameters:
  learning_rate:
    distribution: log_uniform_values
    min: 1e-5
    max: 1e-4
  per_device_train_batch_size:
    values: [4, 8, 16]
  gradient_accumulation_steps:
    values: [1, 2, 4]
  warmup_ratio:
    values: [0.1, 0.2, 0.3]
  weight_decay:
    values: [0.0, 0.01, 0.1]
  num_train_epochs:
    values: [3, 5, 10]
  # 실험 설정
  config:
    value: config.yaml
  train-data:
    value: data/train.csv
  val-data:
    value: data/dev.csv
early_terminate:
  type: hyperband
  s: 2
  eta: 3
  max_iter: 20
EOF
        fi
        SWEEP_CONFIG="sweep_baseline.yaml"
        DESCRIPTION="Baseline 모델 하이퍼파라미터 최적화"
        ;;
    *)
        echo "❌ 알 수 없는 sweep 타입: $SWEEP_TYPE"
        usage
        ;;
esac

echo "📊 Sweep 설정:"
echo "   타입: $SWEEP_TYPE"
echo "   설명: $DESCRIPTION"
echo "   설정 파일: $SWEEP_CONFIG"
echo "   실행 횟수: $COUNT"
echo "   실행 방식: $([ "$USE_DIRECT" = true ] && echo "직접 실행" || echo "sweep_runner.py 사용")"
echo ""

# GPU 상태 확인
echo "🔍 GPU 상태 확인:"
nvidia-smi --query-gpu=name,memory.free,memory.used,memory.total,temperature.gpu,utilization.gpu --format=csv,noheader
echo ""

# 실행 확인
read -p "🤔 Sweep을 시작하시겠습니까? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ 취소되었습니다."
    exit 1
fi

# 로그 디렉토리 생성
LOG_DIR="logs/sweep_${SWEEP_TYPE}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "📁 로그 디렉토리: $LOG_DIR"

# Sweep 실행
if [ "$USE_DIRECT" = true ]; then
    # 직접 wandb sweep 사용
    echo "🏃 WandB Sweep 직접 실행..."
    
    # Sweep 생성
    echo "📝 Sweep 생성 중..."
    SWEEP_ID=$(wandb sweep "$SWEEP_CONFIG" --entity "$WANDB_ENTITY" --project "$WANDB_PROJECT" 2>&1 | tee "$LOG_DIR/sweep_create.log" | grep -oP 'wandb: Created sweep with ID: \K[a-zA-Z0-9]+')
    
    if [ -z "$SWEEP_ID" ]; then
        echo "❌ Sweep ID를 가져올 수 없습니다."
        exit 1
    fi
    
    echo "✅ Sweep ID: $SWEEP_ID"
    echo ""
    
    # Agent 실행
    echo "🏃 WandB Agent 실행 중..."
    wandb agent "$WANDB_ENTITY/$WANDB_PROJECT/$SWEEP_ID" --count "$COUNT" 2>&1 | tee "$LOG_DIR/sweep_agent.log"
else
    # sweep_runner.py 사용
    echo "🏃 sweep_runner.py를 사용한 Sweep 실행..."
    
    python code/sweep_runner.py \
        --base-config config.yaml \
        --sweep-config "$SWEEP_TYPE" \
        --count "$COUNT" \
        --entity "$WANDB_ENTITY" \
        --project "$WANDB_PROJECT" \
        2>&1 | tee "$LOG_DIR/sweep_runner.log"
fi

echo ""
echo "✅ Sweep 완료!"
echo "📊 결과 확인:"
echo "   - WandB: https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT"
echo "   - 로그: $LOG_DIR"
echo ""
echo "💡 팁: 최적의 하이퍼파라미터는 WandB 대시보드에서 확인할 수 있습니다."
