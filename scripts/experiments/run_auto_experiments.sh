#!/bin/bash
# 자동 실험 실행 스크립트 (상대 경로 기준, MPS/CUDA 최적화)

echo "🚀 NLP 대화 요약 자동 실험 시스템 시작"

# 프로젝트 루트에서 실행 확인
if [ ! -d "code" ] || [ ! -d "config" ] || [ ! -d "docs" ]; then
    echo "❌ 프로젝트 루트에서 실행하세요 (code/, config/, docs/ 디렉토리 필요)"
    exit 1
fi

# Python 환경 확인
echo "🔍 Python 환경 확인..."
python --version
if ! python -c "import torch, transformers, pandas" 2>/dev/null; then
    echo "❌ 필수 라이브러리가 설치되지 않았습니다"
    echo "pip install torch transformers pandas pyyaml 실행 후 다시 시도하세요"
    exit 1
fi

# 디바이스 감지
echo "🖥️ 실행 디바이스 감지 중..."
DEVICE=$(python -c "
import sys
sys.path.insert(0, 'code')
try:
    from utils.device_utils import get_optimal_device
    print(get_optimal_device())
except ImportError:
    print('cpu')
")
echo "감지된 디바이스: $DEVICE"

# 실험 설정 디렉토리 확인
EXPERIMENT_DIR="config/experiments"
if [ ! -d "$EXPERIMENT_DIR" ]; then
    echo "📁 실험 설정 디렉토리가 없습니다. 샘플 설정을 생성합니다..."
    python code/auto_experiment_runner.py --create-samples --config-dir "$EXPERIMENT_DIR"
    
    if [ $? -ne 0 ]; then
        echo "❌ 샘플 설정 생성 실패"
        exit 1
    fi
    
    echo "✅ 샘플 설정 생성 완료"
fi

# 실험 설정 파일 목록 출력
echo ""
echo "📋 발견된 실험 설정 파일들:"
for file in "$EXPERIMENT_DIR"/*.yaml "$EXPERIMENT_DIR"/*.yml; do
    if [ -f "$file" ]; then
        echo "  - $(basename "$file")"
    fi
done

# 사용자 확인
echo ""
echo "🤔 위 실험들을 순차적으로 실행하시겠습니까?"
echo "   (각 실험은 약 30분-2시간 소요될 수 있습니다)"
read -p "계속하려면 'y' 또는 'yes'를 입력하세요: " confirm

if [[ ! "$confirm" =~ ^[Yy]([Ee][Ss])?$ ]]; then
    echo "❌ 실행이 취소되었습니다"
    exit 0
fi

# 출력 디렉토리 설정
OUTPUT_DIR="outputs/auto_experiments_$(date +%Y%m%d_%H%M%S)"
echo "📁 결과는 $OUTPUT_DIR 에 저장됩니다"

# 자동 실험 실행
echo ""
echo "🚀 자동 실험 시작..."
echo "=============================================="

python code/auto_experiment_runner.py \
    --base-config "config/base_config.yaml" \
    --config-dir "$EXPERIMENT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --run-all

RESULT=$?

echo ""
echo "=============================================="

if [ $RESULT -eq 0 ]; then
    echo "🎉 모든 실험이 성공적으로 완료되었습니다!"
    echo ""
    echo "📊 결과 확인:"
    echo "  - 실험 요약: $OUTPUT_DIR/experiment_summary.json"
    echo "  - 실험 로그: $OUTPUT_DIR/experiments/"
    echo "  - 모델 저장: $OUTPUT_DIR/models/"
    echo ""
    echo "📈 최고 성능 모델을 확인하려면:"
    echo "  python -c \"
import json
with open('$OUTPUT_DIR/experiment_summary.json', 'r') as f:
    data = json.load(f)
    best = data.get('best_experiment', {})
    if best:
        print(f'최고 성능: {best.get(\\\"experiment_name\\\", \\\"N/A\\\")}')
        print(f'ROUGE Score: {best.get(\\\"best_rouge_combined_f1\\\", 0):.4f}')
    else:
        print('성공한 실험이 없습니다')
\""
else
    echo "❌ 일부 실험이 실패했습니다"
    echo "로그를 확인하여 문제를 해결하세요: logs/auto_experiments.log"
fi

exit $RESULT
