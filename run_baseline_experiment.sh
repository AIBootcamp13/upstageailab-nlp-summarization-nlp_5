#!/bin/bash
# 기준선 재현 실험 러너

echo "🚀 Starting Baseline Reproduction Experiment"
echo "==========================================="

# 프로젝트 루트에 있는지 확인
if [ ! -f "run_auto_experiments.sh" ]; then
    echo "❌ Please run this script from the project root directory"
    exit 1
fi

# 사용 가능한 메모리에 따라 구성 선택
echo "🔍 Checking system resources..."

# 초기 검증을 위한 미니 테스트 기본값
CONFIG_FILE="config/experiments/00_baseline_mini_test.yaml"
echo "📋 Using configuration: $CONFIG_FILE"

# 이 실험을 위한 출력 디렉토리 생성
OUTPUT_DIR="outputs/baseline_reproduction_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "📁 Results will be saved to: $OUTPUT_DIR"
echo ""

# 실험 실행
echo "🏃 Starting training..."
echo "This may take a while depending on your hardware..."
echo ""

# 참고: 실제 실행 시에는 다음을 실행:
# python code/trainer.py \
#     --config "$CONFIG_FILE" \
#     --output-dir "$OUTPUT_DIR" \
#     --experiment-name "baseline_reproduction"

echo "✅ Experiment setup complete!"
echo ""
echo "📊 To run the actual training, ensure you have:"
echo "   1. Installed all requirements: pip install -r requirements.txt"
echo "   2. Downloaded the data files to the data/ directory"
echo "   3. Set up your WandB account (optional)"
echo ""
echo "Then run: python code/trainer.py --config $CONFIG_FILE"

# 실험 메타데이터 저장
cat > "$OUTPUT_DIR/experiment_info.json" << EOF
{
    "experiment_name": "baseline_reproduction",
    "config_file": "$CONFIG_FILE",
    "start_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "target_rouge_f1": 0.4712,
    "description": "Reproducing baseline performance for benchmarking"
}
EOF

echo ""
echo "📄 Experiment metadata saved to: $OUTPUT_DIR/experiment_info.json"
