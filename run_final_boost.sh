#!/bin/bash

# 🚀 Final Boost 실행 스크립트
# 기존 best 모델 활용한 최적화된 추론

echo "🚀 Final Boost Inference - Starting..."
echo "📅 $(date)"
echo "🔧 Using existing best model from exp_optimized_lyj"

# 로그 파일 설정
LOG_FILE="final_boost_$(date +%Y%m%d_%H%M%S).log"

# Python 환경 확인
echo "🐍 Python version: $(python --version)"
echo "🔥 PyTorch version: $(python -c 'import torch; print(torch.__version__)')"

# 추론 실행
echo "🔮 Starting final boost inference..."
nohup python src/inference_final_boost.py --config config_final_boost.yaml > "$LOG_FILE" 2>&1 &

# PID 저장
echo $! > final_boost.pid

echo "✅ Final boost inference started!"
echo "📋 Process ID: $(cat final_boost.pid)"
echo "📄 Log file: $LOG_FILE"
echo "⏱️  Estimated time: 15-20 minutes"

# 진행 상황 모니터링 함수
monitor_progress() {
    echo "📊 Monitoring progress..."
    while kill -0 $(cat final_boost.pid) 2>/dev/null; do
        echo "⏳ $(date): Still running..."
        sleep 30
    done
    echo "✅ $(date): Completed!"
    
    # 결과 확인
    if [ -f "outputs/exp_final_boost_lyj/submission_final.csv/result1" ]; then
        echo "🎉 Results generated successfully!"
        echo "📁 File location: outputs/exp_final_boost_lyj/submission_final.csv/result1"
        echo "📊 File size: $(du -h outputs/exp_final_boost_lyj/submission_final.csv/result1)"
        echo "📈 Sample results:"
        head -5 outputs/exp_final_boost_lyj/submission_final.csv/result1
    else
        echo "❌ Results file not found"
    fi
}

# 백그라운드에서 모니터링 실행
monitor_progress &

echo "🚀 Use 'tail -f $LOG_FILE' to monitor progress"
echo "🛑 Use 'kill $(cat final_boost.pid)' to stop if needed"
