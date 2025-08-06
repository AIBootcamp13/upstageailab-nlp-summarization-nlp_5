#!/bin/bash

# 🏆 Ultimate Strategy Execution Script
# 2시간 내 1위 달성 전략

echo "🚀 ULTIMATE STRATEGY STARTED - TARGET: #1 RANK"
echo "📅 Start Time: $(date)"
echo "🎯 Target Score: 52.0+"

LOG_DIR="./logs/ultimate_strategy"
mkdir -p "$LOG_DIR"

# Phase 1: Quick Boost (15분) - 즉시 점수 향상
echo "⚡ Phase 1: Quick Boost Starting..."
nohup python src/main_base_modified.py --config config_quick_boost.yaml > "$LOG_DIR/quick_boost.log" 2>&1 &
QUICK_PID=$!
echo "Quick Boost PID: $QUICK_PID"

# Phase 2: Ultimate Training (80분) - 최고 성능 훈련  
echo "🔥 Phase 2: Ultimate Training Starting..."
nohup python src/main_base_modified.py --config config_ultimate_lyj.yaml > "$LOG_DIR/ultimate_training.log" 2>&1 &
ULTIMATE_PID=$!
echo "Ultimate Training PID: $ULTIMATE_PID"

# 진행 상황 모니터링
monitor_progress() {
    while true; do
        echo "📊 $(date): Monitoring progress..."
        
        # Quick boost 확인
        if kill -0 $QUICK_PID 2>/dev/null; then
            echo "  ⚡ Quick Boost: Running"
        else
            echo "  ⚡ Quick Boost: Completed"
            if [ -f "./outputs/exp_quick_boost_lyj/submission_quick.csv/result1" ]; then
                echo "  ✅ Quick Boost Results Ready"
            fi
        fi
        
        # Ultimate training 확인
        if kill -0 $ULTIMATE_PID 2>/dev/null; then
            echo "  🔥 Ultimate Training: Running"
            # 최근 로그 확인
            if [ -f "$LOG_DIR/ultimate_training.log" ]; then
                RECENT_LOG=$(tail -3 "$LOG_DIR/ultimate_training.log" | grep -E "epoch|eval_rouge")
                if [ ! -z "$RECENT_LOG" ]; then
                    echo "  📈 Latest: $RECENT_LOG"
                fi
            fi
        else
            echo "  🔥 Ultimate Training: Completed"
            if [ -f "./outputs/exp_ultimate_lyj/submission_ultimate.csv/result1" ]; then
                echo "  ✅ Ultimate Results Ready"
            fi
        fi
        
        # GPU 상태
        GPU_INFO=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits)
        echo "  🔧 GPU: ${GPU_INFO}% utilization"
        
        sleep 60
        
        # 모든 작업 완료 시 종료
        if ! kill -0 $QUICK_PID 2>/dev/null && ! kill -0 $ULTIMATE_PID 2>/dev/null; then
            break
        fi
    done
}

# 백그라운드 모니터링 시작
monitor_progress &
MONITOR_PID=$!

# Phase 3: 대기 및 앙상블 준비
wait $QUICK_PID
echo "⚡ Quick Boost Completed!"

# Quick boost 결과 즉시 제출 가능하도록 복사
if [ -f "./outputs/exp_quick_boost_lyj/submission_quick.csv/result1" ]; then
    cp "./outputs/exp_quick_boost_lyj/submission_quick.csv/result1" "./outputs/submission_quick_ready.csv"
    echo "📋 Quick submission ready: ./outputs/submission_quick_ready.csv"
fi

# Ultimate training 완료 대기
wait $ULTIMATE_PID
echo "🔥 Ultimate Training Completed!"

# Phase 4: Ensemble (최종 5분)
echo "🎭 Phase 3: Final Ensemble..."
python src/ensemble_ultimate.py > "$LOG_DIR/ensemble.log" 2>&1

# 결과 정리
echo "🎉 ULTIMATE STRATEGY COMPLETED!"
echo "📅 End Time: $(date)"

# 생성된 모든 결과 파일 표시
echo "📁 Generated Results:"
find ./outputs -name "*.csv" -newer "$LOG_DIR" -exec ls -la {} \;

# 모니터링 프로세스 종료
kill $MONITOR_PID 2>/dev/null

echo "🏆 Ready for submission! Choose the best result file."
