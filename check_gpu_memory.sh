#!/bin/bash
# GPU 메모리 모니터링 및 배치 크기 테스트 스크립트

echo "🔍 GPU 메모리 및 배치 크기 최적화 테스트"
echo "======================================="

# GPU 정보 확인
echo "📊 현재 GPU 상태:"
nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used --format=csv,nounits,noheader

echo ""
echo "🧪 모델별 권장 배치 크기:"

# GPU 메모리 크기 확인
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,nounits,noheader | head -1)

echo "GPU 메모리: ${GPU_MEMORY}MB"

if [ "$GPU_MEMORY" -lt 10000 ]; then
    echo "⚠️  소형 GPU (< 10GB) 감지"
    echo "mT5 권장 설정:"
    echo "  - per_device_train_batch_size: 1"
    echo "  - per_device_eval_batch_size: 1" 
    echo "  - gradient_accumulation_steps: 8"
    echo ""
    echo "eenzeenee 권장 설정:"
    echo "  - per_device_train_batch_size: 2"
    echo "  - per_device_eval_batch_size: 4"
    echo "  - gradient_accumulation_steps: 4"
    
elif [ "$GPU_MEMORY" -lt 20000 ]; then
    echo "✅ 중형 GPU (10-20GB) 감지 - V100 등"
    echo "mT5 권장 설정:"
    echo "  - per_device_train_batch_size: 1"
    echo "  - per_device_eval_batch_size: 2"
    echo "  - gradient_accumulation_steps: 4"
    echo ""
    echo "eenzeenee 권장 설정:"
    echo "  - per_device_train_batch_size: 4"
    echo "  - per_device_eval_batch_size: 8"
    echo "  - gradient_accumulation_steps: 2"
    
else
    echo "🚀 대형 GPU (20GB+) 감지 - A100 등"
    echo "mT5 권장 설정:"
    echo "  - per_device_train_batch_size: 4"
    echo "  - per_device_eval_batch_size: 8"
    echo "  - gradient_accumulation_steps: 2"
    echo ""
    echo "eenzeenee 권장 설정:"
    echo "  - per_device_train_batch_size: 8"
    echo "  - per_device_eval_batch_size: 16"
    echo "  - gradient_accumulation_steps: 1"
fi

echo ""
echo "🔧 배치 크기 테스트 방법:"
echo "1. 작은 배치부터 시작"
echo "2. CUDA out of memory 오류까지 점진적 증가"
echo "3. 안전 마진 20% 적용"

echo ""
echo "📝 테스트 명령어 예시:"
echo "# eenzeenee 모델 테스트"
echo "uv run python code/trainer.py --config config.yaml --config-section eenzeenee --max_steps 10"
echo ""
echo "# mT5 모델 테스트"  
echo "uv run python code/trainer.py --config config.yaml --config-section xlsum_mt5 --max_steps 10"

echo ""
echo "⚠️  메모리 모니터링: watch -n 1 nvidia-smi"
