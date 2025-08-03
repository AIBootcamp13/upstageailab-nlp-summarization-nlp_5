#!/bin/bash

# 🔥 빠른 GPU 메모리 정리 스크립트
# 실험 시작 전 또는 메모리 부족 시 즉시 사용

set -euo pipefail

# PATH 설정
export PATH="/usr/bin:/usr/local/bin:$PATH"

# nvidia-smi 경로 확인
NVIDIA_SMI="/usr/bin/nvidia-smi"
if [ ! -x "$NVIDIA_SMI" ]; then
    NVIDIA_SMI=$(which nvidia-smi 2>/dev/null || echo "nvidia-smi")
fi

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}🔥 빠른 GPU 메모리 정리${NC}"

# 1. 현재 상태 확인
echo "현재 GPU 상태:"
$NVIDIA_SMI --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | while IFS=',' read -r used total; do
    used=$(echo "$used" | xargs)
    total=$(echo "$total" | xargs)
    percent=$((used * 100 / total))
    echo "  사용량: ${used}MB / ${total}MB (${percent}%)"
done

# 2. PyTorch 캐시 즉시 정리
echo -e "\n${YELLOW}🧹 PyTorch 캐시 정리 중...${NC}"
python3 -c "
import torch
import gc
import os

try:
    if torch.cuda.is_available():
        # CUDA 캐시 정리
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print('✅ CUDA 캐시 정리')
        
        # 메모리 상태 출력
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            print(f'GPU {i}: 할당 {allocated:.1f}GB, 캐시 {cached:.1f}GB')
    
    # 가비지 컬렉션
    gc.collect()
    print('✅ 가비지 컬렉션 완료')
    
except ImportError:
    print('⚠️  PyTorch 없음')
except Exception as e:
    print(f'❌ 오류: {e}')
" 2>/dev/null

# 3. 시스템 메모리 정리 시도
echo -e "\n${YELLOW}🗑️  시스템 캐시 정리 시도...${NC}"
sync 2>/dev/null || true
if [ -w /proc/sys/vm/drop_caches ] 2>/dev/null; then
    echo 3 > /proc/sys/vm/drop_caches
    echo "✅ 시스템 캐시 정리 완료"
else
    echo "ℹ️  시스템 캐시 정리 권한 없음 (정상)"
fi

# 4. 최종 상태 확인
echo -e "\n${GREEN}📊 정리 후 GPU 상태:${NC}"
$NVIDIA_SMI --query-gpu=memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | while IFS=',' read -r used total temp; do
    used=$(echo "$used" | xargs)
    total=$(echo "$total" | xargs)
    temp=$(echo "$temp" | xargs)
    percent=$((used * 100 / total))
    free=$((total - used))
    
    echo "  🗜️  메모리: ${used}MB / ${total}MB (${percent}%)"
    echo "  🔓 여유: ${free}MB"
    echo "  🌡️  온도: ${temp}°C"
    
    if [ "$used" -lt 5000 ]; then
        echo -e "  ${GREEN}✅ 정리 성공: 5GB 미만${NC}"
    elif [ "$used" -lt 10000 ]; then
        echo -e "  ${YELLOW}⚠️  보통: 10GB 미만${NC}"
    else
        echo -e "  ${RED}❌ 주의: 10GB 이상${NC}"
    fi
done

echo -e "\n${CYAN}🎯 정리 완료!${NC}"
