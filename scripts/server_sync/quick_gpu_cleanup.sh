#!/bin/bash

# 🔥 빠른 GPU 메모리 정리 스크립트
# 실험 시작 전 또는 메모리 부족 시 즉시 사용

set -euo pipefail

# PATH 설정
export PATH="/usr/bin:/usr/local/bin:$PATH"

# nvidia-smi 절대 경로 설정
NVIDIA_SMI="/usr/bin/nvidia-smi"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BLUE='\033[0;34m'
NC='\033[0m'

# 전역 변수 (정리 전후 비교용)
MEMORY_BEFORE=0
MEMORY_AFTER=0

# GPU 메모리 상태 가져오기 함수 (숫자만 반환)
get_gpu_memory() {
    local memory_used
    memory_used=$($NVIDIA_SMI --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | xargs | tr -d ',')
    
    if [ -n "$memory_used" ]; then
        # 소수점 값을 정수로 변환
        memory_used=$(echo "$memory_used" | cut -d'.' -f1)
        echo "$memory_used"
    else
        echo "0"
    fi
}

# GPU 상태 표시 함수 (간단 버전)
show_gpu_status() {
    local title="$1"
    echo -e "\n${BLUE}📊 $title${NC}"
    
    $NVIDIA_SMI --query-gpu=memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | while IFS=',' read -r used total temp; do
        used=$(echo "$used" | xargs)
        total=$(echo "$total" | xargs)
        temp=$(echo "$temp" | xargs)
        percent=$((used * 100 / total))
        free=$((total - used))
        
        echo "  🗜️  메모리: ${used}MB / ${total}MB (${percent}%)"
        echo "  🔓 여유: ${free}MB"
        echo "  🌡️  온도: ${temp}°C"
        
        # 상태 평가 (간단)
        if [ "$used" -lt 5000 ]; then
            echo -e "  ${GREEN}✅ 양호: 5GB 미만${NC}"
        elif [ "$used" -lt 10000 ]; then
            echo -e "  ${YELLOW}⚠️  보통: 10GB 미만${NC}"
        elif [ "$used" -lt 15000 ]; then
            echo -e "  ${YELLOW}⚠️  주의: 15GB 미만${NC}"
        else
            echo -e "  ${RED}❌ 위험: 15GB 이상${NC}"
        fi
    done
}

# 정리 결과 비교 표시 함수
show_cleanup_results() {
    local memory_before="$1"
    local memory_after="$2"
    
    echo -e "\n${CYAN}📈 정리 결과 요약${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    local memory_cleaned=$((memory_before - memory_after))
    local memory_cleaned_gb=$((memory_cleaned / 1024))
    local memory_cleaned_percent=0
    
    if [ "$memory_before" -gt 0 ]; then
        memory_cleaned_percent=$((memory_cleaned * 100 / memory_before))
    fi
    
    echo "  📊 정리 전: ${memory_before}MB"
    echo "  📊 정리 후: ${memory_after}MB"
    
    if [ "$memory_cleaned" -gt 0 ]; then
        echo -e "  ${GREEN}✅ 정리량: ${memory_cleaned}MB (${memory_cleaned_gb}GB, ${memory_cleaned_percent}% 감소)${NC}"
        
        if [ "$memory_cleaned" -gt 3000 ]; then
            echo -e "  ${GREEN}🎉 우수한 정리! (3GB 이상)${NC}"
        elif [ "$memory_cleaned" -gt 1000 ]; then
            echo -e "  ${CYAN}👍 좋은 정리! (1GB 이상)${NC}"
        elif [ "$memory_cleaned" -gt 500 ]; then
            echo -e "  ${YELLOW}👌 적당한 정리 (500MB 이상)${NC}"
        fi
    elif [ "$memory_cleaned" -eq 0 ]; then
        echo -e "  ${YELLOW}ℹ️  변화 없음${NC}"
    else
        echo -e "  ${RED}⚠️  메모리 증가: $((memory_after - memory_before))MB${NC}"
    fi
}

echo -e "${CYAN}🔥 빠른 GPU 메모리 정리${NC}"

# 1. 정리 전 상태 확인 및 저장
MEMORY_BEFORE=$(get_gpu_memory)
show_gpu_status "정리 전 GPU 상태"

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

# 4. 정리 후 상태 확인 및 저장
sleep 2  # 정리 효과 반영 대기
MEMORY_AFTER=$(get_gpu_memory)
show_gpu_status "정리 후 GPU 상태"

# 5. 정리 결과 비교 표시
show_cleanup_results "$MEMORY_BEFORE" "$MEMORY_AFTER"

echo -e "\n${CYAN}🎯 빠른 정리 완료!${NC}"
