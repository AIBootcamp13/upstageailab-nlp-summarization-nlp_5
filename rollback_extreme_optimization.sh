#!/bin/bash
# RTX 3090 극한 최적화 롤백 스크립트
# 사용법: bash rollback_extreme_optimization.sh

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🔄 RTX 3090 극한 최적화 롤백 시작${NC}"
echo "========================================"

# 백업 디렉토리 찾기
BACKUP_DIR=$(find config/ -name "experiments_backup_*" -type d | sort -r | head -1)

if [ -z "$BACKUP_DIR" ]; then
    echo -e "${RED}❌ 백업 디렉토리를 찾을 수 없습니다.${NC}"
    echo "config/ 디렉토리에서 experiments_backup_* 폴더를 확인하세요."
    exit 1
fi

echo -e "${BLUE}📂 발견된 백업: $BACKUP_DIR${NC}"

# 현재 experiments 디렉토리를 rollback_temp로 임시 백업
if [ -d "config/experiments" ]; then
    echo -e "${YELLOW}🔄 현재 설정을 임시 백업 중...${NC}"
    mv config/experiments config/experiments_rollback_temp_$(date +%Y%m%d_%H%M%S)
fi

# 백업에서 복원
echo -e "${YELLOW}🔄 백업에서 복원 중...${NC}"
cp -r "$BACKUP_DIR" config/experiments

# 복원 확인
if [ -d "config/experiments" ]; then
    echo -e "${GREEN}✅ 복원 성공!${NC}"
    echo -e "${BLUE}📊 복원된 파일 목록:${NC}"
    ls -la config/experiments/*.yaml | head -10
else
    echo -e "${RED}❌ 복원 실패${NC}"
    exit 1
fi

# GPU 메모리 상태 확인
echo -e "${BLUE}📊 현재 GPU 상태:${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader
else
    echo -e "${YELLOW}⚠️  nvidia-smi를 찾을 수 없습니다.${NC}"
fi

echo
echo -e "${GREEN}🎉 RTX 3090 극한 최적화 롤백 완료!${NC}"
echo -e "   백업에서 실험 설정이 성공적으로 복원되었습니다."
echo -e "   기존 극한 최적화 설정은 experiments_rollback_temp_* 폴더에 임시 저장되었습니다."
echo
echo -e "${YELLOW}💡 다음 단계:${NC}"
echo -e "   1. 실험 설정 확인: ls -la config/experiments/"
echo -e "   2. 테스트 실행: bash run_main_7_experiments.sh -1"
echo -e "   3. 임시 백업 정리: rm -rf config/experiments_rollback_temp_*"
