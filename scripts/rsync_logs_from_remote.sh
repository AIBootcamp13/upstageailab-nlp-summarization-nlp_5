#!/bin/bash

# 원격 서버에서 로그 파일을 맥으로 이동하는 스크립트
# rsync를 사용하여 이미 있는 파일은 건너뛰고, 전송 후 원격 서버에서 삭제

# 설정
REMOTE_USER="root"
REMOTE_HOST="aistages"
REMOTE_PORT="32145"
REMOTE_BASE="/data/ephemeral/home/upstageailab-nlp-summarization-nlp_5"
LOCAL_BASE="/Users/jayden/Developer/Projects/upstageailab-nlp-summarization-nlp_5/nlp-sum-lyj"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🔄 원격 서버에서 로그 파일 동기화 시작${NC}"
echo -e "${YELLOW}원격: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/logs/${NC}"
echo -e "${YELLOW}로컬: ${LOCAL_BASE}/logs/${NC}"

# logs 디렉토리가 로컬에 없으면 생성
mkdir -p "${LOCAL_BASE}/logs"

# rsync 실행
# --ignore-existing: 이미 존재하는 파일은 건너뜀
# --remove-source-files: 전송 성공 후 원본 파일 삭제
# -avz: archive mode, verbose, compress
# --progress: 진행 상황 표시
rsync -avz \
    --ignore-existing \
    --remove-source-files \
    --progress \
    --stats \
    -e "ssh -p ${REMOTE_PORT}" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/logs/" \
    "${LOCAL_BASE}/logs/"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ 로그 파일 동기화 완료!${NC}"
    
    # 빈 디렉토리 정리 (선택사항)
    echo -e "${YELLOW}🧹 원격 서버의 빈 디렉토리 정리 중...${NC}"
    ssh -p ${REMOTE_PORT} "${REMOTE_USER}@${REMOTE_HOST}" "find ${REMOTE_BASE}/logs -type d -empty -delete 2>/dev/null"
    
    # 로컬에 동기화된 파일 목록 표시
    echo -e "${GREEN}📋 동기화된 파일:${NC}"
    find "${LOCAL_BASE}/logs" -type f -newer /tmp/rsync_timestamp 2>/dev/null | head -20
    
else
    echo -e "${RED}❌ 동기화 실패!${NC}"
    exit 1
fi

# 타임스탬프 업데이트
touch /tmp/rsync_timestamp

echo -e "${GREEN}🎉 작업 완료!${NC}"
