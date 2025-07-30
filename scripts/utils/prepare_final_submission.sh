#!/bin/bash
# 최종 제출 준비 스크립트

echo "========================================="
echo "최종 제출 준비 프로세스"
echo "========================================="
echo ""

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # 색상 없음

# 환경 확인
echo "1. 환경 확인 중..."
if ! command -v python &> /dev/null; then
    echo -e "${RED}❌ Python이 설치되어 있지 않습니다.${NC}"
    exit 1
fi

if ! python -c "import torch; import transformers" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  필요한 패키지가 설치되어 있지 않습니다. requirements.txt를 확인하세요.${NC}"
fi

# 디렉토리 생성
echo "2. 디렉토리 구조 생성 중..."
mkdir -p final_submission/backup
mkdir -p final_submission/model
mkdir -p final_submission/logs

# 최고 성능 모델 찾기
echo "3. 최고 성능 모델 확인 중..."
MODEL_FOUND=false

# Solar 앙상블 결과 확인
if [ -f "outputs/solar_ensemble/dynamic_weights/test_results.csv" ]; then
    echo -e "${GREEN}✓ Solar API 앙상블 결과 발견${NC}"
    MODEL_FOUND=true
    USE_ENSEMBLE="--use_ensemble"
else
    echo -e "${YELLOW}⚠️  Solar 앙상블 결과가 없습니다. Fine-tuned 모델 확인 중...${NC}"
    USE_ENSEMBLE=""
    
    # Fine-tuned 모델 확인
    for model_dir in \
        "outputs/phase2_results/10c_all_optimizations" \
        "outputs/phase2_results/10b_phase1_plus_backtrans" \
        "outputs/phase2_results/10a_phase1_plus_token_weight" \
        "models/baseline"
    do
        if [ -d "$model_dir" ]; then
            echo -e "${GREEN}✓ 모델 발견: $model_dir${NC}"
            MODEL_FOUND=true
            break
        fi
    done
fi

if [ "$MODEL_FOUND" = false ]; then
    echo -e "${RED}❌ 학습된 모델을 찾을 수 없습니다.${NC}"
    echo "먼저 모델을 학습하세요: ./run_phase2_experiments.sh"
    exit 1
fi

# 추론 실행
echo ""
echo "4. 테스트 세트 추론 실행 중..."
echo "이 작업은 30-40분 정도 소요될 수 있습니다..."

python final_submission/run_final_inference.py \
    --test_file test.csv \
    --output_dir final_submission \
    --batch_size 16 \
    $USE_ENSEMBLE

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ 추론 실행 중 오류가 발생했습니다.${NC}"
    exit 1
fi

# 제출 파일 검증
echo ""
echo "5. 제출 파일 검증 중..."
if [ -f "sample_submission.csv" ]; then
    python scripts/validate_submission.py \
        --submission final_submission/submission.csv \
        --sample sample_submission.csv \
        --output final_submission/validation_report.json
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ 제출 파일 검증 실패${NC}"
        echo "validation_report.json을 확인하세요."
        exit 1
    fi
else
    echo -e "${YELLOW}⚠️  sample_submission.csv가 없어 검증을 건너뜁니다.${NC}"
fi

# 백업 생성
echo ""
echo "6. 백업 생성 중..."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="final_submission/backup/${TIMESTAMP}"
mkdir -p "$BACKUP_DIR"

# 제출 파일 백업
cp final_submission/submission.csv "$BACKUP_DIR/"

# 설정 파일 백업
cp -r config/experiments "$BACKUP_DIR/configs"

# 코드 백업 (선택적)
echo "주요 코드 백업 중..."
mkdir -p "$BACKUP_DIR/code"
cp -r code/*.py "$BACKUP_DIR/code/"
cp -r code/ensemble "$BACKUP_DIR/code/"
cp -r code/postprocessing "$BACKUP_DIR/code/"

echo -e "${GREEN}✓ 백업 완료: $BACKUP_DIR${NC}"

# 최종 체크리스트
echo ""
echo "========================================="
echo "최종 체크리스트"
echo "========================================="
echo ""

# 파일 존재 확인
echo "📋 파일 확인:"
[ -f "final_submission/submission.csv" ] && echo -e "${GREEN}✓ submission.csv${NC}" || echo -e "${RED}✗ submission.csv${NC}"
[ -f "final_submission/final_report.md" ] && echo -e "${GREEN}✓ final_report.md${NC}" || echo -e "${RED}✗ final_report.md${NC}"
[ -f "final_submission/validation_report.json" ] && echo -e "${GREEN}✓ validation_report.json${NC}" || echo -e "${RED}✗ validation_report.json${NC}"

# 통계 출력
echo ""
echo "📊 제출 파일 통계:"
if [ -f "final_submission/submission.csv" ]; then
    LINE_COUNT=$(wc -l < final_submission/submission.csv)
    FILE_SIZE=$(du -h final_submission/submission.csv | cut -f1)
    echo "  - 행 수: $LINE_COUNT"
    echo "  - 파일 크기: $FILE_SIZE"
fi

# 최종 안내
echo ""
echo "========================================="
echo "✅ 최종 제출 준비 완료!"
echo "========================================="
echo ""
echo "제출 파일 위치: final_submission/submission.csv"
echo ""
echo "제출 전 확인사항:"
echo "1. validation_report.json에서 에러가 없는지 확인"
echo "2. 특수 토큰이 적절히 포함되어 있는지 확인"
echo "3. 파일 크기가 적절한지 확인 (일반적으로 < 10MB)"
echo ""
echo "행운을 빕니다! 🍀"
