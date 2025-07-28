#!/bin/bash
# Linux 환경에서 모든 모델 설정을 Unsloth 활성화로 변경하는 스크립트

set -e

echo "🚀 Linux 환경용 Unsloth 전체 활성화 스크립트"
echo "============================================"

# 현재 OS 확인
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "⚠️  macOS에서는 Unsloth 사용을 권장하지 않습니다."
    echo "계속 진행하시겠습니까? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "❌ 취소되었습니다."
        exit 0
    fi
fi

echo "📋 현재 Unsloth 설정 상태 확인 중..."

# 설정 파일들 목록
CONFIG_FILES=(
    "config.yaml"
    "config/model_configs/bart_base.yaml"
    "config/model_configs/t5_base.yaml"
    "config/model_configs/mt5_base.yaml"
    "config/model_configs/flan_t5_base.yaml"
    "config/model_configs/kogpt2.yaml"
)

# 백업 디렉토리 생성
BACKUP_DIR="config_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "💾 설정 파일 백업 중: $BACKUP_DIR"

# 모든 설정 파일 백업
for file in "${CONFIG_FILES[@]}"; do
    if [ -f "$file" ]; then
        cp "$file" "$BACKUP_DIR/"
        echo "  ✅ 백업됨: $file"
    else
        echo "  ⚠️  파일 없음: $file"
    fi
done

echo ""
echo "🔧 Unsloth 활성화 중..."

# 1. 메인 config.yaml에서 eenzeenee와 xlsum_mt5 섹션의 use_unsloth를 true로 변경
if [ -f "config.yaml" ]; then
    echo "📝 config.yaml 업데이트 중..."
    
    # eenzeenee 섹션
    sed -i.bak 's/^  qlora:$/&\
    use_unsloth: true  # Linux 환경에서 활성화/' config.yaml
    
    # 이미 존재하는 use_unsloth: false를 true로 변경
    sed -i 's/use_unsloth: false  # macOS 환경/use_unsloth: true  # Linux 환경에서 활성화/g' config.yaml
    
    # 백업 파일 제거
    rm -f config.yaml.bak
    
    echo "  ✅ config.yaml 업데이트 완료"
fi

# 2. 개별 모델 설정 파일들 업데이트
for file in "${CONFIG_FILES[@]}"; do
    if [ -f "$file" ] && [[ "$file" != "config.yaml" ]]; then
        echo "📝 $file 업데이트 중..."
        
        # use_unsloth: false를 true로 변경
        sed -i.bak 's/use_unsloth: false/use_unsloth: true/g' "$file"
        
        # use_qlora: false를 true로 변경 (Unsloth는 QLoRA와 함께 사용)
        sed -i 's/use_qlora: false/use_qlora: true/g' "$file"
        
        # 백업 파일 제거
        rm -f "${file}.bak"
        
        echo "  ✅ $file 업데이트 완료"
    fi
done

# 3. kobart_unsloth.yaml은 이미 활성화되어 있으므로 확인만
if [ -f "config/model_configs/kobart_unsloth.yaml" ]; then
    echo "📝 kobart_unsloth.yaml는 이미 Unsloth가 활성화되어 있습니다."
fi

echo ""
echo "🔍 변경 사항 확인 중..."

# 변경된 내용 확인
echo "변경된 파일들의 use_unsloth 설정:"
for file in "${CONFIG_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "📄 $file:"
        grep -n "use_unsloth:" "$file" | head -3 || echo "  (use_unsloth 설정 없음)"
    fi
done

echo ""
echo "📦 requirements.txt에서 unsloth 주석 제거 중..."

# requirements.txt에서 unsloth 주석 제거
if [ -f "requirements.txt" ]; then
    # 백업
    cp requirements.txt "$BACKUP_DIR/"
    
    # 주석 제거
    sed -i.bak 's/# unsloth  # QLoRA support for memory efficiency (macOS에서 컴파일 이슈로 일시 비활성화)/unsloth  # QLoRA support for memory efficiency - Linux에서 활성화됨/g' requirements.txt
    
    rm -f requirements.txt.bak
    echo "  ✅ requirements.txt 업데이트 완료"
fi

echo ""
echo "🎯 권장 사항: Unsloth 설치"
echo "========================="
echo ""
echo "다음 명령어로 Unsloth를 설치하세요:"
echo ""
echo "# PyTorch 2.4+ 설치 (필수)"
echo "pip install torch>=2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
echo ""
echo "# Unsloth 설치"
echo "pip install \"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\""
echo ""
echo "# 추가 의존성"
echo "pip install xformers trl peft accelerate bitsandbytes"
echo ""
echo "또는 제공된 스크립트 사용:"
echo "./install_unsloth.sh"

echo ""
echo "🎉 Unsloth 전체 활성화 완료!"
echo "============================="
echo ""
echo "📊 예상 성능 향상:"
echo "  • 메모리 사용량: 75% 감소"
echo "  • 학습 속도: 2-3배 향상"
echo "  • GPU 효율성: 극대화"
echo ""
echo "🔄 복원 방법:"
echo "  백업 파일들이 $BACKUP_DIR 에 저장되어 있습니다."
echo "  복원하려면: cp $BACKUP_DIR/* ./ 실행"
echo ""
echo "📋 다음 단계:"
echo "  1. Unsloth 설치 (위 명령어 참조)"
echo "  2. 빠른 테스트: python quick_test.py --model-section eenzeenee"
echo "  3. 전체 실험: ./run_eenzeenee_experiment.sh"
echo ""
echo "🏁 모든 실험에서 Unsloth가 자동으로 사용됩니다!"
