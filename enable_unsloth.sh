#!/bin/bash

# Ubuntu 환경에서 unsloth 활성화 스크립트

echo "=== Ubuntu 환경용 unsloth 활성화 스크립트 ==="
echo

# 현재 환경 확인
if [ "$(uname)" != "Linux" ]; then
    echo "⚠️  이 스크립트는 Ubuntu/Linux 환경용입니다."
    echo "현재 환경: $(uname)"
    echo "macOS에서는 수동으로 config.yaml에서 use_unsloth를 true로 변경해주세요."
    exit 1
fi

echo "✅ Linux 환경 확인됨"

# 가상환경 활성화 확인
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f ".venv/bin/activate" ]; then
        echo "🔄 가상환경 활성화 중..."
        source .venv/bin/activate
    else
        echo "❌ 가상환경을 찾을 수 없습니다. setup_env.sh를 먼저 실행해주세요."
        exit 1
    fi
fi

echo "✅ 가상환경 확인됨: $VIRTUAL_ENV"

# unsloth 설치
echo "🚀 unsloth 설치 중..."
pip install unsloth

if [ $? -eq 0 ]; then
    echo "✅ unsloth 설치 완료!"
else
    echo "❌ unsloth 설치 실패. 다음을 확인해주세요:"
    echo "  - CUDA 11.8+ 설치 여부"
    echo "  - PyTorch 2.6.0 설치 여부"
    echo "  - 충분한 디스크 공간"
    exit 1
fi

# config.yaml에서 use_unsloth 활성화
echo "📝 config.yaml에서 unsloth 활성화 중..."

# 백업 생성
cp config.yaml config.yaml.pre_unsloth_backup

# use_unsloth를 true로 변경
sed -i 's/use_unsloth: false/use_unsloth: true/g' config.yaml

echo "✅ config.yaml 업데이트 완료"

# code/config.yaml도 동일하게 업데이트
cp config.yaml code/config.yaml

echo "✅ code/config.yaml 동기화 완료"

# 설치 검증
echo "🔍 설치 검증 중..."
python -c "
try:
    from unsloth import FastLanguageModel
    print('✅ unsloth 정상 설치됨')
except ImportError as e:
    print(f'❌ unsloth import 실패: {e}')
    exit(1)

import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    
use_unsloth = config.get('qlora', {}).get('use_unsloth', False)
if use_unsloth:
    print('✅ config.yaml에서 unsloth 활성화됨')
else:
    print('❌ config.yaml에서 unsloth 활성화 실패')
    exit(1)
"

if [ $? -eq 0 ]; then
    echo
    echo "🎉 unsloth 활성화 완료!"
    echo
    echo "📊 예상 효과:"
    echo "  - 메모리 사용량: 75% 감소"
    echo "  - 학습 속도: 2-3배 향상"
    echo "  - GPU 메모리 효율성: 극대화"
    echo
    echo "🚀 이제 trainer.py를 실행하면 unsloth가 자동으로 사용됩니다!"
else
    echo "❌ 활성화 검증 실패"
    exit 1
fi
