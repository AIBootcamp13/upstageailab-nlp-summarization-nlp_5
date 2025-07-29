#!/bin/bash
# 실험 config 파일들에 누락된 섹션 추가

echo "🔧 실험 config 파일 수정 중..."

# config/experiments 디렉토리의 모든 yaml 파일 처리
for config_file in config/experiments/*.yaml; do
    if [ -f "$config_file" ]; then
        echo "📝 처리 중: $config_file"
        python fix_config.py "$config_file"
    fi
done

echo "✅ 모든 config 파일 수정 완료!"
