#!/bin/bash
# 환경 및 파일 확인 스크립트

echo "🔍 환경 확인 스크립트"
echo "===================="
echo ""

echo "📍 현재 디렉토리:"
pwd
echo ""

echo "📂 디렉토리 구조:"
echo "- 프로젝트 루트:"
ls -la | head -10
echo ""

echo "- config 디렉토리:"
if [ -d "config" ]; then
    ls -la config/
else
    echo "❌ config 디렉토리가 없습니다!"
fi
echo ""

echo "- config/experiments 디렉토리:"
if [ -d "config/experiments" ]; then
    ls -la config/experiments/ | grep "test_.*yaml"
else
    echo "❌ config/experiments 디렉토리가 없습니다!"
fi
echo ""

echo "📄 테스트 파일 확인:"
for file in test_01_mt5_xlsum_1epoch.yaml test_02_eenzeenee_1epoch.yaml test_03_kobart_1epoch.yaml test_04_high_lr_1epoch.yaml test_05_batch_opt_1epoch.yaml; do
    if [ -f "config/experiments/$file" ]; then
        echo "✅ $file 존재"
    else
        echo "❌ $file 없음"
    fi
done
echo ""

echo "🔍 파일 검색:"
find . -name "test_01_mt5_xlsum_1epoch.yaml" 2>/dev/null | head -5
echo ""

echo "📊 Git 상태:"
git status --short
echo ""

echo "🌿 Git 브랜치:"
git branch
echo ""

echo "📍 Python 경로 테스트:"
python -c "import os; print(f'작업 디렉토리: {os.getcwd()}')"
python -c "from pathlib import Path; p = Path('config/experiments/test_01_mt5_xlsum_1epoch.yaml'); print(f'파일 존재: {p.exists()}')"
echo ""

echo "✨ 확인 완료!"
