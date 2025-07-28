#!/bin/bash
# 누락된 패키지 설치 스크립트

echo "🔧 누락된 패키지 설치 중..."

# evaluate 패키지 설치
echo "📦 evaluate 패키지 설치..."
pip install evaluate==0.4.0

# 기타 필수 패키지 확인 및 설치
echo "📦 기타 필수 패키지 확인..."
pip install rouge==1.0.1
pip install rouge-score==0.1.2

echo "✅ 패키지 설치 완료!"
echo ""
echo "설치된 패키지:"
pip list | grep -E "(evaluate|rouge)"
