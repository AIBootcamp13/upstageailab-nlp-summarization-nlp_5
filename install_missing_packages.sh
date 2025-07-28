#!/bin/bash
# 누락된 의존성 패키지 설치 스크립트

echo "🔧 누락된 패키지 설치 시작..."
echo "================================"

# 현재 환경 확인
echo "📍 현재 Python 환경:"
which python
python --version
echo ""

# rouge 패키지 설치 (py-rouge)
echo "📦 rouge 패키지 설치..."
pip install rouge==1.0.1

# requests 의존성 패키지 설치
echo "📦 requests 의존성 패키지 설치..."
pip install charset-normalizer chardet

# 설치 확인
echo ""
echo "✅ 설치 완료. 패키지 상태 확인 중..."
python -c "import rouge; print(f'✅ rouge 버전: {rouge.__version__}')" 2>/dev/null || echo "❌ rouge 설치 실패"
python -c "import charset_normalizer; print(f'✅ charset_normalizer 설치 완료')" 2>/dev/null || echo "❌ charset_normalizer 설치 실패"
python -c "import chardet; print(f'✅ chardet 설치 완료')" 2>/dev/null || echo "❌ chardet 설치 실패"

echo ""
echo "💡 설치가 완료되었습니다. 다시 테스트를 실행해주세요."
echo "   bash run_1epoch_tests.sh"
