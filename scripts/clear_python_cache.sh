#!/bin/bash

# Python 캐시 정리 스크립트

echo "🧹 Python 캐시 정리 중..."

# __pycache__ 디렉토리 삭제
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# .pyc 파일 삭제
find . -name "*.pyc" -delete 2>/dev/null

# .pyo 파일 삭제
find . -name "*.pyo" -delete 2>/dev/null

# .pyd 파일 삭제 (Windows)
find . -name "*.pyd" -delete 2>/dev/null

# Python egg 캐시 정리
find . -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null
find . -name "*.egg" -delete 2>/dev/null

# pytest 캐시 정리
find . -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null

# mypy 캐시 정리
find . -name ".mypy_cache" -type d -exec rm -rf {} + 2>/dev/null

# Git 인덱스 새로고침
if [ -d ".git" ]; then
    echo "🔄 Git 인덱스 새로고침..."
    git update-index --refresh 2>/dev/null || true
    git status --porcelain 2>/dev/null || true
fi

# Python 바이트코드 강제 재컴파일
echo "🔧 Python 모듈 재컴파일..."
python -m compileall -f code/ 2>/dev/null || true

echo "✅ 캐시 정리 완료!"
