#!/usr/bin/env python3
"""
빠른 환경 확인 스크립트
"""

import sys
import os

print("🔍 Python 환경 확인")
print("=" * 50)
print(f"Python 경로: {sys.executable}")
print(f"Python 버전: {sys.version}")
print(f"현재 디렉토리: {os.getcwd()}")
print(f"PYTHONPATH: {sys.path[:3]}...")  # 처음 3개만 표시

print("\n📦 핵심 패키지 확인")
print("=" * 50)

packages = [
    'torch',
    'transformers', 
    'rouge',
    'requests',
    'yaml',
    'wandb'
]

for pkg in packages:
    try:
        module = __import__(pkg)
        version = getattr(module, '__version__', 'installed')
        print(f"✅ {pkg:<15} : {version}")
    except ImportError:
        print(f"❌ {pkg:<15} : NOT FOUND")

print("\n🚀 준비 완료!")
print("테스트 실행: bash run_1epoch_tests.sh")
