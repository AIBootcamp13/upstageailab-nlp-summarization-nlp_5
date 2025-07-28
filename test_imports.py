#!/usr/bin/env python3
"""
단일 모델 간단 테스트 스크립트
가장 빠른 KoBART 모델로 1 epoch 테스트
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트 경로 설정
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "code"))

print("🚀 간단 테스트 시작...")
print(f"작업 디렉토리: {os.getcwd()}")
print(f"Python 경로: {sys.path[:2]}")

try:
    # 기본 imports 테스트
    print("\n📦 Import 테스트...")
    from utils import load_config
    print("✅ utils.load_config")
    
    from utils.data_utils import DataProcessor
    print("✅ utils.data_utils")
    
    from utils.experiment_utils import ExperimentTracker
    print("✅ utils.experiment_utils")
    
    # Rouge 가용성 확인
    from utils import ROUGE_AVAILABLE
    if ROUGE_AVAILABLE:
        print("✅ Rouge 메트릭 사용 가능")
    else:
        print("⚠️  Rouge 메트릭 사용 불가 (설치 필요)")
    
    print("\n✨ 모든 import 성공!")
    print("\n💡 이제 run_1epoch_tests.sh를 실행할 수 있습니다.")
    
except Exception as e:
    print(f"\n❌ 오류 발생: {e}")
    import traceback
    traceback.print_exc()
