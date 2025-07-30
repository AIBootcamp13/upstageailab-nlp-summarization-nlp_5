#!/usr/bin/env python3
"""
환경 자동 감지 시스템 테스트 스크립트

AIStages 서버에서 환경을 자동 감지하고 
Unsloth 활성화 여부를 결정하는 시스템을 테스트합니다.
"""

import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "code"))

from code.utils.environment_detector import EnvironmentDetector

def test_environment_detection():
    """환경 감지 시스템 테스트"""
    print("🔍 환경 자동 감지 시스템 테스트")
    print("=" * 60)
    
    # 환경 감지기 초기화
    detector = EnvironmentDetector()
    
    # 환경 정보 출력
    detector.print_environment_summary()
    
    # 권장 설정 출력
    config = detector.get_recommended_config()
    print(f"\n📋 상세 권장 설정:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 테스트 결과 요약
    env_info = detector.detect_environment()
    
    print(f"\n🎯 테스트 결과 요약")
    print(f"Ubuntu 환경: {'✅' if env_info['is_ubuntu'] else '❌'}")
    print(f"CUDA 사용 가능: {'✅' if env_info['is_cuda_available'] else '❌'}")
    print(f"Unsloth 권장: {'✅' if env_info['unsloth_recommended'] else '❌'}")
    print(f"Unsloth 설치: {'✅' if env_info['unsloth_available'] else '❌'}")
    
    if env_info['unsloth_recommended'] and env_info['unsloth_available']:
        print(f"\n🚀 결론: AIStages 서버에서 Unsloth 자동 활성화 가능!")
        print(f"   - 모든 새로운 모델 학습에서 자동으로 Unsloth 적용")
        print(f"   - 메모리 30-50% 절약, 학습 속도 2-5배 향상 기대")
    elif env_info['unsloth_recommended'] and not env_info['unsloth_available']:
        print(f"\n⚠️  결론: Unsloth 설치 필요")
        print(f"   - 환경은 적합하지만 패키지가 설치되지 않음")
        print(f"   - 설치 후 자동 활성화 가능")
    else:
        print(f"\n❌ 결론: 현재 환경에서는 Unsloth 사용 불가")
        print(f"   - 일반 QLoRA 방식 사용 권장")

if __name__ == "__main__":
    test_environment_detection()
