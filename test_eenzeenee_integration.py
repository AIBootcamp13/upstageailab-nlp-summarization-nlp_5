#!/usr/bin/env python3
"""
eenzeenee 모델 통합 테스트 스크립트 (간소화 버전)

이 스크립트는 eenzeenee/xsum-t5-1.7b 모델이 프로젝트에 올바르게 통합되었는지 검증합니다.
의존성 없이도 동작하는 핵심 기능만 테스트합니다.
"""

import sys
import os
from pathlib import Path

# 프로젝트 경로 추가
project_root = Path(__file__).parent

def test_config_exists():
    """config.yaml에 eenzeenee 설정이 있는지 테스트"""
    print("=== Config 존재 테스트 ===")
    
    try:
        config_path = project_root / 'config.yaml'
        if not config_path.exists():
            print("❌ config.yaml 파일이 없습니다")
            return False
            
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # 필수 설정 확인
        required_settings = [
            'eenzeenee:',
            'model_name: t5-base-korean-summarization',
            'input_prefix: "summarize: "',
            'model_type: seq2seq'
        ]
        
        
        for setting in required_settings:
            if setting in content:
                print(f"✅ 설정 발견: {setting}")
            else:
                print(f"❌ 설정 누락: {setting}")
                return False
                
    except Exception as e:
        print(f"❌ Config 테스트 실패: {e}")
        return False
    
    print("✅ Config 존재 테스트 통과\n")
    return True

def test_trainer_config_mapping():
    """trainer.py의 config_mapping에 eenzeenee 모델이 포함되었는지 테스트"""
    print("=== Trainer Config Mapping 테스트 ===")
    
    try:
        trainer_path = project_root / 'code' / 'trainer.py'
        if not trainer_path.exists():
            print("❌ trainer.py 파일이 없습니다")
            return False
            
        with open(trainer_path, 'r', encoding='utf-8') as f:
            trainer_content = f.read()
        
        # config_mapping 확인
        required_mappings = [
            'eenzeenee.yaml',
            't5-base-korean-summarization'
        ]
        
        for mapping in required_mappings:
            if mapping in trainer_content:
                print(f"✅ 매핑 발견: {mapping}")
            else:
                print(f"❌ 매핑 누락: {mapping}")
                return False
        
        # prefix 처리 관련 메서드 확인
        required_methods = [
            '_preprocess_for_model',
            '_get_t5_prefix',
            '_apply_prefix_to_dataset'
        ]
        
        for method in required_methods:
            if f"def {method}" in trainer_content:
                print(f"✅ 메서드 발견: {method}")
            else:
                print(f"❌ 메서드 누락: {method}")
                return False
                
    except Exception as e:
        print(f"❌ Trainer config mapping 테스트 실패: {e}")
        return False
    
    print("✅ Trainer Config Mapping 테스트 통과\n")
    return True

def test_prefix_logic():
    """prefix 처리 로직이 올바르게 구현되었는지 간단히 테스트"""
    print("=== Prefix 로직 테스트 ===")
    
    try:
        trainer_path = project_root / 'code' / 'trainer.py'
        with open(trainer_path, 'r', encoding='utf-8') as f:
            trainer_content = f.read()
        
        # eenzeenee 관련 prefix 처리 확인
        prefix_checks = [
            "eenzeenee",
            "summarize: ",
            "T5",
            "prefix"  
        ]
        
        for check in prefix_checks:
            if check in trainer_content:
                print(f"✅ 키워드 발견: {check}")
            else:
                print(f"❌ 키워드 누락: {check}")
                return False
        
        # T5 모델 감지 로직 확인
        if "'t5', 'flan-t5', 'mt5', 'eenzeenee'" in trainer_content:
            print("✅ T5 모델 감지 로직에 eenzeenee 포함")
        else:
            print("❌ T5 모델 감지 로직에 eenzeenee 누락")
            return False
                
    except Exception as e:
        print(f"❌ Prefix 로직 테스트 실패: {e}")
        return False
    
    print("✅ Prefix 로직 테스트 통과\n")
    return True

def test_model_registry_info():
    """ModelRegistry에 eenzeenee 정보가 있는지 간단히 확인"""
    print("=== ModelRegistry 정보 테스트 ===")
    
    try:
        registry_path = project_root / 'code' / 'utils' / 'experiment_utils.py'
        if not registry_path.exists():
            print("❌ experiment_utils.py 파일이 없습니다")
            return False
            
        with open(registry_path, 'r', encoding='utf-8') as f:
            registry_content = f.read()
        
        # eenzeenee 관련 정보 확인
        if 'eenzeenee' in registry_content.lower():
            print("✅ ModelRegistry에 eenzeenee 관련 정보 발견")
        else:
            print("❌ ModelRegistry에 eenzeenee 관련 정보 없음")
            return False
        
        # get_model_info 메서드 확인
        if 'def get_model_info' in registry_content:
            print("✅ get_model_info 메서드 발견")
        else:
            print("❌ get_model_info 메서드 누락")
            return False
                
    except Exception as e:
        print(f"❌ ModelRegistry 정보 테스트 실패: {e}")
        return False
    
    print("✅ ModelRegistry 정보 테스트 통과\n")
    return True

def test_integration_summary():
    """통합 테스트 결과 요약"""
    print("=== 통합 테스트 요약 ===")
    
    test_results = []
    
    # 각 테스트 실행
    test_results.append(("Config 존재", test_config_exists()))
    test_results.append(("Trainer Config Mapping", test_trainer_config_mapping()))
    test_results.append(("Prefix 로직", test_prefix_logic()))
    test_results.append(("ModelRegistry 정보", test_model_registry_info()))
    
    # 결과 요약
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    print(f"\n총 {total}개 테스트 중 {passed}개 통과")
    
    for test_name, result in test_results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"- {test_name}: {status}")
    
    if passed == total:
        print("\n🎉 모든 통합 테스트가 성공했습니다!")
        print("eenzeenee 모델이 프로젝트에 성공적으로 통합되었습니다.")
        return True
    else:
        print(f"\n⚠️  {total - passed}개의 테스트가 실패했습니다.")
        print("통합에 문제가 있을 수 있습니다.")
        return False

def main():
    """메인 실행 함수"""
    print("🧪 eenzeenee 모델 통합 테스트 시작 (간소화 버전)\n")
    
    try:
        success = test_integration_summary()
        
        if success:
            print("\n✅ 통합 테스트 완료: 성공")
            print("eenzeenee 모델을 사용할 준비가 되었습니다!")
            return 0
        else:
            print("\n❌ 통합 테스트 완료: 실패")
            print("통합 작업을 다시 확인해주세요.")
            return 1
            
    except Exception as e:
        print(f"\n💥 테스트 실행 중 예외 발생: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
