#!/usr/bin/env python3
"""
ModelRegistry.get_model_info() 메서드 테스트 스크립트
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'code'))

from code.utils.experiment_utils import ModelRegistry

def test_get_model_info():
    """get_model_info 메서드 테스트"""
    registry = ModelRegistry()
    
    # 테스트 케이스들
    test_cases = [
        ("eenzeenee/t5-base-korean-summarization", "seq2seq", "t5"),
        ("digit82/kobart-summarization", "seq2seq", "bart"),
        ("csebuetnlp/mT5_multilingual_XLSum", "seq2seq", "t5"),
        ("unknown-model/test", None, None),
    ]
    
    print("=== ModelRegistry.get_model_info() 테스트 ===")
    
    for model_name, expected_type, expected_arch in test_cases:
        print(f"\n테스트: {model_name}")
        result = registry.get_model_info(model_name)
        
        if result is None:
            if expected_type is None:
                print("✅ 예상대로 None 반환")
            else:
                print(f"❌ 예상 타입: {expected_type}, 실제: None")
        else:
            actual_type = result.get('type')
            actual_arch = result.get('architecture')
            
            if actual_type == expected_type and actual_arch == expected_arch:
                print(f"✅ 타입: {actual_type}, 아키텍처: {actual_arch}")
                if 'eenzeenee' in model_name:
                    print(f"   Prefix 지원: {result.get('requires_prefix', False)}")
                    print(f"   Input Prefix: '{result.get('input_prefix', '')}'")
            else:
                print(f"❌ 예상 - 타입: {expected_type}, 아키텍처: {expected_arch}")
                print(f"   실제 - 타입: {actual_type}, 아키텍처: {actual_arch}")

if __name__ == "__main__":
    test_get_model_info()
