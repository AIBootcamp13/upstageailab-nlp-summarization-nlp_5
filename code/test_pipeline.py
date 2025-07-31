#!/usr/bin/env python3
"""
간단한 파이프라인 테스트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_utils import DataUtils
from utils.device_utils import get_optimal_device
import torch

def test_pipeline():
    print("=== 파이프라인 테스트 시작 ===")
    
    # 1. 디바이스 확인
    print("\n1. 디바이스 확인...")
    device, device_type = get_optimal_device()
    print(f"   - 디바이스: {device}")
    print(f"   - 타입: {device_type}")
    
    # 2. 데이터 로딩 테스트
    print("\n2. 데이터 로딩 테스트...")
    try:
        config = {
            'general': {
                'train_path': '../data/train.csv',
                'val_path': '../data/dev.csv',
                'test_path': '../data/test.csv'
            },
            'tokenizer': {
                'special_tokens': ['#Person1#', '#Person2#', '#Person3#']
            }
        }
        
        data_utils = DataUtils(config)
        datasets = data_utils.load_data()
        
        print(f"   - Train 샘플 수: {len(datasets['train'])}")
        print(f"   - Validation 샘플 수: {len(datasets['validation'])}")
        print(f"   - Test 샘플 수: {len(datasets['test'])}")
        
        # 첫 번째 샘플 확인
        first_sample = datasets['train'][0]
        print(f"   - 첫 번째 샘플 키: {first_sample.keys()}")
        
    except Exception as e:
        print(f"   ❌ 에러: {e}")
        return False
    
    print("\n✅ 모든 테스트 통과!")
    return True

if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)
