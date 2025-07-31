#!/usr/bin/env python3
"""
데이터 로딩 테스트
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_utils import DataProcessor

# 간단한 config
config = {
    'general': {
        'train_path': 'data/train.csv',
        'val_path': 'data/dev.csv', 
        'test_path': 'data/test.csv'
    },
    'tokenizer': {
        'special_tokens': ['#Person1#', '#Person2#', '#Person3#', '#Person4#', '#Person5#', '#Person6#', '#Person7#', '#PhoneNumber#', '#Address#', '#PassportNumber#']
    }
}

print("데이터 로딩 테스트...")
try:
    processor = DataProcessor(config)
    
    # 개별적으로 로드
    train_df = processor.load_data('data/train.csv', is_test=False)
    val_df = processor.load_data('data/dev.csv', is_test=False)
    test_df = processor.load_data('data/test.csv', is_test=True)
    
    print(f"✅ 데이터 로드 성공!")
    print(f"   - Train: {len(train_df)} samples")
    print(f"   - Validation: {len(val_df)} samples")  
    print(f"   - Test: {len(test_df)} samples")
    
    print(f"\n첫 번째 train 샘플:")
    print(f"   - Columns: {list(train_df.columns)}")
    print(f"   - dialogue 길이: {len(train_df.iloc[0]['dialogue'])}")
    
except Exception as e:
    print(f"❌ 에러 발생: {e}")
    import traceback
    traceback.print_exc()
