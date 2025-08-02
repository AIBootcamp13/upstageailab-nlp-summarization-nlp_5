#!/usr/bin/env python3
"""
추론 시스템 테스트 스크립트

실제 추론이 올바르게 동작하는지 확인하는 테스트
"""

import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from code.auto_experiment_runner import AutoExperimentRunner
from code.utils import load_config


def test_inference_system():
    """추론 시스템 테스트"""
    print("🧪 추론 시스템 테스트 시작")
    
    # 1. AutoExperimentRunner 초기화
    runner = AutoExperimentRunner()
    print("✅ AutoExperimentRunner 초기화 완료")
    
    # 2. 테스트용 더미 설정
    test_config = {
        'general': {
            'model_name': 'digit82/kobart-summarization',
            'data_path': './data/',
            'output_dir': './test_outputs/'
        },
        'tokenizer': {
            'encoder_max_len': 512,
            'decoder_max_len': 100,
            'bos_token': '<s>',
            'eos_token': '</s>',
            'special_tokens': ['#Person1#', '#Person2#', '#Person3#']
        },
        'inference': {
            'batch_size': 8,
            'no_repeat_ngram_size': 2,
            'early_stopping': True,
            'generate_max_length': 100,
            'num_beams': 4,
            'remove_tokens': ['<usr>', '<s>', '</s>', '<pad>']
        }
    }
    
    # 3. 테스트 체크포인트 경로 (실제 경로로 변경 필요)
    test_checkpoint = "outputs/dialogue_summarization_20250802_150702/checkpoints/checkpoint-3738"
    
    # 4. _run_test_inference 메서드 테스트
    print("\n📊 _run_test_inference 메서드 테스트")
    try:
        result = runner._run_test_inference(
            experiment_id="test_experiment",
            checkpoint_path=test_checkpoint,
            config=test_config
        )
        
        print("\n✅ 추론 테스트 결과:")
        print(f"  상태: {result.get('status')}")
        print(f"  제출 파일: {result.get('submission_path')}")
        
        if result.get('status') == 'error':
            print(f"  에러: {result.get('error')}")
            
    except Exception as e:
        print(f"\n❌ 추론 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🧪 테스트 완료")


if __name__ == "__main__":
    test_inference_system()
