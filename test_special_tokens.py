#!/usr/bin/env python3
"""
Special tokens 처리 테스트 스크립트
baseline.py와 trainer.py의 special_tokens 처리가 동일한지 확인
"""

import sys
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "code"))

try:
    from utils import load_config
    from trainer import DialogueSummarizationTrainer
except ImportError:
    # code 디렉토리에서 실행
    import os
    os.chdir(Path(__file__).parent)
    from utils import load_config
    from trainer import DialogueSummarizationTrainer
def test_special_tokens_loading():
    """special_tokens 로딩 테스트"""
    print("🔧 Special tokens 처리 테스트 시작")
    
    # eenzeenee T5 설정 로드 (special_tokens 정의되어 있음)
    config_path = "config/experiments/eenzeenee_t5_rtx3090.yaml"
    config = load_config(config_path)
    
    print(f"설정 파일: {config_path}")
    print(f"Special tokens in config: {config.get('tokenizer', {}).get('special_tokens', [])}")
    
    # 트레이너 생성 (토크나이저만 로드)
    trainer = DialogueSummarizationTrainer(config=config, sweep_mode=False)
    
    print(f"모델 체크포인트: {config.get('model', {}).get('checkpoint')}")
    
    # 토크나이저 로딩 (special_tokens 처리 포함)
    trainer._load_tokenizer()
    
    # 결과 확인
    special_tokens_added = getattr(trainer, '_special_tokens_added', False)
    new_vocab_size = getattr(trainer, '_new_vocab_size', None)
    
    print(f"\n📊 토크나이저 로딩 결과:")
    print(f"   Tokenizer type: {type(trainer.tokenizer).__name__}")
    print(f"   Vocab size: {len(trainer.tokenizer)}")
    print(f"   Special tokens added: {special_tokens_added}")
    print(f"   New vocab size: {new_vocab_size}")
    
    if special_tokens_added:
        # 실제 special tokens 확인
        print(f"\n   Special tokens map: {trainer.tokenizer.special_tokens_map}")
        
        # 각 special token의 ID 확인
        special_tokens_list = config.get('tokenizer', {}).get('special_tokens', [])
        for token in special_tokens_list[:3]:  # 처음 3개만 확인
            token_id = trainer.tokenizer.convert_tokens_to_ids(token)
            print(f"   '{token}' -> ID: {token_id}")
    
    print("\n✅ Special tokens 처리 테스트 완료!")
    return trainer

if __name__ == "__main__":
    test_special_tokens_loading()
