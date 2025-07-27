#!/usr/bin/env python3
"""
데이터 증강 테스트 스크립트
"""

import sys
from pathlib import Path
# 상위 디렉토리를 경로에 추가
# 상위 디렉토리를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_augmentation import SynonymReplacement, SentenceReorder, DialogueAugmenter


def test_augmentation():
    """Test augmentation on sample dialogues"""
    
    # 샘플 대화
    test_dialogues = [
        """#Person1#: 안녕하세요. 오늘 회의 시간이 어떻게 되나요?
#Person2#: 네, 오후 3시에 예정되어 있습니다.
#Person1#: 좋아요. 장소는 어디인가요?
#Person2#: 3층 회의실입니다. #Address# 건물이에요.""",
        
        """#Person1#: 죄송합니다. 늦어서요.
#Person2#: 괜찮아요. 막 시작했어요.
#Person1#: 감사합니다. 오늘 주제가 뭐죠?
#Person2#: 신제품 출시 계획입니다.""",
        
        """#Person1#: 전화번호 좀 알려주세요.
#Person2#: 네, #PhoneNumber#입니다.
#Person1#: 감사합니다. 내일 연락드릴게요.
#Person2#: 네, 기다리겠습니다."""
    ]
    
    print("="*60)
    print("DATA AUGMENTATION TEST")
    print("="*60)
    
    # 동의어 치환 테스트
    print("\n1. SYNONYM REPLACEMENT TEST")
    print("-"*40)
    syn_augmenter = SynonymReplacement(augmentation_ratio=1.0, num_replacements=3)
    
    for i, dialogue in enumerate(test_dialogues[:2]):
        print(f"\nOriginal dialogue {i+1}:")
        print(dialogue)
        print(f"\nAugmented dialogue {i+1}:")
        augmented = syn_augmenter.augment(dialogue)
        print(augmented)
        
        # 변경 사항 강조
        if dialogue != augmented:
            print("\n[Changes detected]")
        else:
            print("\n[No changes]")
    
    # 문장 순서 재배열 테스트
    print("\n\n2. SENTENCE REORDER TEST")
    print("-"*40)
    reorder_augmenter = SentenceReorder(augmentation_ratio=1.0)
    
    for i, dialogue in enumerate(test_dialogues[:2]):
        print(f"\nOriginal dialogue {i+1}:")
        print(dialogue)
        print(f"\nReordered dialogue {i+1}:")
        augmented = reorder_augmenter.augment(dialogue)
        print(augmented)
        
        # 순서 변경 여부 확인
        if dialogue != augmented:
            print("\n[Order changed]")
        else:
            print("\n[Order preserved - no valid reordering found]")
    
    # 조합 증강 테스트
    print("\n\n3. COMBINED AUGMENTATION TEST")
    print("-"*40)
    
    config = {
        'augmentation_ratio': 0.5,  # 50% chance for each augmenter
        'seed': 42,
        'synonym_replacement': {
            'enabled': True,
            'num_replacements': 3
        },
        'sentence_reorder': {
            'enabled': True,
            'max_distance': 2
        }
    }
    
    augmenter = DialogueAugmenter(config)
    
    # 소규모 데이터셋에서 테스트
    dialogues = test_dialogues
    summaries = [
        "회의 시간과 장소 확인",
        "회의 지각 사과 및 주제 확인", 
        "전화번호 교환 및 연락 약속"
    ]
    
    aug_dialogues, aug_summaries = augmenter.augment_dataset(dialogues, summaries)
    
    print(f"\nOriginal dataset size: {len(dialogues)}")
    print(f"Augmented dataset size: {len(aug_dialogues)}")
    print(f"Added samples: {len(aug_dialogues) - len(dialogues)}")
    
    # 증강된 예제 보여주기
    if len(aug_dialogues) > len(dialogues):
        print("\nAugmented samples:")
        for i in range(len(dialogues), min(len(dialogues) + 2, len(aug_dialogues))):
            print(f"\nAugmented dialogue {i - len(dialogues) + 1}:")
            print(aug_dialogues[i])
            print(f"Summary: {aug_summaries[i]}")
    
    # 특수 토큰 보존 테스트
    print("\n\n4. SPECIAL TOKEN PRESERVATION TEST")
    print("-"*40)
    
    special_dialogue = """#Person1#: 제 이메일은 #Email#입니다.
#Person2#: 감사합니다. 제 번호는 #PhoneNumber#이고 주소는 #Address#입니다."""
    
    print("Original with special tokens:")
    print(special_dialogue)
    
    augmented = syn_augmenter.augment(special_dialogue, preserve_special_tokens=True)
    print("\nAfter augmentation (tokens preserved):")
    print(augmented)
    
    # 특수 토큰이 보존되었는지 확인
    special_tokens = ['#Email#', '#PhoneNumber#', '#Address#']
    all_preserved = all(token in augmented for token in special_tokens)
    print(f"\nSpecial tokens preserved: {all_preserved}")
    
    print("\n" + "="*60)
    print("TEST COMPLETED")
    print("="*60)


if __name__ == "__main__":
    test_augmentation()
