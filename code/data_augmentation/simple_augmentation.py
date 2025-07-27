"""
Simple Data Augmentation Module for Dialogue Summarization

This module provides basic data augmentation techniques:
1. Synonym Replacement - Replace words with their synonyms
2. Sentence Reordering - Reorder sentences while maintaining logical flow
"""

import random
import re
from typing import List, Tuple, Dict, Optional
import json
import os
from pathlib import Path
import hashlib


class SimpleAugmenter:
    """Base class for simple data augmentation techniques"""
    
    def __init__(self, augmentation_ratio: float = 0.2, seed: int = 42):
        """
        Initialize the augmenter
        
        Args:
            augmentation_ratio: Ratio of data to augment (0.0 to 1.0)
            seed: Random seed for reproducibility
        """
        self.augmentation_ratio = augmentation_ratio
        self.seed = seed
        random.seed(seed)
        
        # 증강된 데이터 캐시 디렉토리
        self.cache_dir = Path("outputs/augmented_data_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def augment(self, text: str, **kwargs) -> str:
        """Override this method in subclasses"""
        raise NotImplementedError
    
    def should_augment(self) -> bool:
        """Randomly decide whether to augment based on ratio"""
        return random.random() < self.augmentation_ratio
    
    def get_cache_key(self, text: str, method: str) -> str:
        """Generate cache key for augmented text"""
        content = f"{method}:{text}:{self.seed}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def load_from_cache(self, cache_key: str) -> Optional[str]:
        """Load augmented text from cache if exists"""
        cache_file = self.cache_dir / f"{cache_key}.txt"
        if cache_file.exists():
            return cache_file.read_text(encoding='utf-8')
        return None
    
    def save_to_cache(self, cache_key: str, augmented_text: str):
        """Save augmented text to cache"""
        cache_file = self.cache_dir / f"{cache_key}.txt"
        cache_file.write_text(augmented_text, encoding='utf-8')


class SynonymReplacement(SimpleAugmenter):
    """Replace words with their synonyms"""
    
    def __init__(self, augmentation_ratio: float = 0.2, seed: int = 42, 
                 num_replacements: int = 3):
        super().__init__(augmentation_ratio, seed)
        self.num_replacements = num_replacements
        
        # 한국어 동의어 사전 (확장 가능)
        self.synonym_dict = {
            # 일반적인 대화 단어들
            "안녕하세요": ["안녕하십니까", "반갑습니다"],
            "감사합니다": ["고맙습니다", "감사드립니다"],
            "죄송합니다": ["미안합니다", "죄송해요"],
            "네": ["예", "그렇습니다", "맞습니다"],
            "아니요": ["아닙니다", "아니에요"],
            "좋아요": ["좋습니다", "괜찮아요", "좋네요"],
            "싫어요": ["싫습니다", "안 좋아요"],
            "알겠습니다": ["알았습니다", "이해했습니다"],
            "모르겠습니다": ["잘 모르겠습니다", "확실하지 않습니다"],
            
            # 일반적인 동사들
            "하다": ["실시하다", "진행하다"],
            "가다": ["이동하다", "향하다"],
            "오다": ["도착하다", "오시다"],
            "보다": ["관찰하다", "확인하다"],
            "먹다": ["식사하다", "드시다"],
            "말하다": ["이야기하다", "언급하다"],
            "생각하다": ["고민하다", "사고하다"],
            "알다": ["인지하다", "파악하다"],
            
            # 일반적인 형용사들
            "좋다": ["훌륭하다", "괜찮다"],
            "나쁘다": ["안 좋다", "별로다"],
            "크다": ["거대하다", "대단하다"],
            "작다": ["소규모다", "적다"],
            "많다": ["다수다", "풍부하다"],
            "적다": ["소수다", "부족하다"],
            
            # 시간 표현들
            "오늘": ["금일", "오늘날"],
            "내일": ["명일", "다음날"],
            "어제": ["전날", "전일"],
            
            # 격식체/비격식체 변형
            "저": ["제가", "본인"],
            "당신": ["귀하", "선생님"],
        }
        
        # 교체하지 않아야 할 특수 토큰들
        self.special_tokens = [
            '#Person1#', '#Person2#', '#Person3#',
            '#PhoneNumber#', '#Address#', '#DateOfBirth#',
            '#PassportNumber#', '#SSN#', '#CardNumber#',
            '#CarNumber#', '#Email#'
        ]
    
    def augment(self, text: str, preserve_special_tokens: bool = True) -> str:
        """
        Replace words with synonyms
        
        Args:
            text: Input text to augment
            preserve_special_tokens: Whether to preserve special tokens
            
        Returns:
            Augmented text with synonym replacements
        """
        # 캐시 먼저 확인
        cache_key = self.get_cache_key(text, "synonym_replacement")
        cached = self.load_from_cache(cache_key)
        if cached:
            return cached
        
        augmented = text
        words_replaced = 0
        
        # 랜덤 교체를 위해 동의어 사전 키들을 셔플
        words = list(self.synonym_dict.keys())
        random.shuffle(words)
        
        for word in words:
            if words_replaced >= self.num_replacements:
                break
                
            # 텍스트에 해당 단어가 없으면 스킵
            if word not in augmented:
                continue
            
            # 특수 토큰의 일부인지 확인하여 스킵
            if preserve_special_tokens:
                skip = False
                for token in self.special_tokens:
                    if word in token:
                        skip = True
                        break
                if skip:
                    continue
            
            # 랜덤 동의어로 교체
            synonyms = self.synonym_dict[word]
            synonym = random.choice(synonyms)
            
            # 과도한 교체를 피하기 위해 첫 번째 발생만 교체
            augmented = augmented.replace(word, synonym, 1)
            words_replaced += 1
        
        # 캐시에 저장
        self.save_to_cache(cache_key, augmented)
        
        return augmented


class SentenceReorder(SimpleAugmenter):
    """Reorder sentences in dialogue while maintaining logical flow"""
    
    def __init__(self, augmentation_ratio: float = 0.2, seed: int = 42,
                 max_reorder_distance: int = 2):
        super().__init__(augmentation_ratio, seed)
        self.max_reorder_distance = max_reorder_distance
    
    def _split_dialogue_turns(self, text: str) -> List[str]:
        """Split dialogue into turns based on Person markers"""
        # 대화 턴을 매칭하는 패턴
        pattern = r'(#Person\d+#:[^#]*?)(?=#Person\d+#:|$)'
        turns = re.findall(pattern, text, re.DOTALL)
        
        # 턴 정리
        turns = [turn.strip() for turn in turns if turn.strip()]
        
        return turns
    
    def _can_reorder_turns(self, turns: List[str], idx1: int, idx2: int) -> bool:
        """
        Check if two turns can be reordered without breaking dialogue flow
        
        Simple heuristic: Don't reorder if:
        1. They are from the same speaker
        2. One turn directly references the other (contains pronouns/references)
        """
        if idx1 >= len(turns) or idx2 >= len(turns):
            return False
        
        turn1 = turns[idx1]
        turn2 = turns[idx2]
        
        # 화자 추출
        speaker1 = turn1.split(':')[0]
        speaker2 = turn2.split(':')[0]
        
        # 같은 화자의 연속 턴은 순서 변경 안 함
        if speaker1 == speaker2:
            return False
        
        # 직접 참조 확인 (간단한 휴리스틱)
        reference_words = ['그것', '그거', '그래서', '그러나', '하지만', '그런데', 
                          '위에서', '아까', '방금', '그때', '그러면']
        
        for ref_word in reference_words:
            if ref_word in turn2.lower():
                return False
        
        return True
    
    def augment(self, text: str) -> str:
        """
        Reorder dialogue turns while maintaining logical flow
        
        Args:
            text: Input dialogue text
            
        Returns:
            Augmented text with reordered turns
        """
        # 캐시 먼저 확인
        cache_key = self.get_cache_key(text, "sentence_reorder")
        cached = self.load_from_cache(cache_key)
        if cached:
            return cached
        
        # 대화를 턴별로 분할
        turns = self._split_dialogue_turns(text)
        
        if len(turns) < 3:  # 의미 있는 순서 변경을 위해 최소 3개 턴 필요
            return text
        
        augmented_turns = turns.copy()
        
        # 일부 턴의 순서 변경 시도
        num_reorders = min(2, len(turns) // 3)  # 최대 2쌍까지 순서 변경
        
        for _ in range(num_reorders):
            # 순서를 변경할 랜덤 위치 선택
            idx = random.randint(1, len(augmented_turns) - 2)
            
            # 근처 턴과 교체 시도
            for distance in range(1, min(self.max_reorder_distance + 1, len(augmented_turns) - idx)):
                if self._can_reorder_turns(augmented_turns, idx, idx + distance):
                    # 턴 교체
                    augmented_turns[idx], augmented_turns[idx + distance] = \
                        augmented_turns[idx + distance], augmented_turns[idx]
                    break
        
        # 대화 재구성
        augmented = '\n'.join(augmented_turns)
        
        # 캐시에 저장
        self.save_to_cache(cache_key, augmented)
        
        return augmented


class DialogueAugmenter:
    """Main class to apply multiple augmentation techniques"""
    
    def __init__(self, config: Dict):
        """
        Initialize augmenter with configuration
        
        Args:
            config: Configuration dictionary with augmentation settings
        """
        self.config = config
        self.augmenters = []
        
        # 설정에 따라 증강기 초기화
        if config.get('synonym_replacement', {}).get('enabled', True):
            self.augmenters.append(
                SynonymReplacement(
                    augmentation_ratio=config.get('augmentation_ratio', 0.2),
                    seed=config.get('seed', 42),
                    num_replacements=config.get('synonym_replacement', {}).get('num_replacements', 3)
                )
            )
        
        if config.get('sentence_reorder', {}).get('enabled', True):
            self.augmenters.append(
                SentenceReorder(
                    augmentation_ratio=config.get('augmentation_ratio', 0.2),
                    seed=config.get('seed', 42),
                    max_reorder_distance=config.get('sentence_reorder', {}).get('max_distance', 2)
                )
            )
    
    def augment_dataset(self, dialogues: List[str], summaries: List[str]) -> Tuple[List[str], List[str]]:
        """
        Augment a dataset of dialogues and summaries
        
        Args:
            dialogues: List of dialogue texts
            summaries: List of corresponding summaries
            
        Returns:
            Tuple of (augmented_dialogues, augmented_summaries)
        """
        augmented_dialogues = []
        augmented_summaries = []
        
        # 원본 데이터 추가
        augmented_dialogues.extend(dialogues)
        augmented_summaries.extend(summaries)
        
        # 증강 적용
        for dialogue, summary in zip(dialogues, summaries):
            for augmenter in self.augmenters:
                if augmenter.should_augment():
                    aug_dialogue = augmenter.augment(dialogue)
                    
                    # 증강으로 실제로 텍스트가 변경된 경우만 추가
                    if aug_dialogue != dialogue:
                        augmented_dialogues.append(aug_dialogue)
                        augmented_summaries.append(summary)  # 원본 요약 유지
        
        return augmented_dialogues, augmented_summaries
    
    def get_augmentation_stats(self, original_size: int, augmented_size: int) -> Dict:
        """Calculate augmentation statistics"""
        return {
            'original_size': original_size,
            'augmented_size': augmented_size,
            'augmentation_ratio': (augmented_size - original_size) / original_size,
            'total_added': augmented_size - original_size
        }


# 사용 예제 및 테스트
if __name__ == "__main__":
    # 동의어 치환 테스트
    test_dialogue = """#Person1#: 안녕하세요. 오늘 날씨가 좋네요.
#Person2#: 네, 정말 좋아요. 어제보다 훨씬 따뜻해요.
#Person1#: 내일도 이렇게 좋을까요?
#Person2#: 아마 그럴 것 같아요. 일기예보를 확인해봐야겠네요."""
    
    print("Original dialogue:")
    print(test_dialogue)
    print("\n" + "="*50 + "\n")
    
    # 동의어 치환 테스트
    syn_augmenter = SynonymReplacement(augmentation_ratio=1.0, num_replacements=3)
    augmented = syn_augmenter.augment(test_dialogue)
    print("After synonym replacement:")
    print(augmented)
    print("\n" + "="*50 + "\n")
    
    # 문장 순서 변경 테스트
    reorder_augmenter = SentenceReorder(augmentation_ratio=1.0)
    augmented = reorder_augmenter.augment(test_dialogue)
    print("After sentence reordering:")
    print(augmented)
