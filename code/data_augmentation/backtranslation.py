"""
백트랜슬레이션 데이터 증강 모듈

한국어-영어-한국어 번역을 통해 의역된 데이터를 생성합니다.
Google Translate API와 로컬 번역 모델을 모두 지원합니다.
"""

import os
import time
import json
import hashlib
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np
from collections import defaultdict

# 번역 라이브러리
try:
    from googletrans import Translator as GoogleTranslator
    GOOGLE_TRANSLATE_AVAILABLE = True
except ImportError:
    GOOGLE_TRANSLATE_AVAILABLE = False
    print("Warning: googletrans not available. Install with: pip install googletrans==4.0.0-rc1")

try:
    from transformers import MarianMTModel, MarianTokenizer
    MARIAN_AVAILABLE = True
except ImportError:
    MARIAN_AVAILABLE = False
    print("Warning: transformers not available for MarianMT")

# 유사도 계산을 위한 라이브러리
from difflib import SequenceMatcher
import re


class BackTranslationAugmenter:
    """백트랜슬레이션을 통한 데이터 증강 클래스"""
    
    def __init__(
        self,
        method: str = "google",  # "google", "marian", "both"
        source_lang: str = "ko",
        intermediate_lang: str = "en",
        cache_dir: Optional[str] = "./cache/backtranslation",
        quality_threshold: float = 0.3,  # 원문과의 최소 차이
        max_similarity: float = 0.9,     # 원문과의 최대 유사도
        batch_size: int = 32,
        rate_limit_delay: float = 0.1,   # API 호출 간 지연 (초)
    ):
        """
        Args:
            method: 번역 방법 ("google", "marian", "both")
            source_lang: 원본 언어
            intermediate_lang: 중간 언어
            cache_dir: 캐시 디렉토리
            quality_threshold: 품질 임계값 (너무 비슷하면 제외)
            max_similarity: 최대 허용 유사도
            batch_size: 배치 크기 (MarianMT용)
            rate_limit_delay: API 호출 간 지연
        """
        self.method = method
        self.source_lang = source_lang
        self.intermediate_lang = intermediate_lang
        self.quality_threshold = quality_threshold
        self.max_similarity = max_similarity
        self.batch_size = batch_size
        self.rate_limit_delay = rate_limit_delay
        
        # 캐시 설정
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_file = self.cache_dir / "translation_cache.json"
            self.cache = self._load_cache()
        else:
            self.cache = {}
        
        # 번역기 초기화
        self._init_translators()
        
        # 통계
        self.stats = {
            'total_processed': 0,
            'cache_hits': 0,
            'api_calls': 0,
            'filtered_too_similar': 0,
            'filtered_too_different': 0,
            'errors': 0
        }
    
    def _init_translators(self):
        """번역기 초기화"""
        if self.method in ["google", "both"] and GOOGLE_TRANSLATE_AVAILABLE:
            self.google_translator = GoogleTranslator()
        else:
            self.google_translator = None
        
        if self.method in ["marian", "both"] and MARIAN_AVAILABLE:
            # MarianMT 모델 로드
            self.marian_models = {}
            
            # 한국어 -> 영어
            model_name_ko_en = "Helsinki-NLP/opus-mt-ko-en"
            self.marian_models['ko_en'] = {
                'tokenizer': MarianTokenizer.from_pretrained(model_name_ko_en),
                'model': MarianMTModel.from_pretrained(model_name_ko_en)
            }
            
            # 영어 -> 한국어
            model_name_en_ko = "Helsinki-NLP/opus-mt-en-ko"
            self.marian_models['en_ko'] = {
                'tokenizer': MarianTokenizer.from_pretrained(model_name_en_ko),
                'model': MarianMTModel.from_pretrained(model_name_en_ko)
            }
        else:
            self.marian_models = None
    
    def _load_cache(self) -> Dict[str, str]:
        """캐시 로드"""
        if self.cache_file.exists():
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_cache(self):
        """캐시 저장"""
        if self.cache_dir:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
    
    def _get_cache_key(self, text: str, src_lang: str, tgt_lang: str, method: str) -> str:
        """캐시 키 생성"""
        content = f"{text}|{src_lang}|{tgt_lang}|{method}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def augment(self, texts: Union[str, List[str]], num_augmentations: int = 1) -> List[str]:
        """
        텍스트를 백트랜슬레이션으로 증강
        
        Args:
            texts: 입력 텍스트 (문자열 또는 리스트)
            num_augmentations: 생성할 증강 수
            
        Returns:
            augmented_texts: 증강된 텍스트 리스트
        """
        if isinstance(texts, str):
            texts = [texts]
        
        all_augmented = []
        
        for text in tqdm(texts, desc="백트랜슬레이션 진행"):
            augmented = self._augment_single(text, num_augmentations)
            all_augmented.extend(augmented)
        
        # 통계 출력
        self._print_stats()
        
        # 캐시 저장
        self._save_cache()
        
        return all_augmented
    
    def _augment_single(self, text: str, num_augmentations: int) -> List[str]:
        """단일 텍스트 증강"""
        augmented_texts = []
        
        # 특수 토큰 보호
        protected_text, token_map = self._protect_special_tokens(text)
        
        for i in range(num_augmentations):
            try:
                # 백트랜슬레이션 수행
                if self.method == "google" and self.google_translator:
                    back_translated = self._google_backtranslate(protected_text)
                elif self.method == "marian" and self.marian_models:
                    back_translated = self._marian_backtranslate(protected_text)
                elif self.method == "both":
                    # 번갈아가며 사용
                    if i % 2 == 0 and self.google_translator:
                        back_translated = self._google_backtranslate(protected_text)
                    elif self.marian_models:
                        back_translated = self._marian_backtranslate(protected_text)
                    else:
                        continue
                else:
                    continue
                
                # 특수 토큰 복원
                back_translated = self._restore_special_tokens(back_translated, token_map)
                
                # 품질 검증
                if self._validate_quality(text, back_translated):
                    augmented_texts.append(back_translated)
                
            except Exception as e:
                print(f"백트랜슬레이션 오류: {e}")
                self.stats['errors'] += 1
        
        self.stats['total_processed'] += 1
        
        return augmented_texts
    
    def _protect_special_tokens(self, text: str) -> Tuple[str, Dict[str, str]]:
        """특수 토큰을 보호하기 위해 치환"""
        special_tokens = [
            '#Person1#', '#Person2#', '#Person3#',
            '#PhoneNumber#', '#Address#', '#DateOfBirth#',
            '#PassportNumber#', '#SSN#', '#CardNumber#',
            '#CarNumber#', '#Email#'
        ]
        
        token_map = {}
        protected_text = text
        
        for i, token in enumerate(special_tokens):
            if token in text:
                placeholder = f"SPECIALTOKEN{i}"
                token_map[placeholder] = token
                protected_text = protected_text.replace(token, placeholder)
        
        return protected_text, token_map
    
    def _restore_special_tokens(self, text: str, token_map: Dict[str, str]) -> str:
        """특수 토큰 복원"""
        restored_text = text
        for placeholder, token in token_map.items():
            restored_text = restored_text.replace(placeholder, token)
        return restored_text
    
    def _google_backtranslate(self, text: str) -> str:
        """Google Translate를 사용한 백트랜슬레이션"""
        # 캐시 확인
        cache_key = self._get_cache_key(text, self.source_lang, self.intermediate_lang, "google")
        if cache_key in self.cache:
            self.stats['cache_hits'] += 1
            return self.cache[cache_key]
        
        try:
            # 비율 제한
            time.sleep(self.rate_limit_delay)
            
            # 한국어 -> 영어
            intermediate = self.google_translator.translate(
                text, src=self.source_lang, dest=self.intermediate_lang
            ).text
            
            # 영어 -> 한국어
            back_translated = self.google_translator.translate(
                intermediate, src=self.intermediate_lang, dest=self.source_lang
            ).text
            
            # 캐시 저장
            self.cache[cache_key] = back_translated
            self.stats['api_calls'] += 1
            
            return back_translated
            
        except Exception as e:
            print(f"Google Translate 오류: {e}")
            return text
    
    def _marian_backtranslate(self, text: str) -> str:
        """MarianMT를 사용한 백트랜슬레이션"""
        # 캐시 확인
        cache_key = self._get_cache_key(text, self.source_lang, self.intermediate_lang, "marian")
        if cache_key in self.cache:
            self.stats['cache_hits'] += 1
            return self.cache[cache_key]
        
        try:
            # 한국어 -> 영어
            ko_en_model = self.marian_models['ko_en']
            inputs = ko_en_model['tokenizer'](text, return_tensors="pt", padding=True)
            translated = ko_en_model['model'].generate(**inputs)
            intermediate = ko_en_model['tokenizer'].decode(translated[0], skip_special_tokens=True)
            
            # 영어 -> 한국어
            en_ko_model = self.marian_models['en_ko']
            inputs = en_ko_model['tokenizer'](intermediate, return_tensors="pt", padding=True)
            translated = en_ko_model['model'].generate(**inputs)
            back_translated = en_ko_model['tokenizer'].decode(translated[0], skip_special_tokens=True)
            
            # 캐시 저장
            self.cache[cache_key] = back_translated
            
            return back_translated
            
        except Exception as e:
            print(f"MarianMT 오류: {e}")
            return text
    
    def _validate_quality(self, original: str, augmented: str) -> bool:
        """증강된 텍스트의 품질 검증"""
        # 길이 비교
        len_ratio = len(augmented) / len(original)
        if len_ratio < 0.5 or len_ratio > 2.0:
            self.stats['filtered_too_different'] += 1
            return False
        
        # 유사도 계산
        similarity = self._calculate_similarity(original, augmented)
        
        # 너무 비슷하면 제외
        if similarity > self.max_similarity:
            self.stats['filtered_too_similar'] += 1
            return False
        
        # 너무 다르면 제외
        if similarity < self.quality_threshold:
            self.stats['filtered_too_different'] += 1
            return False
        
        # 특수 토큰 보존 확인
        if not self._check_special_tokens_preserved(original, augmented):
            return False
        
        return True
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """두 텍스트의 유사도 계산"""
        # 정규화
        text1_normalized = re.sub(r'\s+', ' ', text1.lower().strip())
        text2_normalized = re.sub(r'\s+', ' ', text2.lower().strip())
        
        # SequenceMatcher를 사용한 유사도
        return SequenceMatcher(None, text1_normalized, text2_normalized).ratio()
    
    def _check_special_tokens_preserved(self, original: str, augmented: str) -> bool:
        """특수 토큰이 보존되었는지 확인"""
        special_token_pattern = r'#\w+#'
        
        original_tokens = set(re.findall(special_token_pattern, original))
        augmented_tokens = set(re.findall(special_token_pattern, augmented))
        
        # 모든 특수 토큰이 보존되어야 함
        return original_tokens == augmented_tokens
    
    def _print_stats(self):
        """통계 출력"""
        print("\n=== 백트랜슬레이션 통계 ===")
        print(f"총 처리: {self.stats['total_processed']}")
        print(f"캐시 히트: {self.stats['cache_hits']}")
        print(f"API 호출: {self.stats['api_calls']}")
        print(f"너무 비슷하여 제외: {self.stats['filtered_too_similar']}")
        print(f"너무 달라서 제외: {self.stats['filtered_too_different']}")
        print(f"오류 발생: {self.stats['errors']}")
    
    def augment_dataframe(
        self, 
        df: pd.DataFrame, 
        text_column: str = 'dialogue',
        num_augmentations: int = 1,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        DataFrame의 텍스트 컬럼을 증강
        
        Args:
            df: 입력 DataFrame
            text_column: 증강할 텍스트 컬럼명
            num_augmentations: 각 샘플당 생성할 증강 수
            save_path: 증강된 데이터 저장 경로
            
        Returns:
            augmented_df: 증강된 DataFrame
        """
        augmented_rows = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="DataFrame 증강"):
            # 원본 추가
            augmented_rows.append(row.to_dict())
            
            # 증강 텍스트 생성
            text = row[text_column]
            augmented_texts = self._augment_single(text, num_augmentations)
            
            # 증강된 버전 추가
            for aug_text in augmented_texts:
                aug_row = row.to_dict()
                aug_row[text_column] = aug_text
                aug_row['is_augmented'] = True
                aug_row['augmentation_method'] = 'backtranslation'
                augmented_rows.append(aug_row)
        
        # DataFrame 생성
        augmented_df = pd.DataFrame(augmented_rows)
        
        # 저장
        if save_path:
            augmented_df.to_csv(save_path, index=False)
            print(f"증강된 데이터 저장: {save_path}")
        
        return augmented_df


class MultilingualBackTranslation(BackTranslationAugmenter):
    """다중 언어를 활용한 백트랜슬레이션"""
    
    def __init__(
        self,
        intermediate_langs: List[str] = ["en", "ja", "zh-cn"],
        **kwargs
    ):
        """
        Args:
            intermediate_langs: 중간 언어 리스트
            **kwargs: BackTranslationAugmenter의 인자들
        """
        super().__init__(**kwargs)
        self.intermediate_langs = intermediate_langs
    
    def _augment_single(self, text: str, num_augmentations: int) -> List[str]:
        """다중 언어를 통한 증강"""
        augmented_texts = []
        
        # 각 중간 언어에 대해 백트랜슬레이션
        for i in range(num_augmentations):
            # 중간 언어 선택 (순환)
            intermediate_lang = self.intermediate_langs[i % len(self.intermediate_langs)]
            self.intermediate_lang = intermediate_lang
            
            # 백트랜슬레이션 수행
            result = super()._augment_single(text, 1)
            augmented_texts.extend(result)
        
        return augmented_texts


def create_backtranslation_augmenter(config: Dict) -> BackTranslationAugmenter:
    """설정에 따라 백트랜슬레이션 증강기 생성"""
    
    augmenter_type = config.get('augmenter_type', 'single')
    
    if augmenter_type == 'multilingual':
        return MultilingualBackTranslation(
            method=config.get('method', 'google'),
            intermediate_langs=config.get('intermediate_langs', ['en', 'ja']),
            quality_threshold=config.get('quality_threshold', 0.3),
            max_similarity=config.get('max_similarity', 0.9),
            cache_dir=config.get('cache_dir', './cache/backtranslation')
        )
    else:
        return BackTranslationAugmenter(
            method=config.get('method', 'google'),
            source_lang=config.get('source_lang', 'ko'),
            intermediate_lang=config.get('intermediate_lang', 'en'),
            quality_threshold=config.get('quality_threshold', 0.3),
            max_similarity=config.get('max_similarity', 0.9),
            cache_dir=config.get('cache_dir', './cache/backtranslation')
        )


if __name__ == "__main__":
    # 테스트
    augmenter = BackTranslationAugmenter(method="google")
    
    test_text = "#Person1#: 안녕하세요, 오늘 회의는 몇 시에 시작하나요? #Person2#: 오후 3시에 시작합니다."
    
    augmented = augmenter.augment(test_text, num_augmentations=3)
    
    print("\n원본:")
    print(test_text)
    print("\n증강된 텍스트:")
    for i, aug_text in enumerate(augmented, 1):
        print(f"{i}. {aug_text}")
