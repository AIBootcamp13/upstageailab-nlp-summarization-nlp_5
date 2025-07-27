"""
텍스트 정규화 모듈 - 대화 요약을 위한 비격식 언어 정규화

이 모듈은 번역된 대화 데이터의 노이즈를 줄이고 품질을 향상시킵니다:
1. 약어 및 구어체 표현 정규화
2. 반복 문자/단어 정리
3. 특수문자 및 이스케이프 문자 제거
4. 이모티콘 정규화
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from collections import Counter
import unicodedata

logger = logging.getLogger(__name__)


class TextNormalizer:
    """대화 텍스트 정규화를 위한 메인 클래스"""
    
    def __init__(self, normalize_level: str = "medium"):
        """
        텍스트 정규화기 초기화
        
        Args:
            normalize_level: 정규화 강도 ('light', 'medium', 'heavy')
        """
        self.normalize_level = normalize_level
        
        # 약어 및 구어체 사전
        self.abbreviation_dict = {
            # 일반 약어
            "그럼": "그러면",
            "근데": "그런데",
            "얘기": "이야기",
            "걔": "그 아이",
            "쟤": "저 아이",
            "뭐": "무엇",
            "왜": "왜",
            "젤": "제일",
            "되게": "매우",
            "진짜": "정말",
            "막": "매우",
            
            # 축약형
            "할게": "할 것이에요",
            "할께": "할 것이에요",
            "해줘": "해 주세요",
            "해주세요": "해 주세요",
            "할건데": "할 것인데",
            "하는거": "하는 것",
            "되는거": "되는 것",
            "없는거": "없는 것",
            "있는거": "있는 것",
            
            # 구어체 표현
            "어케": "어떻게",
            "이케": "이렇게",
            "저케": "저렇게",
            "머": "뭐",
            "니": "너",
            "니가": "네가",
            "맞아": "맞아요",
            "그래": "그래요",
            
            # 감탄사/간투사
            "아": "",
            "어": "",
            "음": "",
            "으음": "",
            "그": "",
            "저": "",
            "이": "",
            
            # 영어 약어 (번역 데이터에 남아있을 수 있음)
            "btw": "그런데",
            "BTW": "그런데",
            "fyi": "참고로",
            "FYI": "참고로",
            "asap": "가능한 빨리",
            "ASAP": "가능한 빨리",
        }
        
        # 이모티콘 및 감정 표현 사전
        self.emoticon_dict = {
            # 긍정적 표현
            "ㅋㅋ": "[웃음]",
            "ㅋㅋㅋ": "[웃음]",
            "ㅋㅋㅋㅋ": "[웃음]",
            "ㅎㅎ": "[웃음]",
            "ㅎㅎㅎ": "[웃음]",
            "ㅎㅎㅎㅎ": "[웃음]",
            "ㅋ": "[웃음]",
            "ㅎ": "[웃음]",
            ":)": "[웃음]",
            ":-)": "[웃음]",
            "^^": "[웃음]",
            "^-^": "[웃음]",
            
            # 부정적 표현
            "ㅠㅠ": "[슬픔]",
            "ㅠㅠㅠ": "[슬픔]",
            "ㅜㅜ": "[슬픔]",
            "ㅜㅜㅜ": "[슬픔]",
            ":(": "[슬픔]",
            ":-(": "[슬픔]",
            "ㅡㅡ": "[짜증]",
            "ㅡㅡ;": "[짜증]",
            
            # 놀람/의문
            "??": "[의문]",
            "???": "[의문]",
            "!": "[놀람]",
            "!!": "[놀람]",
            "!!!": "[놀람]",
            "?!": "[놀람]",
            "!?": "[놀람]",
            
            # 기타
            "...": "…",
            "..": "…",
            "....": "…",
        }
        
        # 특수 토큰 보호 리스트
        self.protected_tokens = [
            '#Person1#', '#Person2#', '#Person3#',
            '#PhoneNumber#', '#Address#', '#DateOfBirth#',
            '#PassportNumber#', '#SSN#', '#CardNumber#',
            '#CarNumber#', '#Email#'
        ]
    
    def normalize(self, text: str) -> str:
        """
        텍스트 정규화 메인 함수
        
        Args:
            text: 입력 텍스트
            
        Returns:
            정규화된 텍스트
        """
        # 원본 텍스트 로깅 (디버깅용)
        logger.debug(f"정규화 전: {text[:100]}...")
        
        # 특수 토큰 보호
        protected_parts = self._protect_special_tokens(text)
        
        # 정규화 수행
        normalized = text
        
        if self.normalize_level in ['light', 'medium', 'heavy']:
            # 기본 정리
            normalized = self._clean_whitespace(normalized)
            normalized = self._remove_escape_chars(normalized)
            
            # 반복 문자 정리
            normalized = self._normalize_repeated_chars(normalized)
            
            # 이모티콘 정규화
            normalized = self._normalize_emoticons(normalized)
        
        if self.normalize_level in ['medium', 'heavy']:
            # 약어 및 구어체 정규화
            normalized = self._normalize_abbreviations(normalized)
            
            # 문장부호 정리
            normalized = self._normalize_punctuation(normalized)
        
        if self.normalize_level == 'heavy':
            # 간투사 제거
            normalized = self._remove_fillers(normalized)
            
            # 더 적극적인 정규화
            normalized = self._aggressive_normalization(normalized)
        
        # 특수 토큰 복원
        normalized = self._restore_special_tokens(normalized, protected_parts)
        
        # 최종 정리
        normalized = self._final_cleanup(normalized)
        
        logger.debug(f"정규화 후: {normalized[:100]}...")
        
        return normalized
    
    def _protect_special_tokens(self, text: str) -> Dict[str, str]:
        """특수 토큰을 임시 플레이스홀더로 치환하여 보호"""
        protected = {}
        for i, token in enumerate(self.protected_tokens):
            placeholder = f"__PROTECTED_{i}__"
            if token in text:
                protected[placeholder] = token
                text = text.replace(token, placeholder)
        return protected
    
    def _restore_special_tokens(self, text: str, protected: Dict[str, str]) -> str:
        """보호된 특수 토큰 복원"""
        for placeholder, token in protected.items():
            text = text.replace(placeholder, token)
        return text
    
    def _clean_whitespace(self, text: str) -> str:
        """공백 문자 정리"""
        # 다중 공백을 단일 공백으로
        text = re.sub(r'\s+', ' ', text)
        # 라인 시작/끝 공백 제거
        text = text.strip()
        return text
    
    def _remove_escape_chars(self, text: str) -> str:
        """이스케이프 문자 제거"""
        # 일반적인 이스케이프 시퀀스 제거
        text = text.replace('\\n', ' ')
        text = text.replace('\\t', ' ')
        text = text.replace('\\r', ' ')
        text = text.replace('\\"', '"')
        text = text.replace("\\'", "'")
        text = text.replace('\\\\', '\\')
        
        # 유니코드 정규화
        text = unicodedata.normalize('NFKC', text)
        
        return text
    
    def _normalize_repeated_chars(self, text: str) -> str:
        """반복 문자 정규화"""
        # 3번 이상 반복되는 문자를 2번으로 줄임
        # 예: "아아아아" → "아아"
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        # 반복되는 단어 제거 (같은 단어가 연속 2번 이상)
        # 예: "정말 정말 정말" → "정말"
        words = text.split()
        normalized_words = []
        prev_word = None
        
        for word in words:
            if word != prev_word:
                normalized_words.append(word)
            prev_word = word
        
        return ' '.join(normalized_words)
    
    def _normalize_emoticons(self, text: str) -> str:
        """이모티콘 정규화"""
        for emoticon, replacement in self.emoticon_dict.items():
            text = text.replace(emoticon, replacement)
        
        # 남은 반복 자음/모음 정리
        # ㅋ, ㅎ, ㅠ, ㅜ 등의 반복
        text = re.sub(r'[ㅋ]{2,}', '[웃음]', text)
        text = re.sub(r'[ㅎ]{2,}', '[웃음]', text)
        text = re.sub(r'[ㅠㅜ]{2,}', '[슬픔]', text)
        
        return text
    
    def _normalize_abbreviations(self, text: str) -> str:
        """약어 및 구어체 정규화"""
        # 단어 경계를 고려한 치환
        for abbr, full in self.abbreviation_dict.items():
            if abbr and full:  # 빈 문자열 치환 방지
                # 단어 경계 패턴
                pattern = r'\b' + re.escape(abbr) + r'\b'
                text = re.sub(pattern, full, text, flags=re.IGNORECASE)
        
        return text
    
    def _normalize_punctuation(self, text: str) -> str:
        """문장부호 정규화"""
        # 연속된 문장부호 정리
        text = re.sub(r'[.]{2,}', '…', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        # 문장부호 앞 공백 제거
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        
        # 문장부호 뒤 공백 추가 (문장 끝이 아닌 경우)
        text = re.sub(r'([.,!?;:])([^\s])', r'\1 \2', text)
        
        return text
    
    def _remove_fillers(self, text: str) -> str:
        """간투사 및 필러 제거 (heavy 모드)"""
        # 문장 시작의 간투사 제거
        fillers = ['아', '어', '음', '그', '저', '이', '뭐', '막']
        for filler in fillers:
            # 문장 시작
            text = re.sub(r'^' + filler + r'[,\s]+', '', text)
            # 문장 중간 (쉼표와 함께)
            text = re.sub(r',\s*' + filler + r'[,\s]+', ', ', text)
        
        return text
    
    def _aggressive_normalization(self, text: str) -> str:
        """적극적인 정규화 (heavy 모드)"""
        # 괄호 안 내용 제거 (설명 제거)
        text = re.sub(r'\([^)]*\)', '', text)
        text = re.sub(r'\[[^\]]*\]', '', text)
        
        # 특수 기호 제거 (필수 문장부호 제외)
        text = re.sub(r'[^\w\s가-힣.,!?#]', ' ', text)
        
        return text
    
    def _final_cleanup(self, text: str) -> str:
        """최종 정리"""
        # 다시 한번 공백 정리
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # 빈 괄호 제거
        text = text.replace('()', '')
        text = text.replace('[]', '')
        
        return text
    
    def get_normalization_stats(self, original: str, normalized: str) -> Dict:
        """정규화 통계 반환"""
        return {
            'original_length': len(original),
            'normalized_length': len(normalized),
            'length_reduction': len(original) - len(normalized),
            'reduction_percentage': ((len(original) - len(normalized)) / len(original) * 100) if len(original) > 0 else 0,
            'original_tokens': len(original.split()),
            'normalized_tokens': len(normalized.split()),
            'token_reduction': len(original.split()) - len(normalized.split())
        }


# 배치 처리를 위한 헬퍼 함수
def normalize_batch(texts: List[str], normalize_level: str = "medium") -> List[str]:
    """텍스트 배치 정규화"""
    normalizer = TextNormalizer(normalize_level)
    return [normalizer.normalize(text) for text in texts]


# 테스트 및 예제
if __name__ == "__main__":
    # 테스트 샘플
    test_samples = [
        "#Person1#: 아 진짜 그럼 내일 뭐 할건데??? ㅋㅋㅋㅋ",
        "#Person2#: 음... 그게 말이야... 내일은 #Address#에서 만나자구~~~ ㅎㅎㅎ",
        "#Person1#: 어케 가는거야?? 막 복잡하지 않아??",
        "#Person2#: 아니아니 괜찮아 진짜진짜 쉬워!!! BTW 전화번호 #PhoneNumber#야",
        "아아아아 정말정말정말 힘들어어어어.... ㅠㅠㅠㅠ",
        "그그그 뭐뭐뭐 하는거야??? 이이이거 맞아????",
    ]
    
    print("="*60)
    print("텍스트 정규화 테스트")
    print("="*60)
    
    for level in ['light', 'medium', 'heavy']:
        print(f"\n정규화 레벨: {level}")
        print("-"*40)
        
        normalizer = TextNormalizer(level)
        
        for i, sample in enumerate(test_samples[:3]):
            normalized = normalizer.normalize(sample)
            stats = normalizer.get_normalization_stats(sample, normalized)
            
            print(f"\n샘플 {i+1}:")
            print(f"원본: {sample}")
            print(f"정규화: {normalized}")
            print(f"토큰 감소: {stats['original_tokens']} → {stats['normalized_tokens']} ({stats['token_reduction']}개 감소)")
