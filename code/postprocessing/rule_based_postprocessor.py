"""
Rule-based Postprocessing Module for Dialogue Summarization

This module provides postprocessing techniques to improve summary quality:
1. Duplicate removal - Remove repeated phrases and sentences
2. Length optimization - Adjust summary length to target range
3. Special token validation - Ensure PII tokens are preserved
"""

import re
from typing import List, Dict, Set, Tuple, Optional
import logging
from collections import Counter
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


class PostProcessor:
    """Base class for postprocessing techniques"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize postprocessor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
    
    def process(self, text: str, **kwargs) -> str:
        """Override this method in subclasses"""
        raise NotImplementedError


class DuplicateRemover(PostProcessor):
    """Remove duplicate phrases and sentences from summaries"""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.min_phrase_length = self.config.get('min_phrase_length', 3)
        self.similarity_threshold = self.config.get('similarity_threshold', 0.85)
    
    def _extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text"""
        # 한국어 문장 구분자
        sentences = re.split(r'[.!?]\s*', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_phrases(self, text: str) -> List[str]:
        """Extract meaningful phrases from text"""
        # 일반적인 구분자로 분할
        phrases = re.split(r'[,;]\s*', text)
        # 긴 구문을 접속사로도 분할
        expanded_phrases = []
        for phrase in phrases:
            sub_phrases = re.split(r'\s+(그리고|또한|하지만|그러나|그래서)\s+', phrase)
            expanded_phrases.extend(sub_phrases)
        
        return [p.strip() for p in expanded_phrases if len(p.split()) >= self.min_phrase_length]
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings"""
        return SequenceMatcher(None, str1, str2).ratio()
    
    def _remove_duplicate_sentences(self, text: str) -> str:
        """Remove duplicate or highly similar sentences"""
        sentences = self._extract_sentences(text)
        if len(sentences) <= 1:
            return text
        
        unique_sentences = [sentences[0]]
        
        for sent in sentences[1:]:
            is_duplicate = False
            
            for unique_sent in unique_sentences:
                similarity = self._calculate_similarity(sent.lower(), unique_sent.lower())
                if similarity >= self.similarity_threshold:
                    is_duplicate = True
                    logger.debug(f"Duplicate sentence found: '{sent}' similar to '{unique_sent}' (similarity: {similarity:.2f})")
                    break
            
            if not is_duplicate:
                unique_sentences.append(sent)
        
        # 텍스트 재구성
        result = '. '.join(unique_sentences)
        if not result.endswith('.'):
            result += '.'
        
        return result
    
    def _remove_repeated_phrases(self, text: str) -> str:
        """Remove consecutively repeated phrases"""
        # 반복되는 구문을 찾는 패턴
        # 예: "회의 시간 회의 시간" -> "회의 시간"
        pattern = r'\b(\w+(?:\s+\w+){0,3})\s+\1\b'
        
        cleaned_text = text
        while True:
            new_text = re.sub(pattern, r'\1', cleaned_text)
            if new_text == cleaned_text:
                break
            cleaned_text = new_text
            
        return cleaned_text
    
    def process(self, text: str, **kwargs) -> str:
        """
        Remove duplicates from summary text
        
        Args:
            text: Input summary text
            
        Returns:
            Cleaned summary without duplicates
        """
        if not self.enabled:
            return text
        
        # 반복 구문 먼저 제거
        text = self._remove_repeated_phrases(text)
        
        # 그 다음 중복 문장 제거
        text = self._remove_duplicate_sentences(text)
        
        return text.strip()


class LengthOptimizer(PostProcessor):
    """Optimize summary length to fall within target range"""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.min_tokens = self.config.get('min_tokens', 50)
        self.max_tokens = self.config.get('max_tokens', 100)
        self.target_tokens = self.config.get('target_tokens', 75)
    
    def _count_tokens(self, text: str) -> int:
        """Count approximate number of tokens"""
        # 한국어를 위한 간단한 토큰화
        # 더 정확하려면 실제 토크나이저를 사용해야 함
        tokens = text.split()
        return len(tokens)
    
    def _expand_summary(self, text: str, dialogue: str = None) -> str:
        """Expand summary if too short"""
        current_length = self._count_tokens(text)
        
        if current_length >= self.min_tokens:
            return text
        
        # 필요시 표준 확장 문구 추가
        expansion_templates = [
            "이 대화는 {}에 관한 내용입니다.",
            "주요 논의 사항은 {}입니다.",
            "대화 참여자들은 {}에 대해 이야기했습니다."
        ]
        
        # 가능하면 대화에서 주제 추출 시도
        if dialogue:
            # 간단한 주제 추출 (개선 가능)
            if "회의" in dialogue:
                topic = "회의 일정 및 안건"
            elif "전화" in dialogue or "연락" in dialogue:
                topic = "연락처 교환"
            elif "주문" in dialogue or "예약" in dialogue:
                topic = "예약 및 주문 사항"
            else:
                topic = "일반적인 대화 내용"
            
            expansion = expansion_templates[0].format(topic)
            expanded_text = text + " " + expansion
            
            return expanded_text
        
        return text
    
    def _truncate_summary(self, text: str) -> str:
        """Truncate summary if too long"""
        current_length = self._count_tokens(text)
        
        if current_length <= self.max_tokens:
            return text
        
        # 문장으로 분할
        sentences = re.split(r'[.!?]\s*', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # 제한에 도달할 때까지 문장 추가
        truncated = []
        token_count = 0
        
        for sent in sentences:
            sent_tokens = self._count_tokens(sent)
            if token_count + sent_tokens <= self.max_tokens:
                truncated.append(sent)
                token_count += sent_tokens
            else:
                # 목표보다 너무 적으면 부분 문장 포함 시도
                if token_count < self.min_tokens and truncated:
                    # 문장 단축
                    words = sent.split()
                    remaining_tokens = self.max_tokens - token_count
                    if remaining_tokens > 5:  # 최소한 몇 개 단어는 포함
                        truncated.append(' '.join(words[:remaining_tokens]) + '...')
                break
        
        result = '. '.join(truncated)
        if not result.endswith('.') and not result.endswith('...'):
            result += '.'
            
        return result
    
    def process(self, text: str, dialogue: str = None, **kwargs) -> str:
        """
        Optimize summary length
        
        Args:
            text: Input summary text
            dialogue: Original dialogue (optional, for expansion)
            
        Returns:
            Length-optimized summary
        """
        if not self.enabled:
            return text
        
        # 너무 짧으면 먼저 확장
        text = self._expand_summary(text, dialogue)
        
        # 너무 길면 잘라내기
        text = self._truncate_summary(text)
        
        return text.strip()


class SpecialTokenValidator(PostProcessor):
    """Ensure special tokens from input are preserved in output"""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.special_tokens = self.config.get('special_tokens', [
            '#Person1#', '#Person2#', '#Person3#',
            '#PhoneNumber#', '#Address#', '#DateOfBirth#',
            '#PassportNumber#', '#SSN#', '#CardNumber#',
            '#CarNumber#', '#Email#'
        ])
        self.copy_missing_tokens = self.config.get('copy_missing_tokens', True)
    
    def _extract_tokens(self, text: str) -> Set[str]:
        """Extract special tokens from text"""
        tokens = set()
        for token in self.special_tokens:
            if token in text:
                tokens.add(token)
        return tokens
    
    def _find_token_context(self, dialogue: str, token: str) -> Optional[str]:
        """Find context around a special token in dialogue"""
        if token not in dialogue:
            return None
        
        # 토큰을 찾고 주변 컨텍스트 추출
        index = dialogue.find(token)
        start = max(0, index - 30)
        end = min(len(dialogue), index + len(token) + 30)
        
        context = dialogue[start:end].strip()
        
        # 컨텍스트 정리
        context = re.sub(r'#Person\d+#:', '', context).strip()
        
        return context
    
    def process(self, text: str, dialogue: str = None, **kwargs) -> str:
        """
        Validate and restore special tokens
        
        Args:
            text: Summary text
            dialogue: Original dialogue
            
        Returns:
            Summary with validated/restored special tokens
        """
        if not self.enabled or not dialogue:
            return text
        
        # 두 텍스트에서 토큰 추출
        dialogue_tokens = self._extract_tokens(dialogue)
        summary_tokens = self._extract_tokens(text)
        
        # 누락된 토큰 찾기
        missing_tokens = dialogue_tokens - summary_tokens
        
        if not missing_tokens:
            return text
        
        logger.info(f"Missing special tokens in summary: {missing_tokens}")
        
        if not self.copy_missing_tokens:
            return text
        
        # 컨텍스트와 함께 누락된 토큰 추가 시도
        for token in missing_tokens:
            context = self._find_token_context(dialogue, token)
            
            if context:
                # 토큰을 추가할 위치 결정
                if "전화" in context or "번호" in context and token == "#PhoneNumber#":
                    addition = f" 전화번호는 {token}입니다."
                elif "주소" in context and token == "#Address#":
                    addition = f" 주소는 {token}입니다."
                elif "이메일" in context and token == "#Email#":
                    addition = f" 이메일은 {token}입니다."
                else:
                    # 일반적인 추가
                    addition = f" {token}"
                
                text = text.rstrip('.') + addition
                
                if not text.endswith('.'):
                    text += '.'
        
        return text.strip()


class RuleBasedPostProcessor:
    """Main postprocessor that combines all techniques"""
    
    def __init__(self, config: Dict):
        """
        Initialize the postprocessor pipeline
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # 개별 프로세서 초기화
        self.processors = []
        
        if config.get('duplicate_removal', {}).get('enabled', True):
            self.processors.append(
                DuplicateRemover(config.get('duplicate_removal', {}))
            )
        
        if config.get('length_optimization', {}).get('enabled', True):
            self.processors.append(
                LengthOptimizer(config.get('length_optimization', {}))
            )
        
        if config.get('token_validation', {}).get('enabled', True):
            self.processors.append(
                SpecialTokenValidator(config.get('token_validation', {}))
            )
        
        logger.info(f"Initialized {len(self.processors)} postprocessors")
    
    def process_batch(self, summaries: List[str], dialogues: List[str] = None) -> List[str]:
        """
        Process a batch of summaries
        
        Args:
            summaries: List of summary texts
            dialogues: List of corresponding dialogues (optional)
            
        Returns:
            List of postprocessed summaries
        """
        if not dialogues:
            dialogues = [None] * len(summaries)
        
        processed_summaries = []
        
        for summary, dialogue in zip(summaries, dialogues):
            processed = self.process_single(summary, dialogue)
            processed_summaries.append(processed)
        
        return processed_summaries
    
    def process_single(self, summary: str, dialogue: str = None) -> str:
        """
        Process a single summary through all postprocessors
        
        Args:
            summary: Summary text
            dialogue: Original dialogue (optional)
            
        Returns:
            Postprocessed summary
        """
        processed = summary
        
        for processor in self.processors:
            try:
                processed = processor.process(processed, dialogue=dialogue)
            except Exception as e:
                logger.error(f"Error in {processor.__class__.__name__}: {e}")
                # 다른 프로세서들과 계속 진행
        
        return processed


# 사용 예제 및 테스트
if __name__ == "__main__":
    # 테스트 설정
    config = {
        'duplicate_removal': {
            'enabled': True,
            'min_phrase_length': 3,
            'similarity_threshold': 0.85
        },
        'length_optimization': {
            'enabled': True,
            'min_tokens': 20,
            'max_tokens': 50,
            'target_tokens': 35
        },
        'token_validation': {
            'enabled': True,
            'copy_missing_tokens': True
        }
    }
    
    # 테스트 샘플
    test_summary = "회의는 오후 3시입니다. 회의는 오후 3시입니다. 장소는 회의실입니다."
    test_dialogue = """#Person1#: 회의 시간이 언제인가요?
#Person2#: 오후 3시입니다. 장소는 #Address# 건물 회의실입니다.
#Person1#: 알겠습니다. #PhoneNumber#로 연락드리겠습니다."""
    
    # 개별 프로세서 테스트
    print("Original summary:", test_summary)
    print("\n1. Duplicate Removal Test:")
    dup_remover = DuplicateRemover(config['duplicate_removal'])
    cleaned = dup_remover.process(test_summary)
    print("After duplicate removal:", cleaned)
    
    print("\n2. Length Optimization Test:")
    # 짧은 요약으로 테스트
    short_summary = "회의 확인."
    length_opt = LengthOptimizer(config['length_optimization'])
    expanded = length_opt.process(short_summary, test_dialogue)
    print("After expansion:", expanded)
    
    # 긴 요약으로 테스트
    long_summary = " ".join(["이것은 매우 긴 요약문입니다."] * 10)
    truncated = length_opt.process(long_summary)
    print("After truncation:", truncated)
    
    print("\n3. Token Validation Test:")
    token_validator = SpecialTokenValidator(config['token_validation'])
    validated = token_validator.process(cleaned, test_dialogue)
    print("After token validation:", validated)
    
    print("\n4. Full Pipeline Test:")
    processor = RuleBasedPostProcessor(config)
    final = processor.process_single(test_summary, test_dialogue)
    print("Final result:", final)
