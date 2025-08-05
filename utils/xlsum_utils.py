"""
mT5 XL-Sum 모델 전용 유틸리티

mT5_multilingual_XLSum 모델을 위한 전처리, 설정, 메타정보 제공 기능을 담당합니다.
Hugging Face 공식 예제를 기반으로 한국어 대화 요약 태스크에 최적화되어 있습니다.
"""

import re
import logging
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass


logger = logging.getLogger(__name__)

# mT5 XL-Sum 모델 상수
XLSUM_MODEL_NAME = "csebuetnlp/mT5_multilingual_XLSum"


def xlsum_whitespace_handler(text: str) -> str:
    """
    XL-Sum 모델용 공백 및 줄바꿈 정규화 함수
    
    Hugging Face 공식 예제의 WHITESPACE_HANDLER 패턴을 기반으로 구현.
    연속된 줄바꿈을 공백으로 변환하고, 연속된 공백을 단일 공백으로 통합합니다.
    
    Args:
        text (str): 정규화할 입력 텍스트
        
    Returns:
        str: 공백이 정규화된 텍스트
        
    Example:
        >>> text = "안녕하세요.\\n\\n오늘 날씨가   좋네요."
        >>> xlsum_whitespace_handler(text)
        "안녕하세요. 오늘 날씨가 좋네요."
    """
    if not isinstance(text, str) or not text:
        return str(text) if text else ""
    
    # 줄바꿈을 공백으로 변환 후 연속 공백을 단일 공백으로 통합
    normalized = re.sub(r'\s+', ' ', re.sub(r'\n+', ' ', text.strip()))
    return normalized


def get_xlsum_generation_config() -> Dict[str, Any]:
    """
    mT5 XL-Sum 모델의 최적화된 생성 설정 반환
    
    Hugging Face 공식 예제에서 제공하는 mT5 모델 최적 파라미터를 기반으로 합니다.
    한국어 요약에 특화된 설정으로 조정되었습니다.
    
    Returns:
        Dict[str, Any]: 생성 설정 딕셔너리
            - max_length: 최대 생성 토큰 수 (84)
            - num_beams: 빔 서치 크기 (4)  
            - no_repeat_ngram_size: 반복 방지 n-gram 크기 (2)
            - do_sample: 샘플링 비활성화 (False)
            - early_stopping: 조기 종료 활성화 (True)
    """
    return {
        "max_length": 84,  # mT5 XL-Sum 권장 요약 길이
        "num_beams": 4,    # Beam search 크기
        "no_repeat_ngram_size": 2,  # 반복 방지
        "do_sample": False,  # 결정론적 생성
        "early_stopping": True,  # 조기 종료
        "length_penalty": 1.0,  # 길이 패널티
    }


def get_xlsum_tokenizer_config() -> Dict[str, Any]:
    """
    mT5 XL-Sum 모델의 토크나이저 설정 반환
    
    입력 텍스트의 최대 길이와 패딩/절단 설정을 제공합니다.
    
    Returns:
        Dict[str, Any]: 토크나이저 설정 딕셔너리
            - max_length: 최대 입력 토큰 수 (512)
            - truncation: 절단 활성화 (True)
            - padding: 패딩 방식 ('max_length')
            - return_tensors: 반환 텐서 타입 ('pt')
    """
    return {
        "max_length": 512,  # mT5 최대 입력 길이
        "truncation": True,  # 길이 초과 시 절단
        "padding": "max_length",  # 최대 길이로 패딩
        "return_tensors": "pt",  # PyTorch 텐서 반환
        "add_special_tokens": True,  # 특수 토큰 추가
    }


def preprocess_for_xlsum(text: str, **kwargs) -> str:
    """
    XL-Sum 모델용 텍스트 전처리
    
    한국어 대화 텍스트를 mT5 모델이 처리하기 적합한 형태로 변환합니다.
    공백 정규화와 기본적인 정제 작업을 수행합니다.
    
    Args:
        text (str): 전처리할 입력 텍스트
        **kwargs: 추가 전처리 옵션 (현재 미사용)
        
    Returns:
        str: 전처리된 텍스트
        
    Example:
        >>> dialogue = "#Person1#: 안녕하세요.\\n#Person2#: 반갑습니다."
        >>> preprocess_for_xlsum(dialogue)
        "#Person1#: 안녕하세요. #Person2#: 반갑습니다."
    """
    if not isinstance(text, str) or not text:
        return str(text) if text else ""
    
    # XL-Sum 공백 정규화 적용
    processed_text = xlsum_whitespace_handler(text)
    
    # 추가 정제 작업 (필요시 확장)
    # - 특수 문자 정규화
    # - 불필요한 공백 제거 등
    
    return processed_text


def get_xlsum_model_info() -> Dict[str, Any]:
    """
    mT5 XL-Sum 모델의 메타정보 반환
    
    모델 아키텍처, 지원 언어, 성능 지표 등의 상세 정보를 제공합니다.
    
    Returns:
        Dict[str, Any]: 모델 메타정보 딕셔너리
    """
    return {
        "model_name": XLSUM_MODEL_NAME,
        "architecture": "mT5",
        "base_model": "google/mt5-base", 
        "task": "abstractive_summarization",
        "languages": 45,  # XL-Sum 지원 언어 수
        "dataset": "XL-Sum",
        "paper_url": "https://aclanthology.org/2021.findings-acl.413/",
        "performance": {
            "korean": {
                "rouge_1": 23.6745,
                "rouge_2": 11.4478, 
                "rouge_l": 22.3619
            },
            "english": {
                "rouge_1": 37.601,
                "rouge_2": 15.1536,
                "rouge_l": 29.8817
            }
        },
        "model_size": "2.17GB",
        "max_input_length": 512,
        "max_output_length": 84,
        "recommended_batch_size": 4,
        "training_info": {
            "dataset_size": "1M+ articles", 
            "training_languages": 45,
            "fine_tuned_from": "mt5-base"
        }
    }


def is_xlsum_compatible_model(model_name: str) -> bool:
    """
    모델이 XL-Sum 호환 모델인지 확인
    
    모델명을 기반으로 mT5 XL-Sum 계열 모델인지 판단합니다.
    
    Args:
        model_name (str): 확인할 모델명
        
    Returns:
        bool: XL-Sum 호환 모델 여부
        
    Example:
        >>> is_xlsum_compatible_model("csebuetnlp/mT5_multilingual_XLSum")
        True
        >>> is_xlsum_compatible_model("google/mt5-base")
        False
    """
    if not isinstance(model_name, str) or not model_name:
        return False
        
    model_name_lower = model_name.lower()
    
    # XL-Sum 관련 키워드 체크 (더 정확한 패턴)
    xlsum_keywords = ["xlsum", "xl-sum", "csebuetnlp/mt5_multilingual_xlsum"]
    
    # mT5 관련 키워드 체크  
    mt5_keywords = ["mt5", "multilingual-t5"]
    
    # 요약 관련 키워드 체크
    summarization_keywords = ["sum", "summariz", "abstract"]
    
    # 정확한 XL-Sum 모델 체크
    has_xlsum = any(keyword in model_name_lower for keyword in xlsum_keywords)
    
    # mT5이면서 요약 관련 모델인지 체크
    has_mt5 = any(keyword in model_name_lower for keyword in mt5_keywords)
    has_summarization = any(keyword in model_name_lower for keyword in summarization_keywords)
    
    return has_xlsum or (has_mt5 and has_summarization)


def get_xlsum_preprocessing_prompt(task_type: str = "summarization") -> str:
    """
    XL-Sum 모델용 전처리 프롬프트 반환
    
    mT5 모델은 기본적으로 프롬프트가 불필요하지만, 
    특정 태스크나 설정에서 필요할 경우를 대비한 함수입니다.
    
    Args:
        task_type (str): 태스크 타입 (기본값: "summarization")
        
    Returns:
        str: 전처리 프롬프트 (현재는 빈 문자열)
        
    Note:
        mT5는 prefix 기반 학습을 하지 않았으므로 별도 프롬프트 불필요
    """
    # mT5 XL-Sum은 프롬프트 없이 직접 입력 처리
    # T5와 달리 "summarize:" 등의 prefix 불필요
    return ""


# 편의 함수들
def get_xlsum_default_config() -> Dict[str, Any]:
    """
    XL-Sum 모델의 기본 설정을 모두 포함한 통합 설정 반환
    
    Returns:
        Dict[str, Any]: 통합 설정 딕셔너리
    """
    return {
        "model": get_xlsum_model_info(),
        "tokenizer": get_xlsum_tokenizer_config(), 
        "generation": get_xlsum_generation_config(),
        "preprocessing": {
            "whitespace_handler": xlsum_whitespace_handler,
            "text_preprocessor": preprocess_for_xlsum,
            "prompt": get_xlsum_preprocessing_prompt()
        }
    }


def validate_xlsum_input(text: str, max_length: int = 512) -> bool:
    """
    XL-Sum 모델 입력 유효성 검증
    
    Args:
        text (str): 검증할 텍스트
        max_length (int): 최대 허용 길이
        
    Returns:
        bool: 입력 유효성 여부
    """
    if not isinstance(text, str) or not text.strip():
        logger.warning("Empty or invalid input text")
        return False
        
    if len(text) > max_length * 4:  # 대략적인 토큰 수 추정 (1토큰 ≈ 4자)
        logger.warning(f"Input text too long: {len(text)} chars (max ~{max_length * 4})")
        return False
        
    return True


# 모듈 레벨 상수들
__all__ = [
    "XLSUM_MODEL_NAME",
    "xlsum_whitespace_handler", 
    "get_xlsum_generation_config",
    "get_xlsum_tokenizer_config",
    "preprocess_for_xlsum",
    "get_xlsum_model_info", 
    "is_xlsum_compatible_model",
    "get_xlsum_preprocessing_prompt",
    "get_xlsum_default_config",
    "validate_xlsum_input"
]
