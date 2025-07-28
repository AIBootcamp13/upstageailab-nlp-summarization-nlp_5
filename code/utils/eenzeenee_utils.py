"""
eenzeenee T5 한국어 요약 모델 전용 유틸리티

eenzeenee/t5-base-korean-summarization 모델을 위한 전처리, 설정, 메타정보 제공 기능을 담당합니다.
paust/pko-t5-base를 기반으로 한국어 대화 요약 태스크에 최적화되어 있습니다.
"""

import re
import logging
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass


logger = logging.getLogger(__name__)

# eenzeenee T5 한국어 요약 모델 상수
EENZEENEE_MODEL_NAME = "eenzeenee/t5-base-korean-summarization"


def eenzeenee_whitespace_handler(text: str) -> str:
    """
    eenzeenee 모델용 공백 및 줄바꿈 정규화 함수
    
    한국어 텍스트 특성을 고려한 공백 정규화를 수행합니다.
    연속된 줄바꿈을 공백으로 변환하고, 연속된 공백을 단일 공백으로 통합합니다.
    
    Args:
        text (str): 정규화할 입력 텍스트
        
    Returns:
        str: 공백이 정규화된 텍스트
        
    Example:
        >>> text = "안녕하세요.\\n\\n오늘 날씨가   좋네요."
        >>> eenzeenee_whitespace_handler(text)
        "안녕하세요. 오늘 날씨가 좋네요."
    """
    if not isinstance(text, str) or not text:
        return str(text) if text else ""
    
    # 줄바꿈을 공백으로 변환 후 연속 공백을 단일 공백으로 통합
    normalized = re.sub(r'\s+', ' ', re.sub(r'\n+', ' ', text.strip()))
    return normalized


def get_eenzeenee_generation_config() -> Dict[str, Any]:
    """
    eenzeenee T5 모델의 최적화된 생성 설정 반환
    
    Hugging Face 모델 카드에서 제공하는 최적 파라미터를 기반으로 합니다.
    한국어 요약에 특화된 설정값들을 제공합니다.
    
    Returns:
        Dict[str, Any]: 생성 설정 딕셔너리
        
    Example:
        >>> config = get_eenzeenee_generation_config()
        >>> print(config['max_length'])  # 64
        >>> print(config['num_beams'])   # 3
    """
    return {
        "max_length": 64,           # 한국어 요약 최적 길이 (모델 카드 권장)
        "min_length": 10,           # 최소 요약 길이
        "num_beams": 3,             # 빔 서치 크기 (모델 카드 권장)
        "do_sample": True,          # 샘플링 활성화
        "early_stopping": True,     # 조기 종료
        "no_repeat_ngram_size": 2,  # 반복 방지
        "length_penalty": 1.0,      # 길이 패널티
        "repetition_penalty": 1.2,  # 반복 패널티
        "temperature": 0.8,         # 생성 온도
        "top_k": 50,               # Top-K 샘플링
        "top_p": 0.95,             # Top-P 샘플링
        "pad_token_id": 0,          # 패딩 토큰 ID
        "eos_token_id": 1,          # 종료 토큰 ID
        "forced_bos_token_id": None # BOS 토큰 강제 없음
    }


def get_eenzeenee_tokenizer_config() -> Dict[str, Any]:
    """
    eenzeenee T5 모델의 토크나이저 설정 반환
    
    T5 아키텍처 기반의 인코더-디코더 모델에 최적화된 토크나이저 설정을 제공합니다.
    
    Returns:
        Dict[str, Any]: 토크나이저 설정 딕셔너리
        
    Example:
        >>> config = get_eenzeenee_tokenizer_config()
        >>> print(config['encoder_max_len'])  # 512
        >>> print(config['decoder_max_len'])  # 64
    """
    return {
        "encoder_max_len": 512,     # 입력 텍스트 최대 토큰 수 (모델 카드 권장)
        "decoder_max_len": 64,      # 출력 요약 최대 토큰 수
        "padding": True,            # 패딩 활성화
        "truncation": True,         # 자르기 활성화
        "return_tensors": "pt",     # PyTorch 텐서 반환
        "add_special_tokens": True, # 특수 토큰 추가
        "bos_token": "<pad>",       # T5 기본 BOS 토큰
        "eos_token": "</s>",        # T5 기본 EOS 토큰
        "unk_token": "<unk>",       # 미지 토큰
        "pad_token": "<pad>",       # 패딩 토큰
        "model_max_length": 512     # 모델 최대 길이
    }


def preprocess_for_eenzeenee(text: str, add_prefix: bool = True) -> str:
    """
    eenzeenee 모델용 텍스트 전처리 함수
    
    한국어 대화 텍스트에 특화된 전처리를 수행합니다.
    필수적으로 'summarize: ' prefix를 추가하고 공백을 정규화합니다.
    
    Args:
        text (str): 전처리할 원본 텍스트
        add_prefix (bool): prefix 추가 여부 (기본값: True)
        
    Returns:
        str: 전처리된 텍스트
        
    Example:
        >>> text = "안녕하세요. 오늘 회의에서..."
        >>> preprocessed = preprocess_for_eenzeenee(text)
        >>> print(preprocessed)  # "summarize: 안녕하세요. 오늘 회의에서..."
    """
    if not isinstance(text, str) or not text:
        return "summarize: " if add_prefix else ""
    
    # 공백 정규화
    normalized_text = eenzeenee_whitespace_handler(text)
    
    # prefix 추가 (eenzeenee 모델 필수)
    if add_prefix and not normalized_text.startswith("summarize: "):
        return f"summarize: {normalized_text}"
    
    return normalized_text


def get_eenzeenee_model_info() -> Dict[str, Any]:
    """
    eenzeenee T5 모델의 상세 정보 반환
    
    모델의 메타데이터, 성능 지표, 설정 정보 등을 종합적으로 제공합니다.
    
    Returns:
        Dict[str, Any]: 모델 정보 딕셔너리
        
    Example:
        >>> info = get_eenzeenee_model_info()
        >>> print(info['architecture'])     # 'T5-base'
        >>> print(info['parameters'])       # '220M'
        >>> print(info['language'])         # 'Korean'
    """
    return {
        "model_name": EENZEENEE_MODEL_NAME,
        "model_type": "seq2seq",
        "architecture": "T5-base",
        "base_model": "paust/pko-t5-base",
        "parameters": "220M",
        "language": "Korean",
        "task": "Text Summarization",
        "prefix": "summarize: ",
        "input_max_length": 512,
        "output_max_length": 64,
        "recommended_batch_size": 8,
        "supports_generate": True,
        "model_class": "AutoModelForSeq2SeqLM",
        "tokenizer_class": "AutoTokenizer",
        "training_datasets": [
            "Korean Paper Summarization Dataset (논문자료 요약)",
            "Korean Book Summarization Dataset (도서자료 요약)", 
            "Korean Summary statement and Report Generation Dataset (요약문 및 레포트 생성 데이터)"
        ],
        "performance_metrics": {
            "paper_summarization": {
                "rouge_2_r": 0.0987,
                "rouge_2_p": 0.9667,
                "rouge_2_f": 0.1725
            },
            "book_summarization": {
                "rouge_2_r": 0.1576,
                "rouge_2_p": 0.9718,
                "rouge_2_f": 0.2655
            },
            "report_generation": {
                "rouge_2_r": 0.0988,
                "rouge_2_p": 0.9277,
                "rouge_2_f": 0.1773
            }
        },
        "recommended_settings": {
            "num_beams": 3,
            "do_sample": True,
            "min_length": 10,
            "max_length": 64,
            "temperature": 0.8,
            "top_k": 50,
            "top_p": 0.95
        }
    }


def is_eenzeenee_compatible_model(model_name: str) -> bool:
    """
    주어진 모델명이 eenzeenee 계열 모델인지 확인
    
    Args:
        model_name (str): 확인할 모델명
        
    Returns:
        bool: eenzeenee 모델 여부
        
    Example:
        >>> is_eenzeenee_compatible_model("eenzeenee/t5-base-korean-summarization")
        True
        >>> is_eenzeenee_compatible_model("google/t5-base")
        False
    """
    if not isinstance(model_name, str):
        return False
    
    model_name_lower = model_name.lower()
    
    # eenzeenee 모델 패턴 확인
    eenzeenee_patterns = [
        'eenzeenee',
        't5-base-korean-summarization',
        'eenzeenee/t5-base',
        'eenzeenee/xsum-t5'  # 이전 명명 호환성
    ]
    
    return any(pattern in model_name_lower for pattern in eenzeenee_patterns)


def get_eenzeenee_preprocessing_prompt() -> str:
    """
    eenzeenee 모델 사용시 권장하는 프롬프트 안내
    
    Returns:
        str: 전처리 프롬프트 안내 문자열
    """
    return """
    eenzeenee T5 한국어 요약 모델 사용 가이드:
    
    1. 입력 텍스트에 반드시 'summarize: ' prefix를 추가하세요
    2. 최대 입력 길이: 512 토큰
    3. 권장 출력 길이: 10-64 토큰
    4. 한국어 텍스트에 최적화되어 있습니다
    5. 대화형 텍스트, 논문, 도서 요약에 특화되어 있습니다
    
    사용 예시:
    input_text = "summarize: " + your_korean_text
    """


def validate_eenzeenee_input(text: str) -> Dict[str, Any]:
    """
    eenzeenee 모델 입력 검증 함수
    
    Args:
        text (str): 검증할 입력 텍스트
        
    Returns:
        Dict[str, Any]: 검증 결과
    """
    result = {
        "is_valid": True,
        "warnings": [],
        "errors": [],
        "suggestions": []
    }
    
    if not isinstance(text, str):
        result["is_valid"] = False
        result["errors"].append("입력이 문자열이 아닙니다")
        return result
    
    if not text.strip():
        result["is_valid"] = False
        result["errors"].append("입력 텍스트가 비어있습니다")
        return result
    
    # prefix 확인
    if not text.startswith("summarize: "):
        result["warnings"].append("'summarize: ' prefix가 없습니다")
        result["suggestions"].append("preprocess_for_eenzeenee() 함수를 사용하세요")
    
    # 길이 확인
    if len(text) > 2048:  # 대략적인 토큰 길이 추정
        result["warnings"].append("입력 텍스트가 너무 깁니다 (권장: 512 토큰)")
        result["suggestions"].append("텍스트를 분할하거나 요약해보세요")
    
    # 한국어 포함 확인
    korean_chars = re.findall(r'[가-힣]', text)
    if len(korean_chars) < 10:
        result["warnings"].append("한국어 텍스트가 부족합니다")
        result["suggestions"].append("이 모델은 한국어에 특화되어 있습니다")
    
    return result


# 편의 함수들
def create_eenzeenee_inputs(texts: List[str], tokenizer) -> Dict[str, Any]:
    """
    eenzeenee 모델용 입력 배치 생성
    
    Args:
        texts (List[str]): 처리할 텍스트 리스트
        tokenizer: 토크나이저 객체
        
    Returns:
        Dict[str, Any]: 토크나이저 출력
    """
    # prefix 추가 및 전처리
    processed_texts = [preprocess_for_eenzeenee(text) for text in texts]
    
    # 토크나이저 설정 적용
    tokenizer_config = get_eenzeenee_tokenizer_config()
    
    return tokenizer(
        processed_texts,
        max_length=tokenizer_config["encoder_max_len"],
        padding=tokenizer_config["padding"],
        truncation=tokenizer_config["truncation"],
        return_tensors=tokenizer_config["return_tensors"]
    )


def get_eenzeenee_special_tokens() -> Dict[str, str]:
    """
    eenzeenee 모델에서 사용하는 특수 토큰들 반환
    
    Returns:
        Dict[str, str]: 특수 토큰 딕셔너리
    """
    return {
        "pad_token": "<pad>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "bos_token": "<pad>",  # T5는 BOS로 PAD 사용
        "prefix": "summarize: ",
        "additional_special_tokens": [
            "#Person1#", "#Person2#", "#Person3#",
            "#PhoneNumber#", "#Address#", "#PassportNumber#",
            "#DateOfBirth#", "#SSN#", "#CardNumber#",
            "#CarNumber#", "#Email#"
        ]
    }


# 모듈 레벨 검증
def _validate_module():
    """모듈 임포트 시 기본 검증"""
    try:
        info = get_eenzeenee_model_info()
        logger.info(f"eenzeenee_utils 모듈 로드 완료: {info['model_name']}")
        return True
    except Exception as e:
        logger.error(f"eenzeenee_utils 모듈 검증 실패: {e}")
        return False


# 모듈 로드 시 검증 실행
_validate_module()
