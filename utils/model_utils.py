"""
모델과 토크나이저 호환성을 위한 유틸리티 함수들
"""

import logging
import torch
from typing import Optional, Union
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


def resize_model_embeddings(model: PreTrainedModel, 
                          tokenizer: PreTrainedTokenizer,
                          pad_to_multiple_of: Optional[int] = None) -> PreTrainedModel:
    """
    모델의 embedding 크기를 토크나이저의 vocab size에 맞게 조정
    
    Args:
        model: 조정할 모델
        tokenizer: 토크나이저
        pad_to_multiple_of: embedding 크기를 이 값의 배수로 패딩
        
    Returns:
        조정된 모델
    """
    vocab_size = len(tokenizer)
    model_vocab_size = model.get_input_embeddings().weight.shape[0]
    
    if vocab_size != model_vocab_size:
        logger.info(f"토크나이저 vocab size ({vocab_size})와 모델 vocab size ({model_vocab_size})가 다릅니다.")
        logger.info(f"모델 embedding을 리사이징합니다...")
        
        # embedding 리사이징
        model.resize_token_embeddings(vocab_size, pad_to_multiple_of=pad_to_multiple_of)
        
        # 새로운 vocab size 확인
        new_vocab_size = model.get_input_embeddings().weight.shape[0]
        logger.info(f"✅ 모델 vocab size가 {new_vocab_size}로 조정되었습니다.")
    else:
        logger.info(f"✅ 토크나이저와 모델의 vocab size가 이미 일치합니다: {vocab_size}")
    
    return model


def check_special_tokens_in_vocab(tokenizer: PreTrainedTokenizer, 
                                special_tokens: list) -> dict:
    """
    특수 토큰이 vocab에 있는지 확인
    
    Args:
        tokenizer: 토크나이저
        special_tokens: 확인할 특수 토큰 리스트
        
    Returns:
        토큰별 존재 여부와 인덱스
    """
    results = {}
    vocab = tokenizer.get_vocab()
    
    for token in special_tokens:
        if token in vocab:
            results[token] = {
                'exists': True,
                'index': vocab[token]
            }
        else:
            results[token] = {
                'exists': False,
                'index': None
            }
    
    return results


def safe_add_special_tokens(tokenizer: PreTrainedTokenizer,
                          model: PreTrainedModel,
                          special_tokens: list,
                          model_name: str) -> tuple:
    """
    안전하게 특수 토큰을 추가하고 모델을 조정
    
    Args:
        tokenizer: 토크나이저
        model: 모델
        special_tokens: 추가할 특수 토큰 리스트
        model_name: 모델 이름
        
    Returns:
        (조정된 토크나이저, 조정된 모델)
    """
    # 현재 상태 확인
    logger.info(f"🔍 특수 토큰 추가 전 상태:")
    logger.info(f"   토크나이저 vocab size: {len(tokenizer)}")
    logger.info(f"   모델 vocab size: {model.get_input_embeddings().weight.shape[0]}")
    
    # 기존에 없는 토큰만 필터링
    new_tokens = [token for token in special_tokens if token not in tokenizer.get_vocab()]
    
    if new_tokens:
        logger.info(f"📝 {len(new_tokens)}개의 새로운 특수 토큰 추가: {new_tokens[:5]}...")
        
        # 토크나이저에 토큰 추가
        num_added = tokenizer.add_tokens(new_tokens)
        logger.info(f"✅ {num_added}개의 토큰이 토크나이저에 추가되었습니다.")
        
        # 모델 embedding 리사이징
        model = resize_model_embeddings(model, tokenizer)
    else:
        logger.info("✅ 모든 특수 토큰이 이미 vocab에 있습니다.")
    
    # 최종 상태 확인
    logger.info(f"🔍 특수 토큰 추가 후 상태:")
    logger.info(f"   토크나이저 vocab size: {len(tokenizer)}")
    logger.info(f"   모델 vocab size: {model.get_input_embeddings().weight.shape[0]}")
    
    # 불일치 검사
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        raise ValueError(f"토크나이저와 모델의 vocab size가 여전히 불일치합니다!")
    
    return tokenizer, model


def validate_tokenizer_model_compatibility(tokenizer: PreTrainedTokenizer,
                                         model: PreTrainedModel) -> bool:
    """
    토크나이저와 모델의 호환성 검증
    
    Args:
        tokenizer: 토크나이저
        model: 모델
        
    Returns:
        호환 여부
    """
    vocab_size = len(tokenizer)
    model_vocab_size = model.get_input_embeddings().weight.shape[0]
    
    if vocab_size > model_vocab_size:
        logger.error(f"❌ 토크나이저 vocab size ({vocab_size})가 모델 vocab size ({model_vocab_size})보다 큽니다!")
        logger.error("이는 인덱싱 에러를 일으킬 수 있습니다.")
        return False
    
    logger.info(f"✅ 토크나이저와 모델이 호환됩니다.")
    return True
