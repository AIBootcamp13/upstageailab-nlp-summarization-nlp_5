"""
eenzeenee T5 모델을 위한 유틸리티 함수들

eenzeenee/t5-base-korean-summarization 모델의 특수 처리를 위한 함수들
"""

import logging
from typing import Dict, Any, Optional
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


def check_and_fix_special_tokens(tokenizer: PreTrainedTokenizer, 
                               special_tokens: list,
                               model_name: str) -> PreTrainedTokenizer:
    """
    특수 토큰이 토크나이저의 vocab_size를 초과하지 않도록 확인하고 수정
    
    Args:
        tokenizer: 토크나이저
        special_tokens: 추가할 특수 토큰 리스트
        model_name: 모델 이름
        
    Returns:
        수정된 토크나이저
    """
    if "eenzeenee" in model_name.lower():
        logger.info(f"eenzeenee 모델 감지: 특수 토큰 처리를 조정합니다.")
        
        # 현재 vocab size 확인
        current_vocab_size = len(tokenizer)
        logger.info(f"현재 vocab size: {current_vocab_size}")
        
        # 안전한 특수 토큰만 추가
        safe_special_tokens = []
        for token in special_tokens:
            if token not in tokenizer.get_vocab():
                safe_special_tokens.append(token)
        
        if safe_special_tokens:
            # 특수 토큰을 조심스럽게 추가
            tokenizer.add_special_tokens({
                'additional_special_tokens': safe_special_tokens[:5]  # 최대 5개만
            })
            logger.info(f"안전하게 {len(safe_special_tokens[:5])}개의 특수 토큰 추가")
        
        # 패딩 토큰 확인 및 설정
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("패딩 토큰을 EOS 토큰으로 설정")
            
    return tokenizer


def get_safe_max_length(model_name: str, config: Dict[str, Any]) -> Dict[str, int]:
    """
    모델에 안전한 최대 길이 설정 반환
    
    Args:
        model_name: 모델 이름
        config: 설정 딕셔너리
        
    Returns:
        안전한 최대 길이 설정
    """
    if "eenzeenee" in model_name.lower():
        # eenzeenee 모델은 더 짧은 시퀀스로 제한
        return {
            'encoder_max_len': min(config.get('tokenizer', {}).get('encoder_max_len', 512), 256),
            'decoder_max_len': min(config.get('tokenizer', {}).get('decoder_max_len', 84), 64)
        }
    
    return {
        'encoder_max_len': config.get('tokenizer', {}).get('encoder_max_len', 512),
        'decoder_max_len': config.get('tokenizer', {}).get('decoder_max_len', 84)
    }


def preprocess_for_eenzeenee(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    eenzeenee 모델을 위한 데이터 전처리
    
    Args:
        data: 입력 데이터
        
    Returns:
        전처리된 데이터
    """
    # 텍스트 정규화
    if 'dialogue' in data:
        data['dialogue'] = data['dialogue'].strip()
        # 너무 긴 대화는 자르기
        if len(data['dialogue']) > 2000:
            data['dialogue'] = data['dialogue'][:2000] + "..."
            
    if 'summary' in data:
        data['summary'] = data['summary'].strip()
        # 너무 긴 요약은 자르기
        if len(data['summary']) > 300:
            data['summary'] = data['summary'][:300] + "..."
            
    return data
