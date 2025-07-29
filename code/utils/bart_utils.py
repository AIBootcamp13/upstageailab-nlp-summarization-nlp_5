"""
BART 모델을 위한 커스텀 DataCollator

BART 모델은 token_type_ids를 사용하지 않으므로 이를 제외한 DataCollator가 필요합니다.
"""

from typing import Dict, List, Any, Optional, Union
import torch
from transformers import DataCollatorForSeq2Seq


class DataCollatorForBart(DataCollatorForSeq2Seq):
    """
    BART 모델 전용 DataCollator
    
    BART는 token_type_ids를 사용하지 않으므로 이를 제거합니다.
    """
    
    def __call__(self, features: List[Dict[str, Any]], 
                 return_tensors: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        DataCollator 호출
        
        Args:
            features: 입력 feature 리스트
            return_tensors: 반환 텐서 타입
            
        Returns:
            처리된 배치 데이터
        """
        # 부모 클래스의 __call__ 메서드 호출
        batch = super().__call__(features, return_tensors)
        
        # BART는 token_type_ids를 사용하지 않으므로 제거
        if "token_type_ids" in batch:
            del batch["token_type_ids"]
        
        # decoder_input_ids에서도 token_type_ids 제거 (있다면)
        if "decoder_token_type_ids" in batch:
            del batch["decoder_token_type_ids"]
            
        return batch


class SmartDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    """
    모델 타입에 따라 자동으로 token_type_ids 처리를 조정하는 DataCollator
    """
    
    def __init__(self, tokenizer, model=None, **kwargs):
        super().__init__(tokenizer, model, **kwargs)
        
        # 모델 타입 확인
        self.model_type = None
        if model is not None:
            model_class_name = model.__class__.__name__
            if "Bart" in model_class_name or "bart" in model_class_name.lower():
                self.model_type = "bart"
            elif "T5" in model_class_name or "t5" in model_class_name.lower():
                self.model_type = "t5"
            elif "MT5" in model_class_name or "mt5" in model_class_name.lower():
                self.model_type = "mt5"
                
    def __call__(self, features: List[Dict[str, Any]], 
                 return_tensors: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        모델 타입에 따라 적절히 처리된 배치 반환
        """
        batch = super().__call__(features, return_tensors)
        
        # BART 모델인 경우 token_type_ids 제거
        if self.model_type == "bart":
            if "token_type_ids" in batch:
                del batch["token_type_ids"]
            if "decoder_token_type_ids" in batch:
                del batch["decoder_token_type_ids"]
                
        return batch
