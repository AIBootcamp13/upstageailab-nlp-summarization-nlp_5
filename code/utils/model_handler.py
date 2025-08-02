"""
모델별 처리 핸들러

다양한 모델 아키텍처에 대한 통합 처리
"""

from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class ModelSpecificHandler:
    """모델별 특성 처리 핸들러"""
    
    # 모델별 설정
    MODEL_CONFIGS = {
        "mt5": {
            "model_class": "T5ForConditionalGeneration",
            "tokenizer_class": "T5Tokenizer",
            "use_prefix": True,
            "default_prefix": "dialogue summarization in korean: ",
            "architecture": "t5"
        },
        "t5": {
            "model_class": "T5ForConditionalGeneration", 
            "tokenizer_class": "T5Tokenizer",
            "use_prefix": True,
            "default_prefix": "summarize: ",
            "architecture": "t5"
        },
        "bart": {
            "model_class": "BartForConditionalGeneration",
            "tokenizer_class": "BartTokenizer",
            "use_prefix": False,
            "default_prefix": "",
            "architecture": "bart"
        },
        "kobart": {
            "model_class": "BartForConditionalGeneration",
            "tokenizer_class": "PreTrainedTokenizer",
            "use_prefix": False,
            "default_prefix": "",
            "architecture": "bart"
        }
    }
    
    @classmethod
    def detect_model_type(cls, model_name: str) -> str:
        """모델 이름에서 타입 추론"""
        model_name_lower = model_name.lower()
        
        if "mt5" in model_name_lower:
            return "mt5"
        elif "t5" in model_name_lower:
            return "t5"
        elif "kobart" in model_name_lower:
            return "kobart"
        elif "bart" in model_name_lower:
            return "bart"
        else:
            logger.warning(f"Unknown model type for {model_name}, defaulting to bart")
            return "bart"
    
    @classmethod
    def get_model_config(cls, model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """모델별 최적 설정 반환"""
        # 모델 타입 감지
        model_type = cls.detect_model_type(model_name)
        
        # 기본 설정 가져오기
        model_config = cls.MODEL_CONFIGS.get(model_type, cls.MODEL_CONFIGS["bart"]).copy()
        
        # 사용자 설정으로 업데이트
        if "input_prefix" in config:
            model_config["prefix"] = config["input_prefix"]
        else:
            model_config["prefix"] = model_config["default_prefix"]
        
        # 토크나이저 설정
        model_config["encoder_max_len"] = config.get("tokenizer", {}).get("encoder_max_len", 1024)
        model_config["decoder_max_len"] = config.get("tokenizer", {}).get("decoder_max_len", 200)
        
        # 생성 설정
        model_config["num_beams"] = config.get("inference", {}).get("num_beams", 4)
        model_config["no_repeat_ngram_size"] = config.get("inference", {}).get("no_repeat_ngram_size", 2)
        model_config["early_stopping"] = config.get("inference", {}).get("early_stopping", True)
        model_config["max_length"] = config.get("inference", {}).get("generate_max_length", 100)
        
        return model_config
    
    @classmethod
    def load_model_for_inference(cls, model_path: str, model_config: Dict[str, Any], device):
        """모델 로드 (아키텍처별 처리)"""
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BartForConditionalGeneration
        
        architecture = model_config.get("architecture", "bart")
        
        # 원본 모델명 추출 (체크포인트가 아닌 원본 모델에서 토크나이저 로드)
        # model_path가 체크포인트인 경우 config.json에서 원본 모델명 추출
        import json
        import os
        
        original_model_name = model_path
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                model_config_json = json.load(f)
                # _name_or_path 필드에서 원본 모델명 추출
                if "_name_or_path" in model_config_json:
                    original_model_name = model_config_json["_name_or_path"]
        
        # 토크나이저 로드 (원본 모델에서)
        try:
            tokenizer = AutoTokenizer.from_pretrained(original_model_name)
        except:
            # 실패 시 체크포인트에서 로드
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 모델 로드 (체크포인트에서)
        if architecture == "t5":
            from transformers import T5ForConditionalGeneration
            model = T5ForConditionalGeneration.from_pretrained(model_path)
        else:  # bart
            model = BartForConditionalGeneration.from_pretrained(model_path)
        
        # 디바이스 설정
        model = model.to(device)
        model.eval()
        
        return model, tokenizer
    
    @classmethod
    def prepare_input_with_prefix(cls, texts: List[str], model_config: Dict[str, Any]) -> List[str]:
        """모델별 prefix 처리"""
        if model_config.get("use_prefix", False):
            prefix = model_config.get("prefix", "")
            if prefix:
                return [f"{prefix}{text}" for text in texts]
        return texts
