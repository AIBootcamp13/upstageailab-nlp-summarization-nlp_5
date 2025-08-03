"""
토큰 범위 검증 모듈

토크나이저와 모델 간의 vocabulary 호환성을 검증하고
토큰 ID 범위 초과 문제를 사전에 감지합니다.
"""

import logging
import torch
from typing import Dict, Any, List, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


class TokenValidator:
    """토큰 범위 및 호환성 검증 클래스"""
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        """
        TokenValidator 초기화
        
        Args:
            model_name: 모델 이름
            config: 실험 설정
        """
        self.model_name = model_name
        self.config = config
        self.tokenizer = None
        self.model = None
        self.validation_results = {}
        
    def load_tokenizer_and_model(self) -> Tuple[bool, str]:
        """
        토크나이저와 모델 로드
        
        Returns:
            (성공 여부, 에러 메시지)
        """
        try:
            logger.info(f"토크나이저 로드 중: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            logger.info(f"모델 정보 로드 중: {self.model_name}")
            # 모델 config만 로드 (메모리 절약)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name, 
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            return True, ""
            
        except Exception as e:
            error_msg = f"모델/토크나이저 로드 실패: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def validate_vocabulary_compatibility(self) -> Dict[str, Any]:
        """
        vocabulary 호환성 검증
        
        Returns:
            검증 결과 딕셔너리
        """
        if not self.tokenizer or not self.model:
            return {"success": False, "error": "토크나이저 또는 모델이 로드되지 않음"}
        
        try:
            # 토크나이저와 모델의 vocabulary 크기 비교
            tokenizer_vocab_size = self.tokenizer.vocab_size
            model_vocab_size = self.model.config.vocab_size
            
            logger.info(f"토크나이저 vocab_size: {tokenizer_vocab_size}")
            logger.info(f"모델 vocab_size: {model_vocab_size}")
            
            # 특수 토큰 검증
            special_tokens = self.config.get('tokenizer', {}).get('special_tokens', [])
            valid_special_tokens = []
            invalid_special_tokens = []
            
            for token in special_tokens:
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                if token_id < model_vocab_size:
                    valid_special_tokens.append((token, token_id))
                else:
                    invalid_special_tokens.append((token, token_id))
            
            # 호환성 판단
            is_compatible = (
                tokenizer_vocab_size <= model_vocab_size and
                len(invalid_special_tokens) == 0
            )
            
            result = {
                "success": True,
                "is_compatible": is_compatible,
                "tokenizer_vocab_size": tokenizer_vocab_size,
                "model_vocab_size": model_vocab_size,
                "vocab_size_match": tokenizer_vocab_size == model_vocab_size,
                "valid_special_tokens": valid_special_tokens,
                "invalid_special_tokens": invalid_special_tokens,
                "special_token_count": len(special_tokens),
                "warning_messages": []
            }
            
            # 경고 메시지 생성
            if tokenizer_vocab_size > model_vocab_size:
                result["warning_messages"].append(
                    f"⚠️ 토크나이저 vocab_size({tokenizer_vocab_size})가 모델 vocab_size({model_vocab_size})보다 큽니다"
                )
            
            if invalid_special_tokens:
                result["warning_messages"].append(
                    f"⚠️ 유효하지 않은 특수 토큰 {len(invalid_special_tokens)}개: {[t[0] for t in invalid_special_tokens]}"
                )
            
            return result
            
        except Exception as e:
            return {"success": False, "error": f"vocabulary 검증 중 오류: {str(e)}"}
    
    def validate_sample_data(self, sample_size: int = 100) -> Dict[str, Any]:
        """
        샘플 데이터로 토큰 범위 검증
        
        Args:
            sample_size: 검증할 샘플 수
            
        Returns:
            검증 결과 딕셔너리
        """
        if not self.tokenizer or not self.model:
            return {"success": False, "error": "토크나이저 또는 모델이 로드되지 않음"}
        
        try:
            # 데이터 로드
            train_path = self.config.get('general', {}).get('train_path', 'data/train.csv')
            if not Path(train_path).exists():
                return {"success": False, "error": f"학습 데이터 파일을 찾을 수 없습니다: {train_path}"}
            
            df = pd.read_csv(train_path)
            if len(df) == 0:
                return {"success": False, "error": "학습 데이터가 비어있습니다"}
            
            # 샘플 선택
            sample_df = df.head(min(sample_size, len(df)))
            
            model_vocab_size = self.model.config.vocab_size
            max_token_id = 0
            min_token_id = float('inf')
            problematic_samples = []
            
            # 각 샘플 검증
            for idx, row in sample_df.iterrows():
                try:
                    # 입력 텍스트 토크나이징
                    input_text = str(row.get('input', row.get('dialogue', '')))
                    target_text = str(row.get('target', row.get('summary', '')))
                    
                    # 모델별 prefix 적용
                    input_prefix = self.config.get('input_prefix', '')
                    if input_prefix:
                        input_text = f"{input_prefix}{input_text}"
                    
                    # 토크나이징
                    input_tokens = self.tokenizer.encode(input_text, add_special_tokens=True)
                    target_tokens = self.tokenizer.encode(target_text, add_special_tokens=True)
                    
                    # 토큰 ID 범위 확인
                    all_tokens = input_tokens + target_tokens
                    sample_max = max(all_tokens) if all_tokens else 0
                    sample_min = min(all_tokens) if all_tokens else 0
                    
                    max_token_id = max(max_token_id, sample_max)
                    min_token_id = min(min_token_id, sample_min)
                    
                    # 범위 초과 토큰 감지
                    invalid_tokens = [t for t in all_tokens if t >= model_vocab_size]
                    if invalid_tokens:
                        problematic_samples.append({
                            "sample_idx": idx,
                            "fname": row.get('fname', f'sample_{idx}'),
                            "invalid_token_ids": invalid_tokens,
                            "max_invalid_id": max(invalid_tokens),
                            "input_length": len(input_tokens),
                            "target_length": len(target_tokens)
                        })
                
                except Exception as e:
                    logger.warning(f"샘플 {idx} 검증 중 오류: {str(e)}")
                    continue
            
            # 결과 생성
            is_valid = len(problematic_samples) == 0 and max_token_id < model_vocab_size
            
            result = {
                "success": True,
                "is_valid": is_valid,
                "samples_checked": len(sample_df),
                "model_vocab_size": model_vocab_size,
                "max_token_id_found": max_token_id,
                "min_token_id_found": min_token_id if min_token_id != float('inf') else 0,
                "problematic_samples_count": len(problematic_samples),
                "problematic_samples": problematic_samples[:10],  # 최대 10개만 표시
                "token_range_exceeded": max_token_id >= model_vocab_size,
                "recommendations": []
            }
            
            # 권장사항 생성
            if max_token_id >= model_vocab_size:
                result["recommendations"].append(
                    f"🚨 토큰 ID 범위 초과 감지! 최대 토큰 ID({max_token_id}) >= 모델 vocab_size({model_vocab_size})"
                )
                result["recommendations"].append(
                    "💡 해결책: 특수 토큰을 모델 vocabulary에 추가하거나 토크나이저 재설정 필요"
                )
            
            if len(problematic_samples) > 0:
                result["recommendations"].append(
                    f"⚠️ {len(problematic_samples)}개 샘플에서 문제 발견. 데이터 전처리 확인 필요"
                )
            
            return result
            
        except Exception as e:
            return {"success": False, "error": f"샘플 데이터 검증 중 오류: {str(e)}"}
    
    def run_full_validation(self) -> Dict[str, Any]:
        """
        전체 검증 실행
        
        Returns:
            종합 검증 결과
        """
        logger.info(f"🔍 토큰 검증 시작: {self.model_name}")
        
        # 1. 모델 및 토크나이저 로드
        load_success, load_error = self.load_tokenizer_and_model()
        if not load_success:
            return {
                "success": False,
                "error": load_error,
                "stage": "model_loading"
            }
        
        # 2. Vocabulary 호환성 검증
        vocab_result = self.validate_vocabulary_compatibility()
        if not vocab_result["success"]:
            return {
                "success": False,
                "error": vocab_result["error"],
                "stage": "vocabulary_validation"
            }
        
        # 3. 샘플 데이터 검증
        sample_result = self.validate_sample_data()
        if not sample_result["success"]:
            return {
                "success": False,
                "error": sample_result["error"],
                "stage": "sample_validation"
            }
        
        # 4. 종합 결과
        overall_valid = (
            vocab_result["is_compatible"] and 
            sample_result["is_valid"]
        )
        
        result = {
            "success": True,
            "overall_valid": overall_valid,
            "model_name": self.model_name,
            "vocabulary_validation": vocab_result,
            "sample_validation": sample_result,
            "summary": {
                "vocab_compatible": vocab_result["is_compatible"],
                "sample_data_valid": sample_result["is_valid"],
                "can_proceed": overall_valid
            },
            "recommendations": []
        }
        
        # 전체 권장사항 수집
        all_recommendations = []
        if vocab_result.get("warning_messages"):
            all_recommendations.extend(vocab_result["warning_messages"])
        if sample_result.get("recommendations"):
            all_recommendations.extend(sample_result["recommendations"])
        
        result["recommendations"] = all_recommendations
        
        # 메모리 정리
        if self.model:
            del self.model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        logger.info(f"✅ 토큰 검증 완료: {'통과' if overall_valid else '실패'}")
        
        return result


def validate_model_tokenizer_compatibility(model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    모델-토크나이저 호환성 검증 편의 함수
    
    Args:
        model_name: 모델 이름
        config: 실험 설정
        
    Returns:
        검증 결과
    """
    validator = TokenValidator(model_name, config)
    return validator.run_full_validation()


def fix_token_range_issues(model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    토큰 범위 문제 자동 수정
    
    Args:
        model_name: 모델 이름
        config: 실험 설정
        
    Returns:
        수정된 설정과 결과
    """
    validator = TokenValidator(model_name, config)
    validation_result = validator.run_full_validation()
    
    if validation_result["success"] and not validation_result["overall_valid"]:
        fixed_config = config.copy()
        
        # 특수 토큰 제거 또는 수정
        vocab_result = validation_result["vocabulary_validation"]
        if vocab_result.get("invalid_special_tokens"):
            logger.warning("유효하지 않은 특수 토큰 제거")
            invalid_tokens = [t[0] for t in vocab_result["invalid_special_tokens"]]
            current_special_tokens = fixed_config.get("tokenizer", {}).get("special_tokens", [])
            fixed_special_tokens = [t for t in current_special_tokens if t not in invalid_tokens]
            
            if "tokenizer" not in fixed_config:
                fixed_config["tokenizer"] = {}
            fixed_config["tokenizer"]["special_tokens"] = fixed_special_tokens
        
        # 길이 설정 조정 (토큰 범위 문제가 있는 경우)
        sample_result = validation_result["sample_validation"]
        if sample_result.get("token_range_exceeded"):
            logger.warning("토큰 길이 설정 보수적으로 조정")
            current_encoder_len = fixed_config.get("tokenizer", {}).get("encoder_max_len", 1024)
            current_decoder_len = fixed_config.get("tokenizer", {}).get("decoder_max_len", 200)
            
            # 보수적으로 길이 줄이기
            safe_encoder_len = min(current_encoder_len, 512)
            safe_decoder_len = min(current_decoder_len, 128)
            
            fixed_config["tokenizer"]["encoder_max_len"] = safe_encoder_len
            fixed_config["tokenizer"]["decoder_max_len"] = safe_decoder_len
        
        return {
            "success": True,
            "config_modified": True,
            "original_config": config,
            "fixed_config": fixed_config,
            "validation_result": validation_result
        }
    
    return {
        "success": validation_result["success"],
        "config_modified": False,
        "fixed_config": config,
        "validation_result": validation_result
    }
