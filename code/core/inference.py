"""
독립 추론 엔진

baseline.ipynb에서 분리된 독립적인 추론 엔진으로,
대회 제출 형식을 지원하고 배치 처리를 통해 효율적인 추론을 수행합니다.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")

import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer
)

# 상위 디렉토리의 utils를 import
sys.path.append(str(Path(__file__).parent.parent))
from utils.device_utils import get_optimal_device, setup_device_config
from utils.path_utils import PathManager, path_manager
from utils.data_utils import DataProcessor

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """추론 설정"""
    model_path: str
    batch_size: int = 8
    max_source_length: int = 1024
    max_target_length: int = 256
    num_beams: int = 5
    length_penalty: float = 1.0
    early_stopping: bool = True
    use_cache: bool = True
    device: Optional[str] = None
    fp16: bool = False
    num_workers: int = 0


class InferenceEngine:
    """
    독립 추론 엔진
    
    모델 로드, 단일/배치 예측, DataFrame 처리 등의 기능을 제공합니다.
    생산 환경에서 효율적인 대화 요약 추론을 위한 고성능 엔진입니다.
    
    Features:
        - 다중 입력 형식 지원 (string, list, DataFrame)
        - 배치 최적화 처리 및 DataLoader 통합
        - 자동 디바이스 감지 및 메모리 최적화
        - 대회 제출 형식 지원
        - 진행률 추적 및 캐시 시스템
        - 강력한 에러 처리 및 로깅
        
    Example:
        >>> engine = InferenceEngine('models/best_model')
        >>> summary = engine.predict_single('대화 텍스트')
        >>> summaries = engine.predict_batch(['대화1', '대화 2'])
        >>> df_result = engine.predict_from_dataframe(df)
    """
    
    def __init__(self, config: Union[InferenceConfig, Dict[str, Any]]):
        """
        추론 엔진 초기화
        
        Args:
            config: InferenceConfig 객체 또는 설정 딕셔너리
        """
        if isinstance(config, dict):
            self.config = InferenceConfig(**config)
        else:
            self.config = config
            
        # 디바이스 설정
        self._setup_device()
        
        # 모델 및 토크나이저 로드
        self._load_model_and_tokenizer()
        
        # 데이터 프로세서 초기화
        self.data_processor = DataProcessor({
            'preprocessing': {
                'max_source_length': self.config.max_source_length,
                'max_target_length': self.config.max_target_length
            }
        })
        
        logger.info(f"추론 엔진 초기화 완료: {self.config.model_path}")
    
    def _setup_device(self) -> None:
        """디바이스 설정"""
        if self.config.device:
            self.device = torch.device(self.config.device)
            logger.info(f"수동 설정된 디바이스: {self.device}")
        else:
            # 자동 디바이스 감지
            self.device, device_info = get_optimal_device()
            
            # 디바이스별 최적화 설정 적용
            opt_config = setup_device_config(device_info, 'base')
            
            # 배치 크기가 지정되지 않았으면 최적화된 값 사용
            if self.config.batch_size == 8:  # 기본값인 경우
                self.config.batch_size = opt_config.batch_size
            
            # FP16 설정
            if opt_config.fp16 and not self.config.fp16:
                self.config.fp16 = opt_config.fp16
                
            logger.info(f"자동 감지된 디바이스: {self.device}")
            logger.info(f"배치 크기: {self.config.batch_size}, FP16: {self.config.fp16}")
    
    def _load_model_and_tokenizer(self) -> None:
        """모델과 토크나이저 로드"""
        logger.info(f"모델 로드 중: {self.config.model_path}")
        
        try:
            # 모델 경로 해결
            model_path = path_manager.resolve_path(self.config.model_path)
            if not model_path.exists():
                # Hugging Face Hub에서 로드 시도
                model_path = self.config.model_path
                logger.info(f"로컬 경로가 없어 HuggingFace Hub에서 로드: {model_path}")
            
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            
            # 모델 타입 감지 및 로드
            try:
                # 시퀀스-투-시퀀스 모델 시도
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    str(model_path),
                    torch_dtype=torch.float16 if self.config.fp16 else torch.float32
                )
                self.model_type = "seq2seq"
                logger.info("Seq2Seq 모델로 로드됨")
            except Exception as seq2seq_error:
                # Seq2Seq 모델 로드 실패, 인과 언어 모델 시도
                logger.debug(f"Seq2Seq 모델 로드 실패: {seq2seq_error}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(model_path),
                    torch_dtype=torch.float16 if self.config.fp16 else torch.float32
                )
                self.model_type = "causal"
                logger.info("Causal LM 모델로 로드됨")
            
            # 디바이스로 이동
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # 특수 토큰 설정
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info(f"모델 로드 완료: {self.model.__class__.__name__}")
            
        except Exception as e:
            logger.error(f"모델 로드 실패: {type(e).__name__}: {e}")
            logger.error(f"Model path: {self.config.model_path}")
            logger.error(f"Available models: AutoModelForSeq2SeqLM, AutoModelForCausalLM")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            raise
    
    def predict_single(self, dialogue: str) -> str:
        """
        단일 대화 요약 생성
        
        Args:
            dialogue: 대화 텍스트
            
        Returns:
            생성된 요약
        """
        # 입력 토큰화
        inputs = self.tokenizer(
            dialogue,
            max_length=self.config.max_source_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # 추론
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.config.max_target_length,
                num_beams=self.config.num_beams,
                length_penalty=self.config.length_penalty,
                early_stopping=self.config.early_stopping,
                use_cache=self.config.use_cache
            )
        
        # 디코딩
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary.strip()
    
    def predict_batch(self, dialogues: List[str], show_progress: bool = True) -> List[str]:
        """
        배치 대화 요약 생성
        
        Args:
            dialogues: 대화 텍스트 리스트
            show_progress: 진행률 표시 여부
            
        Returns:
            생성된 요약 리스트
        """
        summaries = []
        
        # 효율적인 배치 처리를 위한 커스텀 Dataset 및 DataLoader 생성
        # 메모리 효율성과 GPU 활용도를 계점한 배치 첫리
        from torch.utils.data import DataLoader, Dataset
        
        class DialogueDataset(Dataset):
            """
            대화 데이터를 위한 간단한 Dataset 클래스
            PyTorch DataLoader와 호환되도록 설계됨
            """
            def __init__(self, dialogues: List[str]) -> None:
                self.dialogues = dialogues
            
            def __len__(self) -> int:
                return len(self.dialogues)
            
            def __getitem__(self, idx: int) -> str:
                return self.dialogues[idx]
        
        dataset = DialogueDataset(dialogues)
        
        # 배치 크기와 워커 수를 조정하여 메모리 사용량과 처리 속도 균형
        # shuffle=False로 설정하여 입력 순서 유지 (결과 매칭에 중요)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config.batch_size,
            shuffle=False,  # 입력 순서 유지 필수
            num_workers=self.config.num_workers  # CPU 코어 활용
        )
        
        # 진행률 표시 설정
        iterator = tqdm(dataloader, desc="추론 중") if show_progress else dataloader
        
        # 배치 추론
        for batch in iterator:
            # 입력 토큰화
            inputs = self.tokenizer(
                batch,
                max_length=self.config.max_source_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            # 추론
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.config.max_target_length,
                    num_beams=self.config.num_beams,
                    length_penalty=self.config.length_penalty,
                    early_stopping=self.config.early_stopping,
                    use_cache=self.config.use_cache
                )
            
            # 배치 디코딩
            batch_summaries = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            summaries.extend([s.strip() for s in batch_summaries])
        
        return summaries
    
    def predict_from_dataframe(self, df: pd.DataFrame, 
                             dialogue_column: str = 'dialogue',
                             output_column: str = 'summary',
                             show_progress: bool = True) -> pd.DataFrame:
        """
        DataFrame에서 직접 추론 수행
        
        Args:
            df: 입력 DataFrame
            dialogue_column: 대화가 포함된 컬럼명
            output_column: 생성된 요약을 저장할 컬럼명
            show_progress: 진행률 표시 여부
            
        Returns:
            요약이 추가된 DataFrame
        """
        # 입력 검증
        if dialogue_column not in df.columns:
            raise ValueError(f"'{dialogue_column}' 컬럼을 찾을 수 없습니다.")
        
        # 대화 추출
        dialogues = df[dialogue_column].tolist()
        
        # 배치 추론
        logger.info(f"{len(dialogues)}개 대화에 대한 추론 시작...")
        summaries = self.predict_batch(dialogues, show_progress=show_progress)
        
        # 결과 추가
        result_df = df.copy()
        result_df[output_column] = summaries
        
        logger.info(f"추론 완료. '{output_column}' 컬럼에 결과 저장됨.")
        return result_df
    
    def save_submission(self, df: pd.DataFrame, output_path: str,
                       fname_column: str = 'fname',
                       summary_column: str = 'summary') -> None:
        """
        대회 제출 형식으로 저장
        
        Args:
            df: 결과 DataFrame
            output_path: 저장 경로
            fname_column: 파일명 컬럼
            summary_column: 요약 컬럼
        """
        # 제출 형식으로 변환
        submission_df = df[[fname_column, summary_column]].copy()
        
        # 경로 해결
        output_path = path_manager.resolve_path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # CSV 저장
        submission_df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"제출 파일 저장 완료: {output_path}")
    
    def __call__(self, dialogue: Union[str, List[str], pd.DataFrame], **kwargs):
        """
        다양한 입력 형식 지원
        
        Args:
            dialogue: 대화 텍스트, 리스트, 또는 DataFrame
            **kwargs: 추가 인자
            
        Returns:
            요약 결과
        """
        if isinstance(dialogue, str):
            return self.predict_single(dialogue)
        elif isinstance(dialogue, list):
            return self.predict_batch(dialogue, **kwargs)
        elif isinstance(dialogue, pd.DataFrame):
            return self.predict_from_dataframe(dialogue, **kwargs)
        else:
            raise ValueError(f"지원하지 않는 입력 타입: {type(dialogue)}")


def create_inference_engine(model_path: str, **kwargs) -> InferenceEngine:
    """
    추론 엔진 생성 헬퍼 함수
    
    Args:
        model_path: 모델 경로
        **kwargs: 추가 설정
        
    Returns:
        InferenceEngine 인스턴스
    """
    config = InferenceConfig(model_path=model_path, **kwargs)
    return InferenceEngine(config)
