"""
NLP 대화 요약 모델 학습 모듈

baseline.ipynb의 핵심 학습 로직을 모듈화한 트레이너 클래스.
WandB Sweep과의 통합을 위해 설계되었으며, 다양한 모델과 설정을 지원합니다.
"""

import os
import sys
from pathlib import Path

# .env 파일 로드 (WandB API 키 등)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️ python-dotenv가 설치되지 않았습니다. .env 파일을 로드할 수 없습니다.")
    pass

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    PreTrainedModel,
    PreTrainedTokenizer
)

# QLoRA 및 unsloth 관련 import (선택적)
try:
    from unsloth import FastLanguageModel
    from peft import LoraConfig, get_peft_model, TaskType
    UNSLOTH_AVAILABLE = True
except ImportError:
    # macOS 환경이나 unsloth가 설치되지 않은 경우
    FastLanguageModel = None
    LoraConfig = None
    get_peft_model = None
    TaskType = None
    UNSLOTH_AVAILABLE = False

from datasets import Dataset, DatasetDict

# evaluate 모듈 선택적 import
try:
    import evaluate
    EVALUATE_AVAILABLE = True
except ImportError:
    print("⚠️ evaluate 라이브러리를 찾을 수 없습니다. ROUGE 메트릭 계산이 제한될 수 있습니다.")
    print("👉 'pip install evaluate' 명령으로 설치하세요.")
    evaluate = None
    EVALUATE_AVAILABLE = False

import wandb
# 로컬 유틸리티 임포트
from utils import load_config
from utils.data_utils import DataProcessor
from utils.metrics import RougeCalculator
from utils.experiment_utils import ExperimentTracker, ModelRegistry
from utils.environment_detector import EnvironmentDetector, get_auto_config, should_use_unsloth
from utils.path_utils import PathManager, path_manager


logger = logging.getLogger(__name__)


# BART 모델을 위한 커스텀 DataCollator
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
                
    def __call__(self, features, return_tensors=None):
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


@dataclass
class TrainingResult:
    """학습 결과 데이터 클래스"""
    best_metrics: Dict[str, float]
    final_metrics: Dict[str, float]
    model_path: str
    config_used: Dict[str, Any]
    training_history: List[Dict[str, Any]] = field(default_factory=list)
    wandb_run_id: Optional[str] = None
    experiment_id: Optional[str] = None
    submission_file: Optional[str] = None  # CSV 제출 파일 경로


class WandbCallback(TrainerCallback):
    """WandB 로깅을 위한 커스텀 콜백"""
    
    def __init__(self, trainer_instance: 'DialogueSummarizationTrainer') -> None:
        self.trainer_instance = trainer_instance
        self.best_metrics = {}
        
    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, 
                   metrics: Dict[str, float], **kwargs):
        """평가 시 WandB에 메트릭 로깅"""
        if wandb.run is not None:
            # ROUGE 점수 결합 (F1 기준)
            rouge_combined = (
                metrics.get('eval_rouge1', 0) * 0.33 +
                metrics.get('eval_rouge2', 0) * 0.33 +
                metrics.get('eval_rougeL', 0) * 0.34
            )
            
            log_metrics = {
                'eval/rouge1_f1': metrics.get('eval_rouge1', 0),
                'eval/rouge2_f1': metrics.get('eval_rouge2', 0),
                'eval/rougeL_f1': metrics.get('eval_rougeL', 0),
                'eval/rouge_combined_f1': rouge_combined,
                'eval/loss': metrics.get('eval_loss', 0),
                'epoch': state.epoch,
                'step': state.global_step
            }
            
            # 베스트 메트릭 업데이트
            if rouge_combined > self.best_metrics.get('rouge_combined_f1', 0):
                self.best_metrics = {
                    'rouge1_f1': metrics.get('eval_rouge1', 0),
                    'rouge2_f1': metrics.get('eval_rouge2', 0),
                    'rougeL_f1': metrics.get('eval_rougeL', 0),
                    'rouge_combined_f1': rouge_combined,
                    'loss': metrics.get('eval_loss', 0)
                }
                log_metrics['best/rouge_combined_f1'] = rouge_combined
            
            wandb.log(log_metrics)
            # 실험 추적기에도 로깅
            if self.trainer_instance.experiment_tracker:
                self.trainer_instance.experiment_tracker.log_metrics(
                    metrics, step=state.global_step
                )
    
    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """학습 종료 시 최종 결과 로깅"""
        if wandb.run is not None:
            wandb.run.summary.update(self.best_metrics)


class DialogueSummarizationTrainer:
    """
    대화 요약 모델 학습 트레이너
    
    baseline.ipynb의 학습 로직을 모듈화하고 WandB Sweep과 통합하여
    생산성 높은 실험 환경을 제공합니다.
    
    Features:
        - 다중 모델 아키텍처 지원 (BART, T5, KoBART 등)
        - 자동 디바이스 감지 및 최적화 (CUDA, MPS, CPU)
        - 실험 추적 및 모델 등록 시스템
        - 커스텀 콜백 및 메트릭 계산
        - 포괄적 에러 처리 및 로깅
        - WandB 통합 실험 관리
        
    Example:
        >>> config = load_config('configs/bart_base.yaml')
        >>> trainer = DialogueSummarizationTrainer(config)
        >>> datasets = trainer.prepare_data()
        >>> result = trainer.train(datasets)
    """
    
    def __init__(self, config: Dict[str, Any], 
                 sweep_mode: bool = False,
                 experiment_name: Optional[str] = None):
        """
        트레이너 초기화
        
        Args:
            config: 설정 딕셔너리 (ConfigManager로부터)
            sweep_mode: WandB Sweep 모드 여부
            experiment_name: 실험명 (None이면 자동 생성)
        """
        self.config = config
        self.sweep_mode = sweep_mode
        self.experiment_name = experiment_name or config.get('meta', {}).get('experiment_name', 'dialogue_summarization')
        
        # 환경 자동 감지 및 설정 최적화
        self.env_detector = EnvironmentDetector()
        self.env_info = self.env_detector.detect_environment()
        self.auto_config = self.env_detector.get_recommended_config()
        
        # 디바이스 설정
        self.device = self._setup_device()
        
        # 모델 및 토크나이저 초기화
        self.model = None
        self.tokenizer = None
        self.data_processor = None
        self.rouge_calculator = None
        self.trainer = None
        
        # 실험 관리
        self.experiment_tracker = None
        self.model_registry = None
        
        # 로깅 설정
        # 로깅 설정
        self._setup_logging()
        
        # 환경 정보 출력
        self._print_environment_info()
        
        logger.info(f"Trainer initialized with config: {self.experiment_name}")
        
        def setup_wandb_with_korean_time(self, config: Dict[str, Any]) -> bool:
            """
            한국 시간 기반 WandB 초기화
            
            Args:
                config: 실험 설정 딕셔너리
                
            Returns:
                WandB 활성화 여부
            """
            if os.environ.get("WANDB_MODE") == "disabled":
                print("⚠️ WandB가 비활성화되어 있습니다.")
                return False
                
            try:
                from utils.experiment_utils import get_wandb_run_name_with_korean_time
                korean_time = get_wandb_run_name_with_korean_time()
                
                # 기존 wandb 설정을 한국 시간으로 개선
                wandb_config = config.get('wandb', {})
                original_name = wandb_config.get('name', self.experiment_name)
                wandb_config['name'] = f"{original_name}_{korean_time}"
                
                # WandB 초기화 (이미 초기화된 경우 스킨)
                if wandb.run is None:
                    wandb.init(
                        entity=wandb_config.get('entity', 'lyjune37-juneictlab'),
                        project=wandb_config.get('project', 'nlp-5'),
                        name=wandb_config['name'],
                        config=config,
                        tags=wandb_config.get('tags', []) + ['rtx3090_optimized', 'korean_time']
                    )
                    print(f"✅ WandB 초기화 완료: {wandb_config['name']}")
                else:
                    print(f"ℹ️ WandB 이미 초기화됨: {wandb.run.name}")
                
                return True
                
            except ImportError as e:
                print(f"⚠️ 한국 시간 유틸리티 import 실패: {e}")
                return False
            except Exception as e:
                print(f"⚠️ WandB 초기화 실패: {e}")
                return False
        
        def save_best_model_as_artifact(self, model_path: str, metrics: Dict[str, float]) -> None:
            """
            Best model을 WandB Artifacts로 저장
            
            Args:
                model_path: 모델 저장 경로
                metrics: 성능 메트릭
            """
            if wandb.run is None:
                print("⚠️ WandB가 초기화되지 않아 Artifacts 저장을 건너뗁니다.")
                return
                
            try:
                from utils.experiment_utils import get_korean_time_format
                korean_time = get_korean_time_format('MMDDHHMM')
                
                # ROUGE 종합 점수 계산
                rouge_combined = metrics.get('rouge_combined_f1', 0)
                if rouge_combined == 0:
                    # 대체 계산 방법
                    rouge1 = metrics.get('eval_rouge1_f1', 0) or metrics.get('rouge1_f1', 0)
                    rouge2 = metrics.get('eval_rouge2_f1', 0) or metrics.get('rouge2_f1', 0)
                    rougeL = metrics.get('eval_rougeL_f1', 0) or metrics.get('rougeL_f1', 0)
                    rouge_combined = (rouge1 + rouge2 + rougeL) / 3
                
                # Artifact 생성
                artifact = wandb.Artifact(
                    name=f"best-model-{korean_time}",
                    type="model",
                    description=f"Best model (ROUGE-F1: {rouge_combined:.4f}) - Korean time: {korean_time}"
                )
                
                # 모델 디렉토리 추가
                model_path = Path(model_path)
                if model_path.exists():
                    artifact.add_dir(str(model_path))
                    
                    # 메타데이터 추가
                    artifact.metadata = {
                        "best_metrics": metrics,
                        "korean_time": korean_time,
                        "model_path": str(model_path),
                        "rouge_combined_f1": rouge_combined
                    }
                    
                    # Artifact 로깅
                    wandb.log_artifact(artifact, aliases=["latest", "best"])
                    print(f"✅ WandB Artifacts 저장 완료: {artifact.name}")
                    print(f"   모델 경로: {model_path}")
                    print(f"   ROUGE-F1: {rouge_combined:.4f}")
                else:
                    print(f"⚠️ 모델 경로가 존재하지 않습니다: {model_path}")
                    
            except Exception as e:
                print(f"⚠️ WandB Artifacts 저장 실패: {e}")
        
        def setup_paths(self) -> None:
            """경로 설정"""
            # 경로 관리자를 사용하여 경로 관리
            experiment_name = self.experiment_name
            
            # Sweep 모드일 때는 run ID를 포함
            if self.sweep_mode and wandb.run:
                experiment_name = f"sweep_{wandb.run.id}"
            else:
                from datetime import datetime
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                experiment_name = f"{self.experiment_name}_{timestamp}"
            
            # 경로 관리자를 통한 경로 설정
            self.output_dir = path_manager.get_output_path(experiment_name)
            self.model_save_dir = path_manager.get_model_path(experiment_name)
            self.results_dir = path_manager.ensure_dir(self.output_dir / "results")
            
            # 로그 디렉토리 설정
            self.log_dir = path_manager.get_log_path(experiment_name)
        
    def initialize_components(self) -> None:
        """모든 컴포넌트 초기화"""
        logger.info("Initializing components...")
        
        # 실험 추적기 초기화
        if self.config.get('experiment_tracking', {}).get('enabled', True):
            self.experiment_tracker = ExperimentTracker(
                experiments_dir=self.output_dir / "experiments"
            )
            self.model_registry = ModelRegistry(
                registry_dir=self.output_dir / "models"
            )
        
        # 토크나이저 로딩
        self._load_tokenizer()
        
        # 모델 로딩
        self._load_model()
        
        # 데이터 프로세서 초기화
        self.data_processor = DataProcessor(
            tokenizer=self.tokenizer,
            config=self.config
        )
        
        # ROUGE 계산기 초기화
        self.rouge_calculator = RougeCalculator(
            use_korean_tokenizer=self.config.get('evaluation', {}).get('rouge_tokenize_korean', True),
            use_stemmer=self.config.get('evaluation', {}).get('rouge_use_stemmer', True)
        )
        
        logger.info("All components initialized successfully")
    
    def prepare_data(self, train_path: Optional[str] = None, 
                    val_path: Optional[str] = None,
                    test_path: Optional[str] = None) -> DatasetDict:
        """
        데이터 준비
        
        Args:
            train_path: 학습 데이터 경로
            val_path: 검증 데이터 경로  
            test_path: 테스트 데이터 경로
            
        Returns:
            처리된 데이터셋 딕셔너리
        """
        # 기본 데이터 경로 설정
        data_path = self.config.get('general', {}).get('data_path', 'data/')
        
        # 상대 경로를 프로젝트 루트 기준으로 처리
        if not Path(data_path).is_absolute():
            # 프로젝트 루트 찾기 (trainer.py는 code/ 디렉토리에 있음)
            project_root = Path(__file__).parent.parent
            base_path = project_root / data_path
        else:
            base_path = Path(data_path)
        
        # 경로가 없으면 config에서 가져오거나 기본 파일명 사용
        if train_path is None:
            train_path = self.config.get('general', {}).get('train_path')
            if train_path is None:
                train_path = str(base_path / 'train.csv')
        if val_path is None:
            val_path = self.config.get('general', {}).get('val_path')
            if val_path is None:
                val_path = str(base_path / 'dev.csv')
        if test_path is None:
            test_path = self.config.get('general', {}).get('test_path')
            if test_path is None and (base_path / 'test.csv').exists():
                test_path = str(base_path / 'test.csv')
        
        logger.info("Loading and processing datasets...")
        logger.info(f"Project root: {Path(__file__).parent.parent}")
        logger.info(f"Base data path: {base_path}")
        logger.info(f"Train path: {train_path}")
        logger.info(f"Val path: {val_path}")
        
        datasets = {}
        
        if train_path and Path(train_path).exists():
            train_data = self.data_processor.load_data(train_path)
            datasets['train'] = self.data_processor.process_data(
                train_data, 
                is_training=True
            )
            logger.info(f"Train dataset size: {len(datasets['train'])}")
        else:
            logger.warning(f"Train data not found at {train_path}")
        
        if val_path and Path(val_path).exists():
            val_data = self.data_processor.load_data(val_path)
            datasets['validation'] = self.data_processor.process_data(
                val_data,
                is_training=False
            )
            logger.info(f"Validation dataset size: {len(datasets['validation'])}")
        else:
            logger.warning(f"Validation data not found at {val_path}")
        
        if test_path and Path(test_path).exists():
            test_data = self.data_processor.load_data(test_path, is_test=True)
            datasets['test'] = self.data_processor.process_data(
                test_data,
                is_training=False,
                is_test=True
            )
            logger.info(f"Test dataset size: {len(datasets['test'])}")
        
        return DatasetDict(datasets)
    
    def train(self, dataset: DatasetDict, 
             resume_from_checkpoint: Optional[str] = None) -> TrainingResult:
        """
        모델 학습
        
        Args:
            dataset: 학습/검증 데이터셋
            resume_from_checkpoint: 체크포인트 경로 (재개 시)
            
        Returns:
            학습 결과
        """
        # 실험 시작
        if self.experiment_tracker:
            experiment_id = self.experiment_tracker.start_experiment(
                name=self.experiment_name,
                description=f"Training {self._get_model_architecture()} model",
                config=self.config,
                model_type=self._get_model_architecture(),
                dataset_info={
                    'train_size': len(dataset.get('train', [])),
                    'val_size': len(dataset.get('validation', []))
                },
                wandb_run_id=wandb.run.id if wandb.run else None
            )
        else:
            experiment_id = None
        
        # 학습 인자 설정
        training_args = self._get_training_arguments()
        
        # 데이터 콜레이터 - 모델 타입에 따라 SmartDataCollatorForSeq2Seq 사용
        data_collator = SmartDataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            max_length=self.config['tokenizer']['encoder_max_len']
        )
        
        # 평가 메트릭 함수 - HuggingFace Trainer의 콜백으로 사용됨
        def compute_metrics(eval_preds: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
            """
            학습 중 평가 단계에서 ROUGE 메트릭을 계산하는 중첩 함수
            
            Args:
                eval_preds: (predictions, labels) 튜플
                
            Returns:
                ROUGE 점수들을 포함한 딕셔너리
            """
            preds, labels = eval_preds
            
            # 토큰 ID를 텍스트로 디코딩 (특수 토큰 제거)
            # 먼저 predictions 디코딩 (보통 문제없음)
            try:
                decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
            except Exception as e:
                logger.warning(f"⚠️  Predictions 디코딩 오류: {e}")
                decoded_preds = ["" for _ in range(len(preds))]
            
            # labels 안전 디코딩 (OverflowError 방지)
            try:
                # HuggingFace에서 사용하는 -100 패딩 토큰을 정상 토큰으로 변환
                # -100은 loss 계산에서 무시되는 라벨이지만 디코딩에서는 문제가 됨
                labels_fixed = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
                
                # 토큰 ID 범위 검증 (추가 안전장치)
                vocab_size = getattr(self.tokenizer, 'vocab_size', 50000)  # 기본값 설정
                labels_fixed = np.clip(labels_fixed, 0, vocab_size - 1)
                
                # 데이터 타입 확인 및 변환
                if hasattr(labels_fixed, 'astype'):
                    labels_fixed = labels_fixed.astype(np.int32)
                
                decoded_labels = self.tokenizer.batch_decode(labels_fixed, skip_special_tokens=True)
                
            except Exception as e:
                logger.warning(f"⚠️  Labels 디코딩 오류: {e}")
                # 폴백: 빈 문자열로 처리
                decoded_labels = ["" for _ in range(len(labels))]
            
            # 대화 요약에 특화된 ROUGE 메트릭 계산 (Multi-reference 지원)
            try:
                # 디코딩된 텍스트로 직접 ROUGE 계산
                rouge_scores = []
                for pred, ref in zip(decoded_preds, decoded_labels):
                    if pred.strip() and ref.strip():  # 빈 문자열 방지
                        score = self.rouge_calculator.calculate_single_reference(pred, ref)
                        rouge_scores.append(score)
                    else:
                        # 빈 문자열에 대한 0점 처리
                        from utils.metrics import RougeScore, EvaluationResult
                        zero_rouge = RougeScore(precision=0.0, recall=0.0, f1=0.0)
                        zero_result = EvaluationResult(
                            rouge1=zero_rouge, rouge2=zero_rouge, rougeL=zero_rouge, rouge_combined_f1=0.0
                        )
                        rouge_scores.append(zero_result)
                
                if rouge_scores:
                    # 평균 점수 계산
                    avg_rouge1_f1 = sum(score.rouge1.f1 for score in rouge_scores) / len(rouge_scores)
                    avg_rouge2_f1 = sum(score.rouge2.f1 for score in rouge_scores) / len(rouge_scores)
                    avg_rougeL_f1 = sum(score.rougeL.f1 for score in rouge_scores) / len(rouge_scores)
                    avg_combined_f1 = avg_rouge1_f1 + avg_rouge2_f1 + avg_rougeL_f1
                    
                    result = {
                        'rouge1_f1': avg_rouge1_f1,
                        'rouge2_f1': avg_rouge2_f1,
                        'rougeL_f1': avg_rougeL_f1,
                        'rouge_combined_f1': avg_combined_f1
                    }
                else:
                    # 모든 샘플이 비어있는 경우
                    result = {
                        'rouge1_f1': 0.0,
                        'rouge2_f1': 0.0,
                        'rougeL_f1': 0.0,
                        'rouge_combined_f1': 0.0
                    }
                    
            except Exception as e:
                logger.warning(f"⚠️  ROUGE 계산 오류: {e}")
                # 폴백: 0점 반환
                result = {
                    'rouge1_f1': 0.0,
                    'rouge2_f1': 0.0, 
                    'rougeL_f1': 0.0,
                    'rouge_combined_f1': 0.0
                }
            
            return result
        
        # 콜백 설정
        callbacks = [WandbCallback(self)]
        
        # 조기 종료 설정
        if self.config['training'].get('early_stopping_patience'):
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.config['training']['early_stopping_patience'],
                    early_stopping_threshold=0.001
                )
            )
        
        # 트레이너 생성
        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset.get('train'),
            eval_dataset=dataset.get('validation'),
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
            callbacks=callbacks
        )
        
        # 학습 시작
        logger.info("Starting training...")
        
        try:
            train_result = self.trainer.train(
                resume_from_checkpoint=resume_from_checkpoint
            )
            
            # 최종 평가
            logger.info("Running final evaluation...")
            eval_results = self.trainer.evaluate()
            
            # 모델 저장
            best_model_path = self.model_save_dir / "best_model"
            self.trainer.save_model(str(best_model_path))
            self.tokenizer.save_pretrained(str(best_model_path))
            
            # 결과 정리
            wandb_callback = callbacks[0]
            training_result = TrainingResult(
                best_metrics=wandb_callback.best_metrics,
                final_metrics=eval_results,
                model_path=str(best_model_path),
                config_used=self.config,
                training_history=[], # 향후 구현
                wandb_run_id=wandb.run.id if wandb.run else None,
                experiment_id=experiment_id
            )
            
            # 실험 종료
            if self.experiment_tracker:
                self.experiment_tracker.end_experiment(
                    experiment_id=experiment_id,
                    final_metrics=eval_results,
                    best_metrics=wandb_callback.best_metrics,
                    status="completed"
                )
            
            # 모델 등록
            if self.model_registry:
                model_id = self.model_registry.register_model(
                    name=f"{self._get_model_architecture()}_{self.experiment_name}",
                    architecture=self._get_model_architecture(),
                    checkpoint=self._get_model_checkpoint(),
                    config=self.config,
                    performance=wandb_callback.best_metrics,
                    training_info={
                        'epochs': self.config['training']['num_train_epochs'],
                        'batch_size': self.config['training']['per_device_train_batch_size'],
                        'learning_rate': self.config['training']['learning_rate']
                    },
                    file_path=str(best_model_path),
                    experiment_id=experiment_id
                )
                logger.info(f"Model registered with ID: {model_id}")
            
            # 결과 저장
            self._save_results(training_result)

            # 테스트 데이터셋이 있으면 예측 및 CSV 생성
            if 'test' in dataset:
                logger.info("🔮 테스트 데이터셋에 대한 예측 생성 중...")
                try:
                    # 예측 생성
                    test_predictions = self.generate_test_predictions(dataset['test'])
                    
                    # CSV 파일 생성
                    submission_path = self._save_submission_csv(test_predictions)
                    logger.info(f"✅ 제출 파일 생성 완료: {submission_path}")
                    
                    # 결과에 추가
                    training_result.submission_file = str(submission_path)
                    
                    # 결과 다시 저장 (제출 파일 경로 포함)
                    self._save_results(training_result)
                except Exception as e:
                    logger.error(f"❌ 테스트 예측 생성 실패: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    # 예측 실패해도 학습은 성공한 것으로 처리

            
                        # WandB Artifacts로 best model 저장
                        try:
                            if training_result.best_metrics and training_result.model_path:
                                self.save_best_model_as_artifact(
                                    model_path=training_result.model_path,
                                    metrics=training_result.best_metrics
                                )
                        except Exception as e:
                            logger.warning(f"⚠️ WandB Artifacts 저장 실패: {e}")
                            # Artifacts 저장 실패는 전체 학습을 중단하지 않음
            
            return training_result
            
        except Exception as e:
            logger.error(f"Training failed with error: {type(e).__name__}: {str(e)}")
            logger.error(f"Current config: {self.config.get('model', {}).get('checkpoint', 'Unknown')}")
            logger.error(f"Device: {self.device}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            logger.error(f"Training failed: {str(e)}")
            if self.experiment_tracker and experiment_id:
                self.experiment_tracker.end_experiment(
                    experiment_id=experiment_id,
                    status="failed",
                    notes=str(e)
                )
            raise
    
    def evaluate(self, dataset: Dataset, 
                metric_key_prefix: str = "eval") -> Dict[str, float]:
        """
        모델 평가
        
        Args:
            dataset: 평가 데이터셋
            metric_key_prefix: 메트릭 키 접두사
            
        Returns:
            평가 결과
        """
        if not self.trainer:
            raise ValueError("Trainer not initialized. Call train() first.")
        
        results = self.trainer.evaluate(
            eval_dataset=dataset,
            metric_key_prefix=metric_key_prefix
        )
        
        return results
    
    def generate_predictions(self, dataset: Dataset, 
                           max_samples: Optional[int] = None) -> List[Dict[str, str]]:
        """
        예측 생성
        
        Args:
            dataset: 입력 데이터셋
            max_samples: 최대 샘플 수 (None이면 전체)
            
        Returns:
            예측 결과 리스트
        """
        self.model.eval()
        predictions = []
        
        # 샘플링
        if max_samples:
            indices = np.random.choice(len(dataset), min(max_samples, len(dataset)), replace=False)
            dataset = dataset.select(indices)
        
        # 생성 설정
        gen_config = self.config['generation']
        
        with torch.no_grad():
            for example in tqdm(dataset, desc="Generating predictions"):
                # 토큰화
                inputs = self.tokenizer(
                    example['input'],
                    max_length=self.config['tokenizer']['encoder_max_len'],
                    truncation=True,
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)
                
                # 생성
                outputs = self.model.generate(
                    **inputs,
                    max_length=gen_config['max_length'],
                    num_beams=gen_config['num_beams'],
                    length_penalty=gen_config.get('length_penalty', 1.0),
                    no_repeat_ngram_size=gen_config.get('no_repeat_ngram_size', 2),
                    early_stopping=gen_config.get('early_stopping', True),
                    do_sample=gen_config.get('do_sample', False),
                    temperature=gen_config.get('temperature', 1.0) if gen_config.get('do_sample') else None,
                    top_k=gen_config.get('top_k', 50) if gen_config.get('do_sample') else None,
                    top_p=gen_config.get('top_p', 0.95) if gen_config.get('do_sample') else None
                )
                
                # 디코딩
                prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                predictions.append({
                    'input': example['input'],
                    'prediction': prediction,
                    'reference': example.get('target', '')
                })
        
        return predictions
    
    def _setup_device(self) -> torch.device:
        """디바이스 설정"""
        from utils.device_utils import get_optimal_device, setup_device_config
        
        device_config = self.config['general'].get('device', 'auto')
        
        if device_config == 'auto':
            # 최적 디바이스 자동 선택
            device, device_info = get_optimal_device()
            
            # 디바이스별 최적화 설정 가져오기
            model_size = self.config.get('model', {}).get('size', 'base')
            optimization_config = setup_device_config(device_info, model_size)
            
            # 최적화 설정을 config에 병합
            if 'training' not in self.config:
                self.config['training'] = {}
            
            # 기존 설정과 병합 (기존 설정 우선)
            opt_dict = optimization_config.to_dict()
            for key, value in opt_dict.items():
                if key not in self.config['training']:
                    self.config['training'][key] = value
            
            logger.info(f"자동 감지된 디바이스: {device_info}")
            logger.info(f"최적화 설정 적용됨: batch_size={optimization_config.batch_size}, "
                       f"mixed_precision={optimization_config.mixed_precision}, "
                       f"num_workers={optimization_config.num_workers}")
        else:
            # 수동 설정
            device = torch.device(device_config)
            logger.info(f"수동 설정된 디바이스: {device}")
        
        logger.info(f"Using device: {device}")
        return device
    
    def _setup_logging(self) -> None:
        """로깅 설정"""
        log_level = self.config.get('logging', {}).get('level', 'INFO')
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'training.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def _print_environment_info(self) -> None:
        """환경 정보 출력"""
        logger.info("\n" + "="*60)
        logger.info("🔍 자동 환경 감지 결과")
        logger.info("="*60)
        logger.info(f"OS: {self.env_info['os']} ({self.env_info['os_release']})")
        logger.info(f"Python: {self.env_info['python_version']}")
        logger.info(f"CPU Cores: {self.env_info['cpu_count']}")
        
        if self.env_info['is_cuda_available']:
            logger.info(f"🎮 CUDA: Available (v{self.env_info['cuda_version']})")
            logger.info(f"GPU Count: {self.env_info['gpu_info']['count']}")
            for device in self.env_info['gpu_info']['devices']:
                logger.info(f"  - {device['name']}: {device['memory_gb']:.1f}GB")
        else:
            logger.info("🎮 CUDA: Not Available")
        
        logger.info(f"\n⚡ Unsloth 지원")
        unsloth_status = "✅ 추천" if self.env_info['unsloth_recommended'] else "❌ 비추천"
        logger.info(f"추천 여부: {unsloth_status}")
        
        install_status = "✅ 설치됨" if self.env_info['unsloth_available'] else "❌ 미설치"
        logger.info(f"설치 상태: {install_status}")
        
        logger.info(f"\n🚀 자동 최적화 설정")
        logger.info(f"use_unsloth: {self.auto_config['use_unsloth']}")
        logger.info(f"recommended_batch_size: {self.auto_config['recommended_batch_size']}")
        logger.info(f"fp16: {self.auto_config['fp16']}, bf16: {self.auto_config['bf16']}")
        logger.info(f"dataloader_num_workers: {self.auto_config['dataloader_num_workers']}")
        logger.info("="*60 + "\n")
    
    def _load_tokenizer(self) -> None:
        """토크나이저 로딩"""
        # model 섹션이 없으면 general에서 model_name 사용
        if 'model' in self.config:
            model_checkpoint = self.config['model']['checkpoint']
        else:
            model_checkpoint = self.config.get('general', {}).get('model_name')
            if not model_checkpoint:
                raise ValueError("Model checkpoint not found in config. Please specify 'model.checkpoint' or 'general.model_name'")
        
        logger.info(f"Loading tokenizer: {model_checkpoint}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_checkpoint,
            use_fast=True
        )
        
        # 특수 토큰 설정 (필요시)
        if 'model' in self.config and self.config['model']['architecture'] in ['kogpt2', 'gpt2']:
            # GPT 계열은 pad_token이 없을 수 있음
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
    def _get_model_architecture(self) -> str:
        """모델 아키텍처 가져오기"""
        if 'model' in self.config:
            return self.config['model']['architecture']
        else:
            # model_name에서 추론
            model_checkpoint = self.config.get('general', {}).get('model_name', '')
            if 'kobart' in model_checkpoint.lower():
                return 'bart'
            elif 't5' in model_checkpoint.lower():
                return 't5'
            elif 'mt5' in model_checkpoint.lower():
                return 'mt5'
            else:
                return 'seq2seq'
    
    def _get_model_checkpoint(self) -> str:
        """모델 체크포인트 가져오기"""
        if 'model' in self.config:
            return self.config['model']['checkpoint']
        else:
            model_checkpoint = self.config.get('general', {}).get('model_name')
            if not model_checkpoint:
                raise ValueError("Model checkpoint not found in config")
            return model_checkpoint
    def _load_model(self) -> None:
        """모델 로딩 (unsloth 및 QLoRA 지원)"""
        # model 섹션이 없으면 general에서 model_name 사용
        if 'model' in self.config:
            model_checkpoint = self.config['model']['checkpoint']
            architecture = self._get_model_architecture()
        else:
            model_checkpoint = self.config.get('general', {}).get('model_name')
            if not model_checkpoint:
                raise ValueError("Model checkpoint not found in config")
            # architecture를 모델 이름에서 추론
            if 'kobart' in model_checkpoint.lower():
                architecture = 'BART'
            elif 't5' in model_checkpoint.lower():
                architecture = 'T5'
            elif 'mt5' in model_checkpoint.lower():
                architecture = 'mT5'
            else:
                architecture = 'seq2seq'  # 기본값
        
        # QLoRA 설정 확인
        qlora_config = self.config.get('qlora', {})
        # QLoRA 설정 확인 및 자동 최적화
        qlora_config = self.config.get('qlora', {})
        
        # 환경 기반 자동 Unsloth 활성화
        config_use_unsloth = qlora_config.get('use_unsloth', False)
        auto_use_unsloth = self.auto_config.get('use_unsloth', False)
        
        # 최종 Unsloth 사용 결정: 설정파일 OR 자동감지
        use_unsloth = (config_use_unsloth or auto_use_unsloth) and UNSLOTH_AVAILABLE
        use_qlora = qlora_config.get('use_qlora', True)  # 기본값 True로 변경
        
        # 자동 최적화 적용
        if auto_use_unsloth and not config_use_unsloth:
            logger.info(f"🚀 환경 자동 감지: Ubuntu + CUDA 환경에서 Unsloth 자동 활성화")
            # 자동 배치 크기 적용
            recommended_batch = self.auto_config.get('recommended_batch_size', 4)
            logger.info(f"📊 기본 배치 크기 권장: {recommended_batch}")
        
        logger.info(f"Loading model: {model_checkpoint} ({architecture})")
        logger.info(f"QLoRA enabled: {use_qlora}, unsloth enabled: {use_unsloth}")
        
        if use_unsloth and architecture in ['kobart', 'bart', 't5', 'mt5']:
            # unsloth로 모델 로딩 (최대 75% 메모리 감소)
            self._load_model_with_unsloth(model_checkpoint, qlora_config)
            
        elif use_qlora:
            # 일반 QLoRA 모델 로딩
            self._load_model_with_qlora(model_checkpoint, architecture, qlora_config)
            
        else:
            # 기존 모델 로딩 방식
            self._load_model_standard(model_checkpoint, architecture)
        
        # 디바이스로 이동 (QLoRA 모델은 이미 적절한 디바이스에 있음)
        if not (use_unsloth or use_qlora):
            self.model = self.model.to(self.device)
        
        # 그래디언트 체크포인팅 (메모리 최적화)
        if self.config['training'].get('gradient_checkpointing', False):
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
            else:
                logger.warning("모델이 gradient_checkpointing을 지원하지 않습니다.")
                
    def _load_model_with_unsloth(self, model_checkpoint: str, qlora_config: Dict[str, Any]) -> None:
        """
        unsloth를 사용한 고효율 모델 로딩 (메모리 75% 감소)
        
        Args:
            model_checkpoint: 모델 체크포인트 경로
            qlora_config: QLoRA 설정
        """
        logger.info("🚀 unsloth로 고효율 모델 로딩 중...")
        
        try:
            # unsloth FastLanguageModel로 모델 로딩
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_checkpoint,
                max_seq_length=self.config['tokenizer'].get('encoder_max_len', 512) + 
                              self.config['tokenizer'].get('decoder_max_len', 200),
                dtype=torch.float16 if self.config['training'].get('fp16') else torch.float32,
                load_in_4bit=qlora_config.get('load_in_4bit', True),
            )
            
            # LoRA 설정 추가
            model = FastLanguageModel.get_peft_model(
                model,
                r=qlora_config.get('lora_rank', 16),
                target_modules=qlora_config.get('target_modules', [
                    "q_proj", "k_proj", "v_proj", "out_proj",
                    "fc1", "fc2"
                ]),
                lora_alpha=qlora_config.get('lora_alpha', 32),
                lora_dropout=qlora_config.get('lora_dropout', 0.1),
                bias="none",
                use_gradient_checkpointing="unsloth",  # unsloth 최적화
                random_state=42,
            )
            
            self.model = model
            logger.info("✅ unsloth 모델 로딩 성공! 메모리 사용량 75% 감소 예상")
            
        except Exception as e:
            logger.error(f"❌ unsloth 모델 로딩 실패: {e}")
            logger.info("폴백 모드: 일반 QLoRA로 대체")
            self._load_model_with_qlora(model_checkpoint, 'kobart', qlora_config)
    
    def _load_model_with_qlora(self, model_checkpoint: str, architecture: str, qlora_config: Dict[str, Any]) -> None:
        """
        일반 QLoRA를 사용한 모델 로딩
        
        Args:
            model_checkpoint: 모델 체크포인트 경로
            architecture: 모델 아키텍처
            qlora_config: QLoRA 설정
        """
        logger.info("🔋 QLoRA로 메모리 효율적 모델 로딩 중...")
        
        try:
            # 4-bit 양자화 설정
            from transformers import BitsAndBytesConfig
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=qlora_config.get('load_in_4bit', True),
                bnb_4bit_compute_dtype=getattr(torch, qlora_config.get('bnb_4bit_compute_dtype', 'float16')),
                bnb_4bit_quant_type=qlora_config.get('bnb_4bit_quant_type', 'nf4'),
                bnb_4bit_use_double_quant=qlora_config.get('bnb_4bit_use_double_quant', True),
            )
            
            # 모델 로딩
            if architecture in ['kobart', 'bart', 't5', 'mt5']:
                # csebuetnlp/mT5_multilingual_XLSum 모델 특수 처리
                if 'mT5_multilingual_XLSum' in model_checkpoint or 'csebuetnlp' in model_checkpoint:
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_checkpoint,
                        quantization_config=bnb_config,
                        device_map="auto",
                        torch_dtype=torch.float16 if self.config['training'].get('fp16') else torch.float32,
                        trust_remote_code=True  # DiaConfig 허용
                    )
                else:
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_checkpoint,
                        quantization_config=bnb_config,
                        device_map="auto",
                        torch_dtype=torch.float16 if self.config['training'].get('fp16') else torch.float32
                    )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_checkpoint,
                    quantization_config=bnb_config,
                    device_map="auto",
                    torch_dtype=torch.float16 if self.config['training'].get('fp16') else torch.float32
                )
            
            # LoRA 설정
            if LoraConfig is not None:
                lora_config = LoraConfig(
                    r=qlora_config.get('lora_rank', 16),
                    lora_alpha=qlora_config.get('lora_alpha', 32),
                    target_modules=qlora_config.get('target_modules', [
                        "q_proj", "k_proj", "v_proj", "out_proj",
                        "fc1", "fc2"
                    ]),
                    lora_dropout=qlora_config.get('lora_dropout', 0.1),
                    bias="none",
                    task_type=TaskType.SEQ_2_SEQ_LM if architecture in ['kobart', 'bart', 't5', 'mt5'] else TaskType.CAUSAL_LM,
                )
                
                model = get_peft_model(model, lora_config)
                logger.info("✅ QLoRA 모델 준비 완료!")
            
            self.model = model
            
        except ImportError:
            logger.error("❌ bitsandbytes 또는 peft 라이브러리가 설치되지 않음")
            logger.info("폴백 모드: 표준 모델 로딩")
            self._load_model_standard(model_checkpoint, architecture)
        except Exception as e:
            logger.error(f"❌ QLoRA 모델 로딩 실패: {e}")
            logger.info("폴백 모드: 표준 모델 로딩")
            self._load_model_standard(model_checkpoint, architecture)
    
    def _load_model_standard(self, model_checkpoint: str, architecture: str) -> None:
        """
        표준 모델 로딩 (기존 방식, DiaConfig 문제 해결 포함)
        
        Args:
            model_checkpoint: 모델 체크포인트 경로
            architecture: 모델 아키텍처
        """
        logger.info("📚 표준 모델 로딩 중...")
        
        # 모델 아키텍처에 따른 로딩
        if architecture in ['kobart', 'bart', 't5', 'mt5']:
            # 시퀀스-투-시퀀스 모델
            model_config = {
                'torch_dtype': torch.float16 if self.config['training'].get('fp16') else torch.float32
            }
            
            # csebuetnlp/mT5_multilingual_XLSum 모델 특수 처리 (DiaConfig 문제 해결)
            if 'mT5_multilingual_XLSum' in model_checkpoint or 'csebuetnlp' in model_checkpoint:
                model_config.update({
                    'trust_remote_code': True,  # DiaConfig 허용
                    'use_cache': False,  # gradient checkpointing과 충돌 방지
                    'output_attentions': False,
                    'output_hidden_states': False
                })
                logger.info("🔧 mT5_multilingual_XLSum 모델 특수 설정 적용 (DiaConfig 지원)")
            
            # 일반 mT5 모델 특수 설정 (그래디언트 체크포인트 안정성)
            elif 'mt5' in model_checkpoint.lower() or 'multilingual' in model_checkpoint.lower():
                model_config.update({
                    'use_cache': False,  # gradient checkpointing과 충돌 방지
                    'output_attentions': False,
                    'output_hidden_states': False
                })
                logger.info("🔧 mT5 모델 안정성 설정 적용")
            
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_checkpoint,
                **model_config
            )
        elif architecture in ['kogpt2', 'gpt2', 'gpt-neo']:
            # 인과 언어 모델
            self.model = AutoModelForCausalLM.from_pretrained(
                model_checkpoint,
                torch_dtype=torch.float16 if self.config['training'].get('fp16') else torch.float32
            )
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        
        logger.info("✅ 표준 모델 로딩 완료!")
    
    def _get_model_specific_config(self, architecture: str, checkpoint: str) -> Dict[str, Any]:
        """모델별 특수 설정 반환"""
        config = {}
        
        # T5 계열
        if architecture in ['t5', 'mt5', 'flan-t5']:
            config['prefix'] = "summarize: "  # T5는 task prefix 필요
            
        # GPT 계열
        elif architecture in ['gpt2', 'kogpt2', 'gpt-neo']:
            config['max_length'] = self.config['tokenizer']['encoder_max_len'] + self.config['tokenizer']['decoder_max_len']
            config['pad_token_id'] = self.tokenizer.pad_token_id
            
        return config
    
    def _preprocess_for_model(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """모델별 데이터 전처리"""
        architecture = self.config['model']['architecture']
        
        if architecture in ['t5', 'mt5', 'flan-t5']:
            # T5는 prefix 추가
            examples['input'] = ["summarize: " + inp for inp in examples['input']]
            
        elif architecture in ['gpt2', 'kogpt2', 'gpt-neo']:
            # GPT는 입력과 타겟을 연결
            examples['input'] = [
                f"{inp} TL;DR: {tgt}" 
                for inp, tgt in zip(examples['input'], examples['target'])
            ]
            
        return examples
    
    def _get_training_arguments(self) -> Seq2SeqTrainingArguments:
        """학습 인자 생성"""
        train_config = self.config['training']
        
        # 기본 인자
        args_dict = {
            'output_dir': str(self.output_dir / 'checkpoints'),
            'overwrite_output_dir': True,
            'do_train': True,
            'do_eval': True,
            'eval_strategy': train_config.get('evaluation_strategy', 'steps'),
            'eval_steps': train_config.get('eval_steps', 500),
            'save_strategy': train_config.get('save_strategy', 'steps'),
            'save_steps': train_config.get('save_steps', 500),
            'save_total_limit': train_config.get('save_total_limit', 3),
            'per_device_train_batch_size': train_config['per_device_train_batch_size'],
            'per_device_eval_batch_size': train_config.get('per_device_eval_batch_size', 
                                                          train_config['per_device_train_batch_size']),
            'gradient_accumulation_steps': train_config.get('gradient_accumulation_steps', 1),
            'learning_rate': train_config['learning_rate'],
            'weight_decay': train_config.get('weight_decay', 0.01),
            'adam_beta1': train_config.get('adam_beta1', 0.9),
            'adam_beta2': train_config.get('adam_beta2', 0.999),
            'adam_epsilon': train_config.get('adam_epsilon', 1e-8),
            'max_grad_norm': train_config.get('max_grad_norm', 1.0),
            'num_train_epochs': 1 if os.environ.get('FORCE_ONE_EPOCH', '').lower() in ['1', 'true', 'yes'] else train_config['num_train_epochs'],
            'lr_scheduler_type': train_config.get('lr_scheduler_type', 'linear'),
            'warmup_ratio': train_config.get('warmup_ratio', 0.1),
            'warmup_steps': train_config.get('warmup_steps', 0),
            'logging_dir': str(self.output_dir / 'logs'),
            'logging_steps': train_config.get('logging_steps', 50),
            'load_best_model_at_end': True,
            'metric_for_best_model': 'eval_rouge_combined_f1',
            'greater_is_better': True,
            'fp16': train_config.get('fp16', False),
            'fp16_opt_level': train_config.get('fp16_opt_level', 'O1'),
            'dataloader_num_workers': train_config.get('dataloader_num_workers', 4),
            'remove_unused_columns': False,
            'label_smoothing_factor': train_config.get('label_smoothing', 0.0),
            'optim': train_config.get('optim', 'adamw_torch'),
            'seed': self.config['general'].get('seed', 42),
            'report_to': ['wandb'] if wandb.run else ['none'],
            'run_name': self.experiment_name if wandb.run else None,
            'push_to_hub': False,
            'predict_with_generate': True,
            'generation_max_length': train_config.get('generation_max_length', self.config['tokenizer']['decoder_max_len']),
            'generation_num_beams': train_config.get('generation_num_beams', 4)
        }
        
        # 시퀀스-투-시퀀스 특화 인자
        seq2seq_args = Seq2SeqTrainingArguments(**args_dict)
        
        return seq2seq_args
    
    def _save_results(self, result: TrainingResult) -> None:
        """결과 저장"""
        # 결과 딕셔너리 생성
        results_dict = {
            'experiment_name': self.experiment_name,
            'model_architecture': self._get_model_architecture(),
            'model_checkpoint': self._get_model_checkpoint(),
            'best_metrics': result.best_metrics,
            'final_metrics': result.final_metrics,
            'model_path': result.model_path,
            'wandb_run_id': result.wandb_run_id,
            'experiment_id': result.experiment_id,
            'config': result.config_used,
            'timestamp': str(Path(result.model_path).parent.parent.name)
        }
        
        # JSON 저장
        results_file = self.results_dir / 'training_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2)
        
        # 요약 텍스트 저장
        summary_file = self.results_dir / 'summary.txt'
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"Training Summary for {self.experiment_name}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model: {self._get_model_architecture()} ({self._get_model_checkpoint()})\n")
            f.write(f"Training Epochs: {self.config['training']['num_train_epochs']}\n")
            f.write(f"Batch Size: {self.config['training']['per_device_train_batch_size']}\n")
            f.write(f"Learning Rate: {self.config['training']['learning_rate']}\n\n")
            f.write("Best Metrics:\n")
            for metric, value in result.best_metrics.items():
                f.write(f"  {metric}: {value:.4f}\n")
            f.write("\nModel saved to: " + result.model_path + "\n")
            if result.wandb_run_id:
                f.write(f"WandB Run ID: {result.wandb_run_id}\n")
            if result.submission_file:
                f.write(f"\nSubmission File: {result.submission_file}\n")
        
        logger.info(f"Results saved to {self.results_dir}")



    def generate_test_predictions(self, test_dataset: Dataset) -> pd.DataFrame:
        """
        테스트 데이터셋에 대한 예측 생성 (baseline.ipynb와 완전히 동일, 모든 모델 지원)
        
        Args:
            test_dataset: 테스트 데이터셋
            
        Returns:
            예측 결과 DataFrame (fname, summary 컬럼)
        """
        self.model.eval()
        summary = []
        fname_list = []
        
        # baseline.ipynb의 inference config 사용
        inference_config = self.config.get('inference', {})
        if not inference_config:
            # 기본값 설정 (baseline.ipynb config와 동일)
            inference_config = {
                'batch_size': 32,
                'no_repeat_ngram_size': 2,
                'early_stopping': True,
                'generate_max_length': 100,
                'num_beams': 4,
                'remove_tokens': self._get_default_remove_tokens()
            }
        
        # DataLoader 생성 (baseline처럼)
        batch_size = inference_config.get('batch_size', 32)
        dataloader = DataLoader(test_dataset, batch_size=batch_size)
        
        with torch.no_grad():
            for item in tqdm(dataloader, desc="Generating predictions"):
                # fname 데이터 수집 (baseline과 동일한 방식)
                if 'fname' in item:
                    fname_list.extend(item['fname'])
                elif 'ID' in item:
                    fname_list.extend(item['ID'])
                else:
                    # 기본값으로 배치 인덱스 사용
                    batch_size_actual = len(item['input_ids'])
                    fname_list.extend([f"test_{len(fname_list) + i:04d}" for i in range(batch_size_actual)])
                
                # baseline.ipynb와 동일한 생성
                generated_ids = self.model.generate(
                    input_ids=item['input_ids'].to(self.device),
                    no_repeat_ngram_size=inference_config['no_repeat_ngram_size'],
                    early_stopping=inference_config['early_stopping'],
                    max_length=inference_config['generate_max_length'],
                    num_beams=inference_config['num_beams'],
                )
                
                # 각 ID별로 디코딩 (baseline과 동일, skip_special_tokens=False)
                for ids in generated_ids:
                    result = self.tokenizer.decode(ids, skip_special_tokens=False)
                    summary.append(result)
        
        # 정확한 평가를 위하여 노이즈에 해당되는 스페셜 토큰을 제거합니다.
        remove_tokens = inference_config.get('remove_tokens', self._get_default_remove_tokens())
        preprocessed_summary = summary.copy()
        for token in remove_tokens:
            if token is not None:  # None 토큰 방지
                preprocessed_summary = [sentence.replace(str(token), " ") for sentence in preprocessed_summary]
        
        # DataFrame 생성 (baseline과 동일)
        output = pd.DataFrame({
            "fname": fname_list[:len(preprocessed_summary)],
            "summary": preprocessed_summary,
        })
        
        return output
    
    def _get_default_remove_tokens(self) -> List[str]:
        """
        모델별 기본 제거 토큰 목록 반환
        
        Returns:
            제거할 토큰 목록
        """
        tokens = ['<usr>']
        
        # 토크나이저 특수 토큰들 추가 (None 체크)
        if hasattr(self.tokenizer, 'bos_token') and self.tokenizer.bos_token is not None:
            tokens.append(self.tokenizer.bos_token)
        if hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token is not None:
            tokens.append(self.tokenizer.eos_token)
        if hasattr(self.tokenizer, 'pad_token') and self.tokenizer.pad_token is not None:
            tokens.append(self.tokenizer.pad_token)
        
        # 모델별 추가 토큰
        model_arch = self._get_model_architecture().lower()
        
        # T5/mT5 계열
        if 't5' in model_arch:
            if hasattr(self.tokenizer, 'additional_special_tokens'):
                tokens.extend([token for token in self.tokenizer.additional_special_tokens if token is not None])
        
        # BART 계열
        elif 'bart' in model_arch:
            # BART 특수 토큰들
            bart_tokens = ['<s>', '</s>', '<pad>', '<unk>']
            tokens.extend(bart_tokens)
        
        # 중복 제거 및 None 필터링
        return list(set([token for token in tokens if token is not None]))
    
    def _save_submission_csv(self, predictions_df: pd.DataFrame) -> Path:
        """
        예측 결과를 CSV 파일로 저장 (baseline.ipynb 형식)
        
        Args:
            predictions_df: 예측 결과 DataFrame
            
        Returns:
            저장된 파일 경로
        """
        from datetime import datetime
        
        # baseline.ipynb처럼 prediction 폴더에 저장
        result_path = self.output_dir / "prediction"
        if not result_path.exists():
            result_path.mkdir(parents=True)
        
        # 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self._get_model_architecture().replace('/', '_')
        
        # baseline.ipynb에서 output.csv 사용하지만 다중 모델 구분을 위해 타임스탬프 추가
        filename = f"output_{model_name}_{timestamp}.csv"
        
        # 저장 경로
        submission_path = result_path / filename
        
        # CSV 저장 (baseline과 동일하게 index=False)
        predictions_df.to_csv(submission_path, index=False)
        
        # 복사본을 최상위 디렉터리에도 저장
        latest_submission_path = self.output_dir.parent / f"submission_latest_{model_name}.csv"
        predictions_df.to_csv(latest_submission_path, index=False)
        
        # 통계 출력
        logger.info(f"\n=== Submission Statistics ===") 
        logger.info(f"Total samples: {len(predictions_df)}")
        logger.info(f"Average summary length: {predictions_df['summary'].str.len().mean():.1f}")
        logger.info(f"Min summary length: {predictions_df['summary'].str.len().min()}")
        logger.info(f"Max summary length: {predictions_df['summary'].str.len().max()}")
        
        # 첫 3개 예측 샘플 출력
        logger.info(f"\n첫 3개 예측 샘플:")
        for i in range(min(3, len(predictions_df))):
            logger.info(f"[{predictions_df.iloc[i]['fname']}] {predictions_df.iloc[i]['summary'][:80]}...")
        
        return submission_path


def create_trainer(config: Union[str, Dict[str, Any]], 
                  sweep_mode: bool = False) -> DialogueSummarizationTrainer:
    """
    트레이너 생성 편의 함수
    
    Args:
        config: 설정 파일 경로 또는 설정 딕셔너리
        sweep_mode: WandB Sweep 모드 여부
        
    Returns:
        초기화된 트레이너 인스턴스
    """
    # 설정 로딩
    if isinstance(config, str):
        config_dict = load_config(config)
    else:
        config_dict = config
    
    # 트레이너 생성
    trainer = DialogueSummarizationTrainer(
        config=config_dict,
        sweep_mode=sweep_mode
    )
    
    # 컴포넌트 초기화
    trainer.initialize_components()
    
    return trainer


if __name__ == "__main__":
    # 테스트/디버깅용 메인 함수
    import argparse
    
    parser = argparse.ArgumentParser(description="Train dialogue summarization model")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--train-data", type=str, help="Train data path")
    parser.add_argument("--val-data", type=str, help="Validation data path")
    parser.add_argument("--test-data", type=str, help="Test data path")
    parser.add_argument("--sweep", action="store_true", help="Run in sweep mode")
    
    args = parser.parse_args()
    
    # 1에포크 모드 확인 및 메시지 출력
    if os.environ.get('FORCE_ONE_EPOCH', '').lower() in ['1', 'true', 'yes']:
        print("🚀 1에포크 모드로 실행합니다!")
        print("📝 학습 epoch 수가 1로 강제 설정됩니다.")
    
    # WandB 초기화 (비 Sweep 모드) - 한국 시간 기반 개선
    if not args.sweep:
        # 환경변수로 비활성화 확인
        if os.environ.get("WANDB_MODE") == "disabled":
            print("⚠️ WandB가 비활성화되어 있습니다.")
        else:
            # .env에서 로드된 WandB 설정 확인
            wandb_entity = os.getenv('WANDB_ENTITY', 'lyjune37-juneictlab')
            wandb_project = os.getenv('WANDB_PROJECT', 'nlp-5')
            
            # 한국 시간 기반 run name 생성
            try:
                from utils.experiment_utils import get_wandb_run_name_with_korean_time
                run_name = get_wandb_run_name_with_korean_time(
                    model_name="manual_training", 
                    prefix="auto"
                )
            except ImportError:
                run_name = "manual_training"
            
            wandb.init(
                entity=wandb_entity,
                project=wandb_project,
                name=run_name,
                config={"manual_run": True, "korean_time_enabled": True},
                tags=["rtx3090_optimized", "korean_time"]
            )
    
    # 트레이너 생성 및 학습
    trainer = create_trainer(args.config, sweep_mode=args.sweep)
    
    # 데이터 준비
    datasets = trainer.prepare_data(
        train_path=args.train_data,
        val_path=args.val_data,
        test_path=args.test_data
    )
    
    # 학습 실행
    result = trainer.train(datasets)
    
    print(f"Training completed! Best ROUGE combined F1: {result.best_metrics.get('rouge_combined_f1', 0):.4f}")
