"""
NLP 대화 요약 모델 학습 모듈

baseline.ipynb의 핵심 학습 로직을 모듈화한 트레이너 클래스.
WandB Sweep과의 통합을 위해 설계되었으며, 다양한 모델과 설정을 지원합니다.
"""

import os
import sys
import json
import logging
import time
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
    PreTrainedTokenizer,
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
import evaluate
import wandb

# 로컬 유틸리티 임포트
try:
    from utils import load_config
    from utils.data_utils import DataProcessor
    from utils.metrics import RougeCalculator
    from utils.experiment_utils import ExperimentTracker, ModelRegistry
    from utils.environment_detector import EnvironmentDetector, get_auto_config, should_use_unsloth
    from utils.path_utils import PathManager, path_manager
    # 통합 에러 처리 시스템 import
    from utils.error_handling import (
        handle_error, log_structured, log_performance_metric, log_experiment_event,
        safe_execute, get_logging_manager, get_error_handler
    )
except ImportError:
    # code 디렉토리에서 실행되는 경우
    import sys

    sys.path.append("..")
    from utils import load_config
    from utils.data_utils import DataProcessor
    from utils.metrics import RougeCalculator
    from utils.experiment_utils import ExperimentTracker, ModelRegistry
    from utils.environment_detector import EnvironmentDetector, get_auto_config, should_use_unsloth
    from utils.path_utils import PathManager, path_manager
    # 통합 에러 처리 시스템 import
    from utils.error_handling import (
        handle_error, log_structured, log_performance_metric, log_experiment_event,
        safe_execute, get_logging_manager, get_error_handler
    )
logger = logging.getLogger(__name__)


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


class WandbCallback(TrainerCallback):
    """WandB 로깅을 위한 커스텀 콜백"""

    def __init__(self, trainer_instance: "DialogueSummarizationTrainer") -> None:
        self.trainer_instance = trainer_instance
        self.best_metrics = {}

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics: Dict[str, float], **kwargs):
        """평가 시 WandB에 메트릭 로깅"""
        if wandb.run is not None:
            # ROUGE 점수 결합 (F1 기준)
            rouge_combined = (
                metrics.get("eval_rouge1", 0) * 0.33
                + metrics.get("eval_rouge2", 0) * 0.33
                + metrics.get("eval_rougeL", 0) * 0.34
            )

            log_metrics = {
                "eval/rouge1_f1": metrics.get("eval_rouge1", 0),
                "eval/rouge2_f1": metrics.get("eval_rouge2", 0),
                "eval/rougeL_f1": metrics.get("eval_rougeL", 0),
                "eval/rouge_combined_f1": rouge_combined,
                "eval/loss": metrics.get("eval_loss", 0),
                "epoch": state.epoch,
                "step": state.global_step,
            }

            # 베스트 메트릭 업데이트
            if rouge_combined > self.best_metrics.get("rouge_combined_f1", 0):
                self.best_metrics = {
                    "rouge1_f1": metrics.get("eval_rouge1", 0),
                    "rouge2_f1": metrics.get("eval_rouge2", 0),
                    "rougeL_f1": metrics.get("eval_rougeL", 0),
                    "rouge_combined_f1": rouge_combined,
                    "loss": metrics.get("eval_loss", 0),
                }
                log_metrics["best/rouge_combined_f1"] = rouge_combined

            wandb.log(log_metrics)

            # 실험 추적기에도 로깅
            if self.trainer_instance.experiment_tracker:
                self.trainer_instance.experiment_tracker.log_metrics(metrics, step=state.global_step)

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

    def __init__(self, config: Dict[str, Any], sweep_mode: bool = False, experiment_name: Optional[str] = None):
        """
        트레이너 초기화
    
        Args:
            config: 설정 딕셔너리 (ConfigManager로부터)
            sweep_mode: WandB Sweep 모드 여부
            experiment_name: 실험명 (None이면 자동 생성)
        """
        # 통합 로깅 초기화
        self.logging_manager = get_logging_manager()
        
        # 실험 시작 로깅
        log_experiment_event(
            event_type="trainer_init",
            event_data={
                "sweep_mode": sweep_mode,
                "experiment_name": experiment_name,
                "config_keys": list(config.keys()) if config else []
            },
            component="trainer"
        )
        
        self.config = config
        self.sweep_mode = sweep_mode
        self.experiment_name = experiment_name or config.get("meta", {}).get(
            "experiment_name", "dialogue_summarization"
        )
        
        # 실험 ID를 로깅 관리자에 설정 (체크포인트와 연동)
        self.logging_manager.set_experiment_id(self.experiment_name)

        # 디바이스 설정
        self.device = self._setup_device()

        # 경로 설정
        self.setup_paths()

        # 컴포넌트 초기화
        self.model = None
        self.tokenizer = None
        self.data_processor = None
        self.rouge_calculator = None
        self.trainer = None

        # 실험 관리
        self.experiment_tracker = None
        self.model_registry = None

        # 로깅 설정
        self._setup_logging()

        logger.info(f"Trainer initialized with config: {self.experiment_name}")

    def setup_paths(self) -> None:
        """경로 설정"""
        # 경로 관리자를 사용하여 경로 관리
        experiment_name = self.experiment_name

        # Sweep 모드일 때는 run ID를 포함
        if self.sweep_mode and wandb.run:
            experiment_name = f"sweep_{wandb.run.id}"
        else:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"{self.experiment_name}_{timestamp}"

        # 경로 관리자를 통한 경로 설정
        self.output_dir = path_manager.get_output_path(experiment_name)
        self.model_save_dir = path_manager.get_model_path(experiment_name)
        self.results_dir = path_manager.ensure_dir(self.output_dir / "results")

        # 로그 디렉토리 설정
        self.log_dir = path_manager.get_log_path(experiment_name)

        # 파일 핸들러 추가 (output_dir이 설정된 후)
        self._add_file_handler()

    def initialize_components(self) -> None:
        """모든 컴포넌트 초기화"""
        logger.info("Initializing components...")

        # 실험 추적기 초기화
        if self.config.get("experiment_tracking", {}).get("enabled", True):
            self.experiment_tracker = ExperimentTracker(experiments_dir=self.output_dir / "experiments")
            self.model_registry = ModelRegistry(registry_dir=self.output_dir / "models")

        # 토크나이저 로딩
        self._load_tokenizer()

        # 모델 로딩
        self._load_model()

        # 데이터 프로세서 초기화
        self.data_processor = DataProcessor(tokenizer=self.tokenizer, config=self.config)

        # ROUGE 계산기 초기화
        self.rouge_calculator = RougeCalculator(
            use_korean_tokenizer=self.config.get("evaluation", {}).get("rouge_tokenize_korean", True),
            use_stemmer=self.config.get("evaluation", {}).get("rouge_use_stemmer", True),
        )
        logger.info("All components initialized successfully")

    def prepare_data(
        self, train_path: Optional[str] = None, val_path: Optional[str] = None, test_path: Optional[str] = None
    ) -> DatasetDict:
        """
        데이터 준비 - baseline.py 방식으로 수정
        """
        # 경로 결정
        data_paths = self.config.get("data", self.config.get("general", {}))
        train_path = train_path or data_paths.get("train_path")
        val_path = val_path or data_paths.get("val_path")
        test_path = test_path or data_paths.get("test_path")

        logger.info("Loading and processing datasets (baseline style)...")

        datasets = {}

        # Train 데이터 처리
        if train_path:
            logger.info(f"Loading train data from: {train_path}")

            # pandas로 CSV 읽기
            train_df = pd.read_csv(train_path)
            train_df = train_df[["fname", "dialogue", "summary"]]

            # baseline의 make_input 로직
            encoder_inputs = []
            decoder_inputs = []
            decoder_outputs = []

            if self.tokenizer is None:
                raise ValueError("Tokenizer failed to load")
            bos_token = self.tokenizer.bos_token
            eos_token = self.tokenizer.eos_token

            for dialogue, summary in zip(train_df["dialogue"], train_df["summary"]):
                encoder_input = dialogue
                decoder_input = f"{bos_token} {summary}"
                decoder_output = f"{summary} {eos_token}"

                encoder_inputs.append(encoder_input)
                decoder_inputs.append(decoder_input)
                decoder_outputs.append(decoder_output)

            # 토크나이징
            tokenized_encoder = self.tokenizer(
                encoder_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config["tokenizer"]["encoder_max_len"],
                return_token_type_ids=False,
            )

            tokenized_decoder_inputs = self.tokenizer(
                decoder_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config["tokenizer"]["decoder_max_len"],
                return_token_type_ids=False,
            )

            tokenized_decoder_outputs = self.tokenizer(
                decoder_outputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config["tokenizer"]["decoder_max_len"],
                return_token_type_ids=False,
            )

            # HuggingFace Dataset 형식으로 변환
            train_dataset = Dataset.from_dict(
                {
                    "input_ids": tokenized_encoder["input_ids"],
                    "attention_mask": tokenized_encoder["attention_mask"],
                    "decoder_input_ids": tokenized_decoder_inputs["input_ids"],
                    "labels": tokenized_decoder_outputs["input_ids"],
                }
            )

            datasets["train"] = train_dataset
            logger.info(f"Train dataset size: {len(train_dataset)}")

        # Validation 데이터 처리
        if val_path:
            logger.info(f"Loading validation data from: {val_path}")

            # pandas로 CSV 읽기
            val_df = pd.read_csv(val_path)
            val_df = val_df[["fname", "dialogue", "summary"]]

            # baseline의 make_input 로직
            encoder_inputs = []
            decoder_inputs = []
            decoder_outputs = []

            bos_token = self.tokenizer.bos_token
            eos_token = self.tokenizer.eos_token

            for dialogue, summary in zip(val_df["dialogue"], val_df["summary"]):
                encoder_input = dialogue
                decoder_input = f"{bos_token} {summary}"
                decoder_output = f"{summary} {eos_token}"

                encoder_inputs.append(encoder_input)
                decoder_inputs.append(decoder_input)
                decoder_outputs.append(decoder_output)

            # 토크나이징
            tokenized_encoder = self.tokenizer(
                encoder_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config["tokenizer"]["encoder_max_len"],
                return_token_type_ids=False,
            )

            tokenized_decoder_inputs = self.tokenizer(
                decoder_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config["tokenizer"]["decoder_max_len"],
                return_token_type_ids=False,
            )

            tokenized_decoder_outputs = self.tokenizer(
                decoder_outputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config["tokenizer"]["decoder_max_len"],
                return_token_type_ids=False,
            )

            # HuggingFace Dataset 형식으로 변환
            val_dataset = Dataset.from_dict(
                {
                    "input_ids": tokenized_encoder["input_ids"],
                    "attention_mask": tokenized_encoder["attention_mask"],
                    "decoder_input_ids": tokenized_decoder_inputs["input_ids"],
                    "labels": tokenized_decoder_outputs["input_ids"],
                }
            )

            datasets["validation"] = val_dataset
            logger.info(f"Validation dataset size: {len(val_dataset)}")

        # Test 데이터 처리
        if test_path:
            logger.info(f"Loading test data from: {test_path}")

            # pandas로 CSV 읽기
            test_df = pd.read_csv(test_path)
            test_df = test_df[["fname", "dialogue"]]  # test는 summary 없음

            # 토크나이징 (encoder만)
            tokenized_encoder = self.tokenizer(
                test_df["dialogue"].tolist(),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config["tokenizer"]["encoder_max_len"],
                return_token_type_ids=False,
            )

            # HuggingFace Dataset 형식으로 변환
            test_dataset = Dataset.from_dict(
                {"input_ids": tokenized_encoder["input_ids"], "attention_mask": tokenized_encoder["attention_mask"]}
            )

            datasets["test"] = test_dataset
            logger.info(f"Test dataset size: {len(test_dataset)}")

        return DatasetDict(datasets)

    def train(self, dataset: DatasetDict, resume_from_checkpoint: Optional[str] = None) -> TrainingResult:
        """
        모델 학습

        Args:
            dataset: 학습/검증 데이터셋
            resume_from_checkpoint: 체크포인트 경로 (재개 시)

        Returns:
            학습 결과
        """
        # 실험 연속성 추적 시작
        from utils.experiment_continuity import get_continuity_manager
        continuity_manager = get_continuity_manager()
        
        # 실험 ID 생성 (기존 experiment_tracker ID 사용)
        experiment_id = f"exp_{int(time.time())}"
        
        # 체크포인트 초기화
        checkpoint = continuity_manager.start_experiment(
            experiment_id=experiment_id,
            experiment_name=self.experiment_name,
            config=self.config
        )
        
        logger.info(f"📁 실험 연속성 추적 시작: {experiment_id}")
        
        # 실험 시작
        if self.experiment_tracker:
            # WandB 초기화 (조장님 지시사항 반영)
            if self.config["training"].get("report_to", "none") in ["all", "wandb"]:
                from utils.experiment_utils import get_korean_time_format

                # 한국 시간 기준 MMDDHHMM 형식
                korean_time = get_korean_time_format("MMDDHHMM")

                # 모델 아키텍처 첫 글자 추출
                model_arch = self.config.get("model", {}).get("architecture", "unknown")
                model_prefix = model_arch[0] if model_arch != "unknown" else "x"

                # WandB run name 생성
                run_name = f"{model_prefix}_{self.experiment_name}_{korean_time}"

                # 환경 변수 설정
                os.environ["WANDB_LOG_MODEL"] = "false"  # mT5는 크기가 커서 로컬만 저장
                os.environ["TOKENIZERS_PARALLELISM"] = "true"

                # 견고한 WandB 초기화 (네트워크 실패 시 오프라인 모드 자동 전환)
                from utils.wandb_utils import safe_setup_wandb_for_experiment
                
                wandb_result = safe_setup_wandb_for_experiment(
                    config=self.config,
                    experiment_name=self.experiment_name,
                    sweep_mode=self.sweep_mode
                )
                
                # 연결 상태 로깅
                if wandb_result.get('offline_mode', False):
                    logger.warning("🔄 WandB 오프라인 모드에서 실험 진행")
                else:
                    logger.info("✅ WandB 온라인 모드로 실험 시작")

            experiment_id = self.experiment_tracker.start_experiment(
                name=self.experiment_name,
                description=f"Training {self.config.get('model', {}).get('architecture', 'unknown')} model",
                config=self.config,
                model_type=self.config.get("model", {}).get("architecture", "unknown"),
                dataset_info={
                    "train_size": len(dataset.get("train", [])),
                    "val_size": len(dataset.get("validation", [])),
                },
                wandb_run_id=wandb.run.id if wandb.run else None,
            )
        else:
            experiment_id = None

        # 학습 인자 설정
        training_args = self._get_training_arguments()

        # 데이터 콜레이터
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            max_length=self.config["tokenizer"]["encoder_max_len"],
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

            # 🛡️ 안전한 토큰 디코딩 (IndexError 방지)
            def safe_decode(token_ids, tokenizer):
                """토큰 ID 범위를 체크하여 안전하게 디코딩"""
                try:
                    # 토큰 ID가 vocab 범위를 벗어났는지 체크
                    if hasattr(tokenizer, 'vocab_size'):
                        vocab_size = tokenizer.vocab_size
                    else:
                        vocab_size = len(tokenizer.get_vocab())
                    
                    # 범위를 벗어난 토큰 ID를 클램핑
                    if isinstance(token_ids, np.ndarray):
                        token_ids = np.clip(token_ids, 0, vocab_size - 1)
                    
                    return tokenizer.batch_decode(token_ids, skip_special_tokens=True)
                except Exception as e:
                    logger.warning(f"Token decoding failed: {e}. Using fallback.")
                    # 폴백: 빈 문자열 리스트 반환
                    return [""] * len(token_ids)
            
            # 안전한 토큰 디코딩 적용
            decoded_preds = safe_decode(preds, self.tokenizer)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

            # HuggingFace에서 사용하는 -100 패딩 토큰을 정상 토큰으로 변환
            # -100은 loss 계산에서 무시되는 라벨이지만 디코딩에서는 문제가 됨
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            decoded_labels = safe_decode(labels, self.tokenizer)

            # 대화 요약에 특화된 ROUGE 메트릭 계산 (Multi-reference 지원)
            rouge_scores = []
            for pred, ref in zip(decoded_preds, decoded_labels):
                score = self.rouge_calculator.calculate_single_reference(pred, ref)
                rouge_scores.append(score)
            
            # 평균 점수 계산
            avg_rouge1_f1 = np.mean([score.rouge1.f1 for score in rouge_scores])
            avg_rouge2_f1 = np.mean([score.rouge2.f1 for score in rouge_scores])
            avg_rougeL_f1 = np.mean([score.rougeL.f1 for score in rouge_scores])
            avg_combined_f1 = avg_rouge1_f1 + avg_rouge2_f1 + avg_rougeL_f1
            
            result = {
                'rouge1_f1': avg_rouge1_f1,
                'rouge2_f1': avg_rouge2_f1,
                'rougeL_f1': avg_rougeL_f1,
                'rouge_combined_f1': avg_combined_f1,
                'rouge1_precision': np.mean([score.rouge1.precision for score in rouge_scores]),
                'rouge1_recall': np.mean([score.rouge1.recall for score in rouge_scores]),
                'rouge2_precision': np.mean([score.rouge2.precision for score in rouge_scores]),
                'rouge2_recall': np.mean([score.rouge2.recall for score in rouge_scores]),
                'rougeL_precision': np.mean([score.rougeL.precision for score in rouge_scores]),
                'rougeL_recall': np.mean([score.rougeL.recall for score in rouge_scores])
            }

            return result

        # 콜백 설정
        callbacks = [WandbCallback(self)]

        # 조기 종료 설정
        if self.config["training"].get("early_stopping_patience"):
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.config["training"]["early_stopping_patience"],
                    early_stopping_threshold=0.001,
                )
            )

        # 트레이너 생성
        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset.get("train"),
            eval_dataset=dataset.get("validation"),
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )

        # 학습 시작
        logger.info("Starting training...")
        
        # 실험 연속성 import
        from utils.experiment_continuity import save_experiment_checkpoint
        
        # 학습 시작 체크포인트 저장
        training_info = {
            'total_epochs': self.config['training'].get('num_train_epochs', 0),
            'train_dataset_size': len(dataset.get('train', [])),
            'eval_dataset_size': len(dataset.get('validation', [])),
            'batch_size': self.config['training'].get('per_device_train_batch_size', 0)
        }
        save_experiment_checkpoint('training_started', progress_info=training_info)
        logger.info("📋 학습 시작 체크포인트 저장")

        try:
            train_result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)

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
                training_history=[],  # 향후 구현
                wandb_run_id=wandb.run.id if wandb.run else None,
                experiment_id=experiment_id,
            )

            # 실험 종료
            if self.experiment_tracker:
                self.experiment_tracker.end_experiment(
                    experiment_id=experiment_id,
                    final_metrics=eval_results,
                    best_metrics=wandb_callback.best_metrics,
                    status="completed",
                )

            # 모델 등록
            if self.model_registry:
                model_id = self.model_registry.register_model(
                    name=f"{self.config.get('model', {}).get('architecture', 'unknown')}_{self.experiment_name}",
                    architecture=self.config.get("model", {}).get("architecture", "unknown"),
                    checkpoint=self.config.get("model", {}).get(
                        "checkpoint", self.config.get("general", {}).get("model_name", "")
                    ),
                    config=self.config,
                    performance=wandb_callback.best_metrics,
                    training_info={
                        "epochs": self.config["training"]["num_train_epochs"],
                        "batch_size": self.config["training"]["per_device_train_batch_size"],
                        "learning_rate": self.config["training"]["learning_rate"],
                    },
                    file_path=str(best_model_path),
                    experiment_id=experiment_id,
                )
                logger.info(f"Model registered with ID: {model_id}")

            # 결과 저장
            self._save_results(training_result)
            
            # 실험 성공 체크포인트 저장
            from utils.experiment_continuity import get_continuity_manager
            continuity_manager = get_continuity_manager()
            continuity_manager.finish_experiment(success=True, final_metrics=eval_results)
            logger.info("📋 실험 성공 체크포인트 저장")
            
            return training_result

        except Exception as e:
            logger.error(f"Training failed with error: {type(e).__name__}: {str(e)}")
            logger.error(f"Current config: {self.config.get('model', {}).get('checkpoint', 'Unknown')}")
            logger.error(f"Device: {self.device}")
            import traceback

            logger.debug(f"Full traceback: {traceback.format_exc()}")
            logger.error(f"Training failed: {str(e)}")
            
            # 실험 실패 체크포인트 저장
            from utils.experiment_continuity import get_continuity_manager
            continuity_manager = get_continuity_manager()
            continuity_manager.finish_experiment(success=False, final_metrics={'error': str(e)})
            logger.info("📋 실험 실패 체크포인트 저장")
            
            if self.experiment_tracker and experiment_id:
                self.experiment_tracker.end_experiment(experiment_id=experiment_id, status="failed", notes=str(e))
            raise

    def evaluate(self, dataset: Dataset, metric_key_prefix: str = "eval") -> Dict[str, float]:
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

        results = self.trainer.evaluate(eval_dataset=dataset, metric_key_prefix=metric_key_prefix)

        return results

    def generate_predictions(self, dataset: Dataset, max_samples: Optional[int] = None) -> List[Dict[str, str]]:
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
        gen_config = self.config["generation"]

        with torch.no_grad():
            for example in tqdm(dataset, desc="Generating predictions"):
                # 토큰화
                inputs = self.tokenizer(
                    example["input"],
                    max_length=self.config["tokenizer"]["encoder_max_len"],
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)

                # 생성
                outputs = self.model.generate(
                    **inputs,
                    max_length=gen_config["max_length"],
                    num_beams=gen_config["num_beams"],
                    length_penalty=gen_config.get("length_penalty", 1.0),
                    no_repeat_ngram_size=gen_config.get("no_repeat_ngram_size", 2),
                    early_stopping=gen_config.get("early_stopping", True),
                    do_sample=gen_config.get("do_sample", False),
                    temperature=gen_config.get("temperature", 1.0) if gen_config.get("do_sample") else None,
                    top_k=gen_config.get("top_k", 50) if gen_config.get("do_sample") else None,
                    top_p=gen_config.get("top_p", 0.95) if gen_config.get("do_sample") else None,
                )

                # 디코딩
                prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                predictions.append(
                    {"input": example["input"], "prediction": prediction, "reference": example.get("target", "")}
                )

        return predictions

    def _setup_device(self) -> torch.device:
        """디바이스 설정"""
        from utils.device_utils import get_robust_optimal_device, setup_device_config
        
        device_config = self.config["general"].get("device", "auto")
        
        if device_config == "auto":
            # 견고한 최적 디바이스 자동 선택 (컴테이너 환경 대응)
            device, device_info = get_robust_optimal_device()

            # 디바이스별 최적화 설정 가져오기
            model_size = self.config.get("model", {}).get("size", "base")
            optimization_config = setup_device_config(device_info, model_size)

            # 최적화 설정을 config에 병합
            if "training" not in self.config:
                self.config["training"] = {}

            # 기존 설정과 병합 (기존 설정 우선)
            opt_dict = optimization_config.to_dict()
            for key, value in opt_dict.items():
                if key not in self.config["training"]:
                    self.config["training"][key] = value

            logger.info(f"자동 감지된 디바이스: {device_info}")
            logger.info(
                f"최적화 설정 적용됨: batch_size={optimization_config.batch_size}, "
                f"mixed_precision={optimization_config.mixed_precision}, "
                f"num_workers={optimization_config.num_workers}"
            )
        else:
            # 수동 설정
            device = torch.device(device_config)
            logger.info(f"수동 설정된 디바이스: {device}")

        logger.info(f"Using device: {device}")
        return device

    def _setup_logging(self) -> None:
        """로깅 설정"""
        log_level = self.config.get("logging", {}).get("level", "INFO")

        # 기본 로깅 설정 (콘솔만)
        logging.basicConfig(
            level=getattr(logging, log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

        # output_dir이 설정된 후에 파일 핸들러 추가
        if hasattr(self, "output_dir") and self.output_dir:
            self._add_file_handler()

    def _add_file_handler(self) -> None:
        """로깅에 파일 핸들러 추가 (output_dir 설정 후 호출)"""
        if hasattr(self, "output_dir") and self.output_dir:
            # 기존 핸들러 가져오기
            root_logger = logging.getLogger()

            # 파일 핸들러 추가
            file_handler = logging.FileHandler(self.output_dir / "training.log")
            file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            root_logger.addHandler(file_handler)
            logger.info(f"로깅 파일 생성: {self.output_dir / 'training.log'}")

    def _load_tokenizer(self) -> None:
        """토크나이저 로딩 개선"""
        # model 섹션이 없으면 general에서 model_name 사용
        if "model" in self.config:
            model_checkpoint = self.config.get("model", {}).get(
                "checkpoint", self.config.get("general", {}).get("model_name", "")
            )
        else:
            model_checkpoint = self.config.get("general", {}).get("model_name")
            if not model_checkpoint:
                raise ValueError(
                    "Model checkpoint not found in config. Please specify 'model.checkpoint' or 'general.model_name'"
                )

        logger.info(f"Loading tokenizer: {model_checkpoint}")

        try:
            # 모델별 토크나이저 설정
            tokenizer_kwargs = {"trust_remote_code": True, "use_fast": True}

            # T5/mT5 모델 특별 처리
            if "mt5" in model_checkpoint.lower() or "t5" in model_checkpoint.lower():
                tokenizer_kwargs["use_fast"] = False  # T5/mT5는 SentencePiece로 use_fast=False 사용
                tokenizer_kwargs["legacy"] = False  # T5 legacy 모드 비활성화

            # 안전한 토크나이저 로딩 (네트워크 실패 시 캐시 활용)
            from utils.model_loading_utils import safe_load_tokenizer
            
            self.tokenizer = safe_load_tokenizer(model_checkpoint, **tokenizer_kwargs)
            
            # None 반환 시 직접 로딩 시도 (특정 모델 호환성 문제 해결)
            if self.tokenizer is None:
                logger.warning(f"safe_load_tokenizer 실패, 직접 로딩 시도: {model_checkpoint}")
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, **tokenizer_kwargs)
                except Exception as direct_error:
                    logger.error(f"직접 로딩도 실패: {direct_error}")
                    # 대체 모델로 폴백 (KoBART 기본)
                    logger.warning("기본 KoBART 토크나이저로 폴백")
                    self.tokenizer = AutoTokenizer.from_pretrained("digit82/kobart-summarization")

            # 특수 토큰 설정 (필요시)
            model_architecture = self.config.get("model", {}).get("architecture", "")
            if model_architecture in ["kogpt2", "gpt2"]:
                # GPT 계열은 pad_token이 없을 수 있음
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info(f"✅ 토크나이저 로딩 성공: {type(self.tokenizer).__name__}")

        except Exception as e:
            logger.error(f"❌ 토크나이저 로딩 실패: {e}")
            raise

    def _get_architecture_from_model_name(self) -> str:
        """모델 이름으로부터 아키텍처 추론"""
        model_name = self.config.get("general", {}).get("model_name", "").lower()

        if "bart" in model_name or "kobart" in model_name:
            return "bart"
        elif "mt5" in model_name:
            return "mt5"
        elif "t5" in model_name:
            return "t5"
        elif "gpt" in model_name:
            return "gpt2"
        else:
            return "bart"  # 기본값

    def _load_model(self) -> None:
        """모델 로딩 (unsloth 및 QLoRA 지원)"""
        model_checkpoint = self.config.get("model", {}).get(
            "checkpoint", self.config.get("general", {}).get("model_name", "")
        )
        architecture = self.config.get("model", {}).get("architecture", "bart")

        # QLoRA 설정 확인
        qlora_config = self.config.get("qlora", {})
        use_unsloth = qlora_config.get("use_unsloth", False) and UNSLOTH_AVAILABLE
        use_qlora = qlora_config.get("use_qlora", False)

        logger.info(f"Loading model: {model_checkpoint} ({architecture})")
        logger.info(f"QLoRA enabled: {use_qlora}, unsloth enabled: {use_unsloth}")

        if use_unsloth and architecture in ["kobart", "bart", "t5", "mt5"]:
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
        if self.config["training"].get("gradient_checkpointing", False):
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
            else:
                logger.warning("모델이 gradient_checkpointing을 지원하지 않습니다.")
        
        # 모델 로딩 완료 체크포인트 저장
        from utils.experiment_continuity import save_experiment_checkpoint
        model_info = {
            'architecture': architecture,
            'checkpoint': model_checkpoint,
            'model_loaded': True,
            'use_qlora': use_qlora,
            'use_unsloth': use_unsloth,
            'parameters': sum(p.numel() for p in self.model.parameters()) if hasattr(self, 'model') else 0
        }
        save_experiment_checkpoint('model_loaded', model_info=model_info)
        logger.info("📋 모델 로딩 완료 체크포인트 저장")

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
                max_seq_length=self.config["tokenizer"].get("encoder_max_len", 512)
                + self.config["tokenizer"].get("decoder_max_len", 200),
                dtype=torch.float16 if self.config["training"].get("fp16") else torch.float32,
                load_in_4bit=qlora_config.get("load_in_4bit", True),
            )

            # LoRA 설정 추가
            model = FastLanguageModel.get_peft_model(
                model,
                r=qlora_config.get("lora_rank", 16),
                target_modules=qlora_config.get(
                    "target_modules", ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
                ),
                lora_alpha=qlora_config.get("lora_alpha", 32),
                lora_dropout=qlora_config.get("lora_dropout", 0.1),
                bias="none",
                use_gradient_checkpointing="unsloth",  # unsloth 최적화
                random_state=42,
            )

            self.model = model
            logger.info("✅ unsloth 모델 로딩 성공! 메모리 사용량 75% 감소 예상")

        except Exception as e:
            logger.error(f"❌ unsloth 모델 로딩 실패: {e}")
            logger.info("폴백 모드: 일반 QLoRA로 대체")
            self._load_model_with_qlora(model_checkpoint, "kobart", qlora_config)

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
                load_in_4bit=qlora_config.get("load_in_4bit", True),
                bnb_4bit_compute_dtype=getattr(torch, qlora_config.get("bnb_4bit_compute_dtype", "bfloat16")),
                bnb_4bit_quant_type=qlora_config.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_use_double_quant=qlora_config.get("bnb_4bit_use_double_quant", True),
            )
            
            # 안전한 모델 로딩 (네트워크 실패 시 캐시 활용)
            from utils.model_loading_utils import safe_load_model
            
            if architecture in ["kobart", "bart", "t5", "mt5"]:
                model = safe_load_model(
                    AutoModelForSeq2SeqLM,
                    model_checkpoint,
                    quantization_config=bnb_config,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                )
            else:
                model = safe_load_model(
                    AutoModelForCausalLM,
                    model_checkpoint,
                    quantization_config=bnb_config,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                )

            # LoRA 설정 - PEFT 어댑터 반드시 적용
            if LoraConfig is not None:
                lora_config = LoraConfig(
                    r=qlora_config.get("lora_rank", 64),
                    lora_alpha=qlora_config.get("lora_alpha", 128),
                    target_modules=qlora_config.get("target_modules", ["q_proj", "k_proj", "v_proj", "out_proj"]),
                    lora_dropout=qlora_config.get("lora_dropout", 0.05),
                    bias="none",
                    task_type=(
                        TaskType.SEQ_2_SEQ_LM if architecture in ["kobart", "bart", "t5", "mt5"] else TaskType.CAUSAL_LM
                    ),
                )

                # PEFT 모델 생성 - 이 부분이 핵심
                model = get_peft_model(model, lora_config)
                model.print_trainable_parameters()  # 학습 가능 파라미터 확인
                logger.info("✅ QLoRA + PEFT 모델 준비 완료!")
            else:
                raise ImportError("PEFT 라이브러리를 찾을 수 없습니다")

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
        표준 모델 로딩 (기존 방식)
    
        Args:
            model_checkpoint: 모델 체크포인트 경로
            architecture: 모델 아키텍처
        """
        logger.info("📚 표준 모델 로딩 중...")
    
        # 안전한 모델 로딩 (네트워크 실패 시 캐시 활용)
        from utils.model_loading_utils import safe_load_model
        
        # 모델 아키텍처에 따른 로딩
        if architecture in ["kobart", "bart", "t5", "mt5"]:
            # 시퀀스-투-시퀀스 모델
            self.model = safe_load_model(
                AutoModelForSeq2SeqLM,
                model_checkpoint, 
                torch_dtype=torch.float16 if self.config["training"].get("fp16") else torch.float32
            )
        elif architecture in ["kogpt2", "gpt2", "gpt-neo"]:
            # 인과 언어 모델
            self.model = safe_load_model(
                AutoModelForCausalLM,
                model_checkpoint, 
                torch_dtype=torch.float16 if self.config["training"].get("fp16") else torch.float32
            )
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
    
            logger.info("✅ 표준 모델 로딩 성공")
    

    def _get_model_specific_config(self, architecture: str, checkpoint: str) -> Dict[str, Any]:
        """모델별 특수 설정 반환"""
        config = {}

        # T5 계열
        if architecture in ["t5", "mt5", "flan-t5"]:
            config["prefix"] = "summarize: "  # T5는 task prefix 필요

        # GPT 계열
        elif architecture in ["gpt2", "kogpt2", "gpt-neo"]:
            config["max_length"] = (
                self.config["tokenizer"]["encoder_max_len"] + self.config["tokenizer"]["decoder_max_len"]
            )
            config["pad_token_id"] = self.tokenizer.pad_token_id

        return config

    def _preprocess_for_model(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """모델별 데이터 전처리"""
        architecture = self.config.get("model", {}).get("architecture", self._get_architecture_from_model_name())

        if architecture in ["t5", "mt5", "flan-t5"]:
            # T5는 prefix 추가
            examples["input"] = ["summarize: " + inp for inp in examples["input"]]

        elif architecture in ["gpt2", "kogpt2", "gpt-neo"]:
            # GPT는 입력과 타겟을 연결
            examples["input"] = [f"{inp} TL;DR: {tgt}" for inp, tgt in zip(examples["input"], examples["target"])]

        return examples

    def _get_training_arguments(self) -> Seq2SeqTrainingArguments:
        """학습 인자 생성"""
        train_config = self.config["training"]

        # 기본 인자
        args_dict = {
            "output_dir": str(self.output_dir / "checkpoints"),
            "overwrite_output_dir": True,
            "do_train": True,
            "do_eval": True,
            "eval_strategy": train_config.get("eval_strategy", "steps"),
            "eval_steps": train_config.get("eval_steps", 500),
            "save_strategy": train_config.get("save_strategy", "steps"),
            "save_steps": train_config.get("save_steps", 500),
            "save_total_limit": train_config.get("save_total_limit", 3),
            "per_device_train_batch_size": train_config["per_device_train_batch_size"],
            "per_device_eval_batch_size": train_config.get(
                "per_device_eval_batch_size", train_config["per_device_train_batch_size"]
            ),
            "gradient_accumulation_steps": train_config.get("gradient_accumulation_steps", 1),
            "learning_rate": train_config["learning_rate"],
            "weight_decay": train_config.get("weight_decay", 0.01),
            "adam_beta1": train_config.get("adam_beta1", 0.9),
            "adam_beta2": train_config.get("adam_beta2", 0.999),
            "adam_epsilon": train_config.get("adam_epsilon", 1e-8),
            "max_grad_norm": train_config.get("max_grad_norm", 1.0),
            "num_train_epochs": train_config["num_train_epochs"],
            "lr_scheduler_type": train_config.get("lr_scheduler_type", "linear"),
            "warmup_ratio": train_config.get("warmup_ratio", 0.1),
            "warmup_steps": train_config.get("warmup_steps", 0),
            "logging_dir": str(self.output_dir / "logs"),
            "logging_steps": train_config.get("logging_steps", 50),
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_rouge_combined_f1",
            "greater_is_better": True,
            "fp16": train_config.get("fp16", False),
            "fp16_opt_level": train_config.get("fp16_opt_level", "O1"),
            "dataloader_num_workers": train_config.get("dataloader_num_workers", 4),
            "remove_unused_columns": False,
            "seed": self.config["general"].get("seed", 42),
            "report_to": ["wandb"] if wandb.run else ["none"],
            "run_name": self.experiment_name if wandb.run else None,
            "push_to_hub": False,
            "predict_with_generate": True,
            "generation_max_length": self.config["generation"]["max_length"],
            "generation_num_beams": self.config["generation"]["num_beams"],
        }
        
        # save_steps와 eval_steps 동기화 (load_best_model_at_end 사용 시 필수)
        if args_dict.get("load_best_model_at_end"):
            eval_steps = args_dict.get("eval_steps", 500)
            save_steps = args_dict.get("save_steps", 500)
            
            # save_steps가 eval_steps의 배수가 아니면 동기화
            if save_steps % eval_steps != 0:
                args_dict["save_steps"] = eval_steps
                logger.warning(f"save_steps({save_steps})가 eval_steps({eval_steps})의 배수가 아니어서 {eval_steps}로 조정")
        
        # 시퀀스-투-시퀀스 특화 인자
        seq2seq_args = Seq2SeqTrainingArguments(**args_dict)

        return seq2seq_args

    def _save_results(self, result: TrainingResult) -> None:
        """결과 저장"""
        # 결과 딕셔너리 생성
        results_dict = {
            "experiment_name": self.experiment_name,
            "model_architecture": self.config.get("model", {}).get("architecture", "unknown"),
            "model_checkpoint": self.config.get("model", {}).get(
                "checkpoint", self.config.get("general", {}).get("model_name", "")
            ),
            "best_metrics": result.best_metrics,
            "final_metrics": result.final_metrics,
            "model_path": result.model_path,
            "wandb_run_id": result.wandb_run_id,
            "experiment_id": result.experiment_id,
            "config": result.config_used,
            "timestamp": str(Path(result.model_path).parent.parent.name) if result.model_path else "unknown",
        }

        # JSON 저장
        results_file = self.results_dir / "training_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2)

        # 요약 텍스트 저장
        summary_file = self.results_dir / "summary.txt"
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(f"Training Summary for {self.experiment_name}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model: {self.config.get('model', {}).get('architecture', 'unknown')} ({self.config.get('model', {}).get('checkpoint', 'unknown')})\n")
            f.write(f"Training Epochs: {self.config.get('training', {}).get('num_train_epochs', 'unknown')}\n")
            f.write(f"Batch Size: {self.config.get('training', {}).get('per_device_train_batch_size', 'unknown')}\n")
            f.write(f"Learning Rate: {self.config.get('training', {}).get('learning_rate', 'unknown')}\n\n")
            f.write("Best Metrics:\n")
            for metric, value in result.best_metrics.items():
                f.write(f"  {metric}: {value:.4f}\n")
            f.write("\nModel saved to: " + result.model_path + "\n")
            if result.wandb_run_id:
                f.write(f"WandB Run ID: {result.wandb_run_id}\n")

        logger.info(f"Results saved to {self.results_dir}")


def create_trainer(config: Union[str, Dict[str, Any]], sweep_mode: bool = False, one_epoch_mode: bool = False) -> DialogueSummarizationTrainer:
    """
    트레이너 생성 편의 함수
    
    Args:
        config: 설정 파일 경로 또는 설정 딕셔너리
        sweep_mode: WandB Sweep 모드 여부
        one_epoch_mode: 1에포크 모드 여부 (빠른 테스트용)
    
    Returns:
        초기화된 트레이너 인스턴스
    """
    # 설정 로딩
    if isinstance(config, str):
        config_dict = load_config(config)
    else:
        config_dict = config
    
    # 1에포크 모드 적용
    if one_epoch_mode:
        original_epochs = config_dict["training"].get("num_train_epochs", 3)
        config_dict["training"]["num_train_epochs"] = 1
        logger.info(f"🚀 1에포크 모드 활성화: {original_epochs}에포크 → 1에포크로 단축")
    
    # 트레이너 생성
    trainer = DialogueSummarizationTrainer(config=config_dict, sweep_mode=sweep_mode)

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
    parser.add_argument("--one-epoch", action="store_true", help="Run only one epoch for quick testing")

    args = parser.parse_args()

    # WandB 초기화 (비 Sweep 모드)
    if not args.sweep:
        # 기존 세션 정리
        if wandb.run is not None:
            wandb.finish()
    
        wandb.init(
            project="nlp-dialogue-summarization",
            name="manual_training",
            config={"manual_run": True},
            reinit=True,
            resume="never",
        )
    
    # 트레이너 생성 및 학습
    trainer = create_trainer(args.config, sweep_mode=args.sweep, one_epoch_mode=args.one_epoch)

    # 데이터 준비
    datasets = trainer.prepare_data(train_path=args.train_data, val_path=args.val_data, test_path=args.test_data)

    # 학습 실행
    result = trainer.train(datasets)

    print(f"Training completed! Best ROUGE combined F1: {result.best_metrics.get('rouge_combined_f1', 0):.4f}")
    
