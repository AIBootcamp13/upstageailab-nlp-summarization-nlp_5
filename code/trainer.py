"""
NLP 대화 요약 모델 학습 모듈

baseline.ipynb의 핵심 학습 로직을 모듈화한 트레이너 클래스.
WandB Sweep과의 통합을 위해 설계되었으며, 다양한 모델과 설정을 지원합니다.
"""

import os
import sys
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
    from utils.wandb_utils import setup_wandb_for_experiment, log_model_to_wandb
except ImportError:
    # code 디렉토리에서 실행되는 경우
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils import load_config
    from utils.data_utils import DataProcessor
    from utils.metrics import RougeCalculator
    from utils.experiment_utils import ExperimentTracker, ModelRegistry
    from utils.environment_detector import EnvironmentDetector, get_auto_config, should_use_unsloth
    from utils.path_utils import PathManager, path_manager
    from utils.wandb_utils import setup_wandb_for_experiment, log_model_to_wandb
logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """학습 결과 데이터 클래스"""
    model_path: str
    best_metrics: Dict[str, float]
    final_metrics: Dict[str, float]
    wandb_run_id: Optional[str] = None
    experiment_id: Optional[str] = None
    config_used: Optional[Dict[str, Any]] = None
    submission_path: Optional[str] = None  # 제출 파일 경로 추가


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
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
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
            use_korean_tokenizer=self.config.get("evaluation", {}).get("rouge_tokenize_korean", True),
            use_stemmer=self.config.get("evaluation", {}).get("rouge_use_stemmer", True)
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
        data_paths = self.config.get('data', {})
        
        train_path = train_path or data_paths.get('train_path')
        val_path = val_path or data_paths.get('val_path')
        test_path = test_path or data_paths.get('test_path')
        
        logger.info("Loading and processing datasets...")
        
        datasets = {}
        
        if train_path:
            train_data = self.data_processor.load_data(train_path)
            datasets['train'] = self.data_processor.process_data(
                train_data, 
                is_training=True
            )
            logger.info(f"Train dataset size: {len(datasets['train'])}")
        
        if val_path:
            val_data = self.data_processor.load_data(val_path)
            datasets['validation'] = self.data_processor.process_data(
                val_data,
                is_training=False
            )
            logger.info(f"Validation dataset size: {len(datasets['validation'])}")
        
        if test_path:
            test_data = self.data_processor.load_data(test_path)
            datasets['test'] = self.data_processor.process_data(
                test_data,
                is_training=False
            )
            logger.info(f"Test dataset size: {len(datasets['test'])}")
        
        return DatasetDict(datasets)
    
    def train(self, dataset: DatasetDict, 
             resume_from_checkpoint: Optional[str] = None) -> TrainingResult:
        """
        학습 실행
        
        Args:
            dataset: 학습/검증 데이터셋
            resume_from_checkpoint: 체크포인트 경로 (재개 시)
            
        Returns:
            학습 결과
        """
        # WandB 초기화 (sweep 모드가 아니고 이미 초기화되지 않은 경우)
        if not self.sweep_mode and wandb.run is None:
            wandb_config = setup_wandb_for_experiment(
                config=self.config,
                experiment_name=self.experiment_name,
                sweep_mode=self.sweep_mode
            )
            wandb.init(**wandb_config)
            logger.info(f"WandB run initialized: {wandb.run.name}")
        
        # 실험 추적 시작
        experiment_id = None
        if self.experiment_tracker:
            experiment_id = self.experiment_tracker.start_experiment(
                name=self.experiment_name,
                description=f"Training {self.config['model']['architecture']} model",
                config=self.config,
                model_type=self.config['model']['architecture'],
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
        
        # 데이터 콜레이터
        data_collator = DataCollatorForSeq2Seq(
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
            decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            # HuggingFace에서 사용하는 -100 패딩 토큰을 정상 토큰으로 변환
            # -100은 loss 계산에서 무시되는 라벨이지만 디코딩에서는 문제가 됨
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            # 대화 요약에 특화된 ROUGE 메트릭 계산 (Multi-reference 지원)
            result = self.rouge_calculator.compute_metrics(decoded_preds, decoded_labels)
            
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
            # 최종 평가
            logger.info("Running final evaluation...")
            eval_results = self.trainer.evaluate()
            
            # 평가 결과를 eval_results.json으로 저장
            eval_results_file = self.model_save_dir / 'eval_results.json'
            with open(eval_results_file, 'w', encoding='utf-8') as f:
                json.dump(eval_results, f, ensure_ascii=False, indent=2)
            logger.info(f"Evaluation results saved to {eval_results_file}")
            
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
                    name=f"{self.config['model']['architecture']}_{self.experiment_name}",
                    architecture=self.config['model']['architecture'],
                    checkpoint=self.config['model']['checkpoint'],
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
                
                # WandB Model Registry에 모델 등록
                if wandb.run is not None:
                    log_model_to_wandb(
                        model_path=str(best_model_path),
                        model_name=f"{self.config['model']['architecture']}_{self.experiment_name}",
                        metrics=wandb_callback.best_metrics,
                        config=self.config,
                        aliases=["latest", "best"] if wandb_callback.best_metrics.get('rouge_combined_f1', 0) > 0.3 else ["latest"]
                    )
                
                # 결과 저장
            self._save_results(training_result)
            
            # test.csv에 대한 자동 추론 수행
            if self.config.get('inference', {}).get('run_test_inference', True):
                try:
                    logger.info("Running inference on test.csv...")
                    from post_training_inference import generate_submission_after_training
                    
                    submission_path = generate_submission_after_training(
                        experiment_name=self.experiment_name,
                        model_path=str(best_model_path),
                        config_dict=self.config
                    )
                    
                    training_result.submission_path = submission_path
                    logger.info(f"Test inference completed. Submission file: {submission_path}")
                    
                except Exception as e:
                    logger.warning(f"Failed to run test inference: {e}")
                    # Test inference 실패는 전체 학습을 실패로 처리하지 않음
            
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
            use_qlora = self.config.get('qlora', {}).get('use_qlora', False)
            optimization_config = setup_device_config(device_info, model_size, use_qlora)
            
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
        
        # 기본 로깅 설정 (콘솔만)
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # output_dir이 설정된 후에 파일 핸들러 추가
        if hasattr(self, 'output_dir') and self.output_dir:
            self._add_file_handler()
    
    def _add_file_handler(self) -> None:
        """로깅에 파일 핸들러 추가 (output_dir 설정 후 호출)"""
        if hasattr(self, 'output_dir') and self.output_dir:
            # 기존 핸들러 가져오기
            root_logger = logging.getLogger()
            
            # 파일 핸들러 추가
            file_handler = logging.FileHandler(self.output_dir / 'training.log')
            file_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            root_logger.addHandler(file_handler)
            logger.info(f"로깅 파일 생성: {self.output_dir / 'training.log'}")
    
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
        
        # 모델별 토크나이저 설정
        use_fast = True
        if 'mt5' in model_checkpoint.lower() or 't5' in model_checkpoint.lower():
            use_fast = False  # T5/mT5는 SentencePiece로 use_fast=False 사용
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_checkpoint,
            use_fast=use_fast
        )
        
        # 특수 토큰 설정 (필요시)
        if self.config['model']['architecture'] in ['kogpt2', 'gpt2']:
            # GPT 계열은 pad_token이 없을 수 있음
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _load_model(self) -> None:
        """모델 로딩 (unsloth 및 QLoRA 지원)"""
        model_checkpoint = self.config['model']['checkpoint']
        architecture = self.config['model']['architecture']
        
        # QLoRA 설정 확인
        qlora_config = self.config.get('qlora', {})
        use_unsloth = qlora_config.get('use_unsloth', False) and UNSLOTH_AVAILABLE
        use_qlora = qlora_config.get('use_qlora', False)
        
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
        표준 모델 로딩 (기존 방식)
        
        Args:
            model_checkpoint: 모델 체크포인트 경로
            architecture: 모델 아키텍처
        """
        logger.info("📚 표준 모델 로딩 중...")
        
        # 모델 아키텍처에 따른 로딩
        if architecture in ['kobart', 'bart', 't5', 'mt5']:
            # 시퀀스-투-시퀀스 모델
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_checkpoint,
                torch_dtype=torch.float16 if self.config['training'].get('fp16') else torch.float32
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
            'num_train_epochs': train_config['num_train_epochs'],
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
            'generation_max_length': self.config['generation']['max_length'],
            'generation_num_beams': self.config['generation']['num_beams']
        }
        
        # 시퀀스-투-시퀀스 특화 인자
        seq2seq_args = Seq2SeqTrainingArguments(**args_dict)
        
        return seq2seq_args
    
    def _save_results(self, result: TrainingResult) -> None:
        """결과 저장"""
        # 결과 딕셔너리 생성
        results_dict = {
            'experiment_name': self.experiment_name,
            'model_architecture': self.config['model']['architecture'],
            'model_checkpoint': self.config['model']['checkpoint'],
            'best_metrics': result.best_metrics,
            'final_metrics': result.final_metrics,
            'model_path': result.model_path,
            'wandb_run_id': result.wandb_run_id,
            'experiment_id': result.experiment_id,
            'config': result.config_used,
            'timestamp': str(Path(result.model_path).parent.parent.name),
            'submission_path': result.submission_path  # 제출 파일 경로 추가
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
            f.write(f"Model: {self.config['model']['architecture']} ({self.config['model']['checkpoint']})\n")
            f.write(f"Training Epochs: {self.config['training']['num_train_epochs']}\n")
            f.write(f"Batch Size: {self.config['training']['per_device_train_batch_size']}\n")
            f.write(f"Learning Rate: {self.config['training']['learning_rate']}\n\n")
            f.write("Best Metrics:\n")
            for metric, value in result.best_metrics.items():
                f.write(f"  {metric}: {value:.4f}\n")
            f.write("\nModel saved to: " + result.model_path + "\n")
            if result.wandb_run_id:
                f.write(f"WandB Run ID: {result.wandb_run_id}\n")
        
        logger.info(f"Results saved to {self.results_dir}")


def create_trainer(config: Union[str, Dict[str, Any]], 
                  sweep_mode: bool = False,
                  disable_eval: bool = False) -> DialogueSummarizationTrainer:
    """
    트레이너 생성 편의 함수
    
    Args:
        config: 설정 파일 경로 또는 설정 딕셔너리
        sweep_mode: WandB Sweep 모드 여부
        disable_eval: 평가 비활성화 여부
        
    Returns:
        초기화된 트레이너 인스턴스
    """
    # 설정 로딩
    if isinstance(config, str):
        config_dict = load_config(config)
    else:
        config_dict = config
    
    # 평가 비활성화 옵션 처리
    if disable_eval:
        if 'training' not in config_dict:
            config_dict['training'] = {}
        config_dict['training']['do_eval'] = False
        config_dict['training']['evaluation_strategy'] = 'no'
        config_dict['training']['load_best_model_at_end'] = False  # best model 로드 비활성화
        config_dict['training']['metric_for_best_model'] = None  # best model 메트릭 비활성화
        print("⚠️  평가 비활성화: evaluation_strategy=no, do_eval=False, load_best_model_at_end=False")
    
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
    parser.add_argument("--disable-eval", action="store_true", help="Disable evaluation (for 1-epoch mode)")
    parser.add_argument("--one-epoch", action="store_true", help="Run training for only 1 epoch (for testing)")
    
    args = parser.parse_args()
    
    # WandB 초기화 (비 Sweep 모드)
    if not args.sweep:
        wandb.init(
            project="nlp-dialogue-summarization",
            name="manual_training",
            config={"manual_run": True}
        )
    
    # 트레이너 생성 및 학습
    trainer = create_trainer(args.config, sweep_mode=args.sweep, disable_eval=args.disable_eval)
    
    # 1에포크 모드 처리
    if args.one_epoch:
        logger.info("1에포크 모드 활성화: 학습 에포크를 1로 설정")
        trainer.config['training']['num_epochs'] = 1
        trainer.config['training']['max_steps'] = None  # max_steps 비활성화
        trainer.config['training']['evaluation_strategy'] = "no"  # 평가 비활성화
        trainer.config['training']['save_strategy'] = "epoch"  # 에포크마다 저장
        trainer.config['training']['logging_steps'] = 10  # 로깅 빈도 증가
        trainer.config['training']['load_best_model_at_end'] = False  # best model 로드 비활성화
        trainer.config['training']['metric_for_best_model'] = None  # best model 메트릭 비활성화
    
    # 데이터 준비
    datasets = trainer.prepare_data(
        train_path=args.train_data,
        val_path=args.val_data,
        test_path=args.test_data
    )
    
    # 학습 실행
    result = trainer.train(datasets)
    
    print(f"Training completed! Best ROUGE combined F1: {result.best_metrics.get('rouge_combined_f1', 0):.4f}")
