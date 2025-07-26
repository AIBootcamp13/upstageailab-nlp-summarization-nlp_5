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

from datasets import Dataset, DatasetDict
import evaluate
import wandb

# 로컬 유틸리티 임포트
from utils.config_manager import ConfigManager
from utils.data_utils import DataProcessor
from utils.metrics import RougeCalculator
from utils.experiment_utils import ExperimentTracker, ModelRegistry


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
    
    def __init__(self, trainer_instance):
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
    
    baseline.ipynb의 학습 로직을 모듈화하고 WandB Sweep과 통합
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
        
    def setup_paths(self):
        """경로 설정"""
        base_output_dir = Path(self.config['general']['output_dir'])
        
        # Sweep 모드일 때는 run ID를 포함
        if self.sweep_mode and wandb.run:
            self.output_dir = base_output_dir / f"sweep_{wandb.run.id}"
        else:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.output_dir = base_output_dir / f"{self.experiment_name}_{timestamp}"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 하위 디렉토리
        self.model_save_dir = self.output_dir / "models"
        self.results_dir = self.output_dir / "results"
        
        for dir_path in [self.model_save_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def initialize_components(self):
        """모든 컴포넌트 초기화"""
        logger.info("Initializing components...")
        
        # 실험 추적기 초기화
        if self.config.get('experiment_tracking', {}).get('enabled', True):
            self.experiment_tracker = ExperimentTracker(
                experiments_dir=self.output_dir / "experiments"
            )
            self.model_registry = ModelRegistry(
                models_dir=self.output_dir / "models"
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
            tokenizer=self.tokenizer,
            use_stemmer=self.config.get('evaluation', {}).get('rouge_use_stemmer', True),
            tokenize_korean=self.config.get('evaluation', {}).get('rouge_tokenize_korean', True)
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
        
        # 평가 메트릭 함수
        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            
            # 디코딩
            decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            # -100 처리 (패딩)
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            # ROUGE 계산
            result = self.rouge_calculator.compute_metrics(decoded_preds, decoded_labels)
            
            return result
        
        # 콜백 설정
        callbacks = [WandbCallback(self)]
        
        # Early Stopping 설정
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
            
            # 결과 저장
            self._save_results(training_result)
            
            return training_result
            
        except Exception as e:
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
        device_config = self.config['general'].get('device', 'auto')
        
        if device_config == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device_config)
        
        logger.info(f"Using device: {device}")
        return device
    
    def _setup_logging(self):
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
    
    def _load_tokenizer(self):
        """토크나이저 로딩"""
        model_checkpoint = self.config['model']['checkpoint']
        logger.info(f"Loading tokenizer: {model_checkpoint}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_checkpoint,
            use_fast=True
        )
        
        # 특수 토큰 설정 (필요시)
        if self.config['model']['architecture'] in ['kogpt2', 'gpt2']:
            # GPT 계열은 pad_token이 없을 수 있음
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _load_model(self):
        """모델 로딩"""
        model_checkpoint = self.config['model']['checkpoint']
        architecture = self.config['model']['architecture']
        
        logger.info(f"Loading model: {model_checkpoint} ({architecture})")
        
        # 모델 아키텍처에 따른 로딩
        if architecture in ['kobart', 'bart', 't5', 'mt5']:
            # Seq2Seq 모델
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_checkpoint,
                torch_dtype=torch.float16 if self.config['training'].get('fp16') else torch.float32
            )
        elif architecture in ['kogpt2', 'gpt2', 'gpt-neo']:
            # Causal LM 모델
            self.model = AutoModelForCausalLM.from_pretrained(
                model_checkpoint,
                torch_dtype=torch.float16 if self.config['training'].get('fp16') else torch.float32
            )
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        
        # 디바이스로 이동
        self.model = self.model.to(self.device)
        
        # Gradient checkpointing (메모리 최적화)
        if self.config['training'].get('gradient_checkpointing', False):
            self.model.gradient_checkpointing_enable()
    
    def _get_training_arguments(self) -> Seq2SeqTrainingArguments:
        """학습 인자 생성"""
        train_config = self.config['training']
        
        # 기본 인자
        args_dict = {
            'output_dir': str(self.output_dir / 'checkpoints'),
            'overwrite_output_dir': True,
            'do_train': True,
            'do_eval': True,
            'evaluation_strategy': train_config.get('evaluation_strategy', 'steps'),
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
        
        # Seq2Seq 특화 인자
        seq2seq_args = Seq2SeqTrainingArguments(**args_dict)
        
        return seq2seq_args
    
    def _save_results(self, result: TrainingResult):
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
        config_manager = ConfigManager()
        config_dict = config_manager.load_config(config)
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
    
    # WandB 초기화 (비 Sweep 모드)
    if not args.sweep:
        wandb.init(
            project="nlp-dialogue-summarization",
            name="manual_training",
            config={"manual_run": True}
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
