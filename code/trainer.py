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

from code.utils import load_config
from code.utils.data_utils import DataProcessor
from code.utils.metrics import RougeCalculator
from code.utils.experiment_utils import ExperimentTracker, ModelRegistry
from code.utils.path_utils import PathManager, path_manager

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

# Device-specific imports
import platform
if platform.system() == "Darwin":  # macOS
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# WandB
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    
# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """학습 설정 데이터 클래스"""
    # 모델 관련
    model_name: str = "gogamza/kobart-base-v2"
    use_unsloth: bool = False
    use_qlora: bool = False
    model_type: str = "seq2seq"  # seq2seq or causal_lm
    
    # 학습 하이퍼파라미터
    num_train_epochs: int = 3
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 500
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # 데이터 관련
    max_input_length: int = 512
    max_target_length: int = 128
    num_beams: int = 4
    
    # 저장 및 평가
    output_dir: str = "./outputs"
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_rouge_l"
    greater_is_better: bool = True
    
    # 기타
    seed: int = 42
    fp16: bool = False
    push_to_hub: bool = False
    gradient_checkpointing: bool = False
    optim: str = "adamw_torch"
    
    # QLoRA 설정
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # 실험 관련
    experiment_name: str = "nmt_baseline"
    run_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    # Device 설정
    device: Optional[str] = None
    no_cuda: bool = False
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """딕셔너리로부터 설정 생성"""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})


class CustomMetricsCallback(TrainerCallback):
    """커스텀 메트릭 계산을 위한 콜백"""
    
    def __init__(self, trainer_instance):
        self.trainer_instance = trainer_instance
        
    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        """평가 시 추가 메트릭 계산"""
        if metrics is not None:
            # ROUGE 점수를 백분율로 변환
            for key in metrics:
                if 'rouge' in key and not key.endswith('_percent'):
                    metrics[f"{key}_percent"] = metrics[key] * 100
                    
            # 주요 메트릭 로깅
            logger.info(f"Evaluation at step {state.global_step}:")
            logger.info(f"  - ROUGE-1: {metrics.get('eval_rouge_1', 0)*100:.2f}%")
            logger.info(f"  - ROUGE-2: {metrics.get('eval_rouge_2', 0)*100:.2f}%")
            logger.info(f"  - ROUGE-L: {metrics.get('eval_rouge_l', 0)*100:.2f}%")
            logger.info(f"  - Loss: {metrics.get('eval_loss', 0):.4f}")


class NMTTrainer:
    """대화 요약 모델 학습을 위한 트레이너 클래스"""
    
    def __init__(self, config: Union[Dict, TrainingConfig], wandb_config: Optional[Dict] = None):
        """
        Args:
            config: 학습 설정 (딕셔너리 또는 TrainingConfig 객체)
            wandb_config: WandB 설정 (선택적)
        """
        # 설정 초기화
        if isinstance(config, dict):
            self.config = TrainingConfig.from_dict(config)
        else:
            self.config = config
            
        # WandB 설정 병합
        if wandb_config:
            for key, value in wandb_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                    
        # 경로 설정
        self.path_manager = path_manager
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Device 설정
        self._setup_device()
        
        # 컴포넌트 초기화
        self.model = None
        self.tokenizer = None
        self.data_processor = None
        self.trainer = None
        self.metrics_calculator = RougeCalculator()
        
        # 실험 추적
        self.experiment_tracker = ExperimentTracker(
            project_name="nlp-dialogue-summarization",
            experiment_name=self.config.experiment_name,
            config=self.config.__dict__
        )
        
        logger.info(f"Trainer initialized with config: {self.config.model_name}")
        
    def _setup_device(self):
        """디바이스 설정"""
        if self.config.device:
            self.device = torch.device(self.config.device)
        else:
            if platform.system() == "Darwin" and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available() and not self.config.no_cuda:
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        
        logger.info(f"Using device: {self.device}")
        
        # GPU 메모리 정보 로깅
        if self.device.type == "cuda":
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU Memory: {gpu_memory:.1f}GB")
    
    def load_model_and_tokenizer(self):
        """모델과 토크나이저 로드"""
        model_registry = ModelRegistry()
        
        # Unsloth 사용 여부 확인
        if self.config.use_unsloth and UNSLOTH_AVAILABLE:
            logger.info("Loading model with Unsloth...")
            self._load_unsloth_model()
        else:
            # 일반 모델 로드
            model_info = model_registry.get_model_info(self.config.model_name)
            if not model_info:
                raise ValueError(f"Unknown model: {self.config.model_name}")
            
            logger.info(f"Loading model: {self.config.model_name}")
            
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
            
            # 모델 타입에 따라 로드
            if model_info.get("type") == "causal_lm":
                model_class = AutoModelForCausalLM
                self.config.model_type = "causal_lm"
            else:
                model_class = AutoModelForSeq2SeqLM
                self.config.model_type = "seq2seq"
            
            # 모델 로드
            self.model = model_class.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.config.fp16 else torch.float32
            )
            
            # 패딩 토큰 설정
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                if self.model.config.pad_token_id is None:
                    self.model.config.pad_token_id = self.tokenizer.eos_token_id
            
            # QLoRA 적용
            if self.config.use_qlora:
                self._apply_qlora()
                
        # 디바이스로 이동
        if not self.config.use_unsloth:
            self.model = self.model.to(self.device)
            
        # 모델 정보 로깅
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
    def _load_unsloth_model(self):
        """Unsloth를 사용한 모델 로드"""
        if not UNSLOTH_AVAILABLE:
            raise ImportError("Unsloth is not available. Please install it first.")
            
        # Unsloth 모델 로드
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_input_length,
            dtype=torch.float16 if self.config.fp16 else None,
            load_in_4bit=self.config.use_qlora,
        )
        
        # LoRA 어댑터 추가
        if self.config.use_qlora:
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=self.config.lora_r,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"],
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                use_gradient_checkpointing=self.config.gradient_checkpointing,
                random_state=self.config.seed,
            )
    
    def _apply_qlora(self):
        """QLoRA 설정 적용"""
        if not self.config.use_qlora:
            return
            
        logger.info("Applying QLoRA configuration...")
        
        # LoRA 설정
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM if self.config.model_type == "seq2seq" else TaskType.CAUSAL_LM,
            target_modules=["q_proj", "v_proj"]  # 모델에 따라 조정 필요
        )
        
        # PEFT 모델 생성
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
    def load_data(self, train_path: Optional[str] = None, 
                  valid_path: Optional[str] = None,
                  test_path: Optional[str] = None):
        """데이터 로드 및 전처리"""
        # 경로 설정
        train_path = train_path or self.path_manager.get_data_path("train.csv")
        valid_path = valid_path or self.path_manager.get_data_path("dev.csv")
        test_path = test_path or self.path_manager.get_data_path("test.csv")
        
        # 데이터 프로세서 초기화
        self.data_processor = DataProcessor(
            tokenizer=self.tokenizer,
            max_input_length=self.config.max_input_length,
            max_target_length=self.config.max_target_length,
            model_type=self.config.model_type
        )
        
        # 데이터 로드
        logger.info("Loading datasets...")
        self.train_dataset = self.data_processor.load_and_process_data(train_path, is_train=True)
        self.valid_dataset = self.data_processor.load_and_process_data(valid_path, is_train=False)
        
        if test_path and Path(test_path).exists():
            self.test_dataset = self.data_processor.load_and_process_data(test_path, is_train=False)
        else:
            self.test_dataset = None
            
        logger.info(f"Train samples: {len(self.train_dataset)}")
        logger.info(f"Valid samples: {len(self.valid_dataset)}")
        if self.test_dataset:
            logger.info(f"Test samples: {len(self.test_dataset)}")
    
    def compute_metrics(self, eval_pred):
        """평가 메트릭 계산"""
        predictions, labels = eval_pred
        
        # 토큰 ID를 텍스트로 디코딩
        if isinstance(predictions, tuple):
            predictions = predictions[0]
            
        # Numpy array 처리
        if len(predictions.shape) == 3:
            predictions = np.argmax(predictions, axis=-1)
            
        # 디코딩
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # 레이블 처리 (-100은 무시)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # 후처리
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        
        # ROUGE 점수 계산
        rouge_scores = self.metrics_calculator.compute_rouge(decoded_preds, decoded_labels)
        
        return {
            "rouge_1": rouge_scores["rouge1"],
            "rouge_2": rouge_scores["rouge2"],
            "rouge_l": rouge_scores["rougeL"]
        }
    
    def setup_trainer(self):
        """Trainer 설정"""
        # 학습 인자 설정
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=50,
            evaluation_strategy=self.config.evaluation_strategy,
            save_strategy=self.config.save_strategy,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            predict_with_generate=True,
            generation_max_length=self.config.max_target_length,
            generation_num_beams=self.config.num_beams,
            fp16=self.config.fp16 and self.device.type == "cuda",
            push_to_hub=self.config.push_to_hub,
            gradient_checkpointing=self.config.gradient_checkpointing,
            optim=self.config.optim,
            learning_rate=self.config.learning_rate,
            max_grad_norm=self.config.max_grad_norm,
            seed=self.config.seed,
            remove_unused_columns=True,
            label_names=["labels"],
            report_to=["wandb"] if WANDB_AVAILABLE and wandb.run is not None else [],
            run_name=self.config.run_name,
        )
        
        # MPS 디바이스 설정
        if self.device.type == "mps":
            training_args.use_mps_device = True
            training_args.fp16 = False  # MPS는 fp16 미지원
            
        # 데이터 콜레이터
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            padding=True,
            max_length=self.config.max_input_length
        )
        
        # 콜백 설정
        callbacks = [
            EarlyStoppingCallback(early_stopping_patience=3),
            CustomMetricsCallback(self)
        ]
        
        # Trainer 초기화
        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=self.train_dataset,
            eval_dataset=self.valid_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=callbacks,
        )
        
        logger.info("Trainer setup completed")
        
    def train(self):
        """모델 학습"""
        logger.info("Starting training...")
        
        # 학습 시작 시간 기록
        import time
        start_time = time.time()
        
        # 학습 실행
        train_result = self.trainer.train()
        
        # 학습 시간 계산
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time/60:.2f} minutes")
        
        # 최종 모델 저장
        self.trainer.save_model()
        
        # 학습 메트릭 저장
        metrics = train_result.metrics
        metrics["training_time"] = training_time
        
        with open(self.output_dir / "train_results.json", "w") as f:
            json.dump(metrics, f, indent=2)
            
        # 실험 추적
        self.experiment_tracker.log_metrics(metrics)
        
        return metrics
    
    def evaluate(self, dataset=None):
        """모델 평가"""
        if dataset is None:
            dataset = self.valid_dataset
            
        logger.info("Evaluating model...")
        
        # 평가 실행
        eval_result = self.trainer.evaluate(eval_dataset=dataset)
        
        # 결과 저장
        with open(self.output_dir / "eval_results.json", "w") as f:
            json.dump(eval_result, f, indent=2)
            
        # 주요 메트릭 출력
        logger.info("Evaluation Results:")
        logger.info(f"  - Loss: {eval_result.get('eval_loss', 0):.4f}")
        logger.info(f"  - ROUGE-1: {eval_result.get('eval_rouge_1', 0)*100:.2f}%")
        logger.info(f"  - ROUGE-2: {eval_result.get('eval_rouge_2', 0)*100:.2f}%")
        logger.info(f"  - ROUGE-L: {eval_result.get('eval_rouge_l', 0)*100:.2f}%")
        
        return eval_result
    
    def predict(self, dataset=None, save_predictions=True):
        """테스트 데이터 예측"""
        if dataset is None:
            dataset = self.test_dataset or self.valid_dataset
            
        logger.info("Generating predictions...")
        
        # 예측 실행
        predictions = self.trainer.predict(
            test_dataset=dataset,
            max_length=self.config.max_target_length,
            num_beams=self.config.num_beams
        )
        
        # 디코딩
        decoded_preds = self.tokenizer.batch_decode(
            predictions.predictions,
            skip_special_tokens=True
        )
        
        # 저장
        if save_predictions:
            output_file = self.output_dir / "predictions.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                for pred in decoded_preds:
                    f.write(pred.strip() + "\n")
            logger.info(f"Predictions saved to {output_file}")
            
        return decoded_preds
    
    def save_model(self, path: Optional[str] = None):
        """모델 저장"""
        save_path = path or str(self.output_dir / "final_model")
        
        logger.info(f"Saving model to {save_path}")
        
        # Unsloth 모델 저장
        if self.config.use_unsloth and UNSLOTH_AVAILABLE:
            self.model.save_pretrained_merged(save_path, self.tokenizer)
        else:
            # 일반 모델 저장
            self.trainer.save_model(save_path)
            self.tokenizer.save_pretrained(save_path)
            
        # 설정 저장
        config_path = Path(save_path) / "training_config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.__dict__, f, indent=2)
            
        logger.info("Model saved successfully")
        
    def load_model(self, path: str):
        """저장된 모델 로드"""
        logger.info(f"Loading model from {path}")
        
        # 설정 로드
        config_path = Path(path) / "training_config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config_dict = json.load(f)
                self.config = TrainingConfig.from_dict(config_dict)
        
        # 모델과 토크나이저 로드
        model_class = AutoModelForSeq2SeqLM if self.config.model_type == "seq2seq" else AutoModelForCausalLM
        
        self.model = model_class.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        
        # 디바이스로 이동
        self.model = self.model.to(self.device)
        
        logger.info("Model loaded successfully")
        
    def run_sweep_experiment(self):
        """WandB Sweep 실험 실행"""
        # 모든 구성 요소 초기화
        self.load_model_and_tokenizer()
        self.load_data()
        self.setup_trainer()
        
        # 학습
        train_metrics = self.train()
        
        # 평가
        eval_metrics = self.evaluate()
        
        # 최종 메트릭 반환
        final_metrics = {
            **train_metrics,
            **eval_metrics,
            "model_name": self.config.model_name,
            "total_params": sum(p.numel() for p in self.model.parameters()),
            "trainable_params": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
        
        return final_metrics
        
    def cleanup(self):
        """리소스 정리"""
        # GPU 메모리 정리
        if self.model is not None:
            del self.model
        if self.trainer is not None:
            del self.trainer
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # WandB 종료
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.finish()
            
        logger.info("Cleanup completed")


def main():
    """독립 실행을 위한 메인 함수"""
    # 설정 로드
    config_path = Path(__file__).parent.parent / "config.yaml"
    config_dict = load_config(str(config_path))
    
    # 기본 실험 설정
    experiment_config = {
        "model_name": "gogamza/kobart-base-v2",
        "num_train_epochs": 3,
        "per_device_train_batch_size": 16,
        "learning_rate": 2e-5,
        "experiment_name": "baseline_experiment",
        "output_dir": "./outputs/baseline"
    }
    
    # 설정 병합
    experiment_config.update(config_dict.get("training", {}))
    
    # 트레이너 초기화
    trainer = NMTTrainer(experiment_config)
    
    try:
        # 실험 실행
        trainer.load_model_and_tokenizer()
        trainer.load_data()
        trainer.setup_trainer()
        
        # 학습
        train_metrics = trainer.train()
        logger.info(f"Training metrics: {train_metrics}")
        
        # 평가
        eval_metrics = trainer.evaluate()
        logger.info(f"Evaluation metrics: {eval_metrics}")
        
        # 예측 (선택적)
        if trainer.test_dataset:
            predictions = trainer.predict()
            logger.info(f"Generated {len(predictions)} predictions")
            
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        raise
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
