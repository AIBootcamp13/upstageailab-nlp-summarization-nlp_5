"""
설정 파일 관리 및 로딩 유틸리티

6조 방식의 YAML 설정 관리를 NLP 프로젝트에 맞게 개선한 ConfigManager 클래스.
기존 config.yaml과 새로운 확장 설정 구조를 모두 지원하며, 
WandB Sweep 파라미터 동적 병합 및 설정 검증 기능을 제공합니다.
"""

import os
import yaml
import copy
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import logging
from datetime import datetime
import json
from collections.abc import MutableMapping
from utils.path_utils import PathManager, path_manager


class ConfigValidationError(Exception):
    """설정 검증 오류"""
    pass


class ConfigManager:
    """
    설정 파일 통합 관리자
    
    6조 방식의 YAML 설정 관리를 NLP 프로젝트에 맞게 개선한 고급 설정 관리자입니다.
    단순한 설정 로딩을 넘어서 실험 계절막의 이기종 설정 관리를 제공합니다.
    
    Features:
        - 기존 config.yaml과 새로운 base_config.yaml 모두 지원
        - YAML 파일 간 상속 및 딴 병합 (Deep Merge)
        - 환경변수 기반 동적 오버라이드
        - WandB Sweep 파라미터 동적 병합
        - 포괄적 설정 검증 및 스키마 지원
        - 모델별 특화 설정 및 설정 마이그레이션
        - 오류 복구 및 기본값 자동 처리
        
    Example:
        >>> config_manager = ConfigManager()
        >>> config = config_manager.load_config('base_config.yaml', model_config='bart_base')
        >>> config = config_manager.merge_sweep_params(wandb_params)
    """
    
    def __init__(self, base_dir: Optional[str] = None, validate: bool = True):
        """
        ConfigManager 초기화
        
        Args:
            base_dir: 설정 파일 기본 디렉토리 (기본값: 프로젝트 루트)
            validate: 설정 검증 활성화 여부
        """
        # 경로 관리자를 통해 프로젝트 루트 경로 사용
        self.base_dir = path_manager.project_root if base_dir is None else Path(base_dir)
        self.validate = validate
        self.config = None
        self.is_legacy = False
        self.loaded_files = []
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        
        # 설정 스키마 (검증용)
        self._schema = self._load_schema()
        
        # 환경변수 매핑
        self._env_mapping = {
            'WANDB_PROJECT': 'wandb.project',
            'WANDB_ENTITY': 'wandb.entity', 
            'MODEL_NAME': 'general.model_name',
            'OUTPUT_DIR': 'general.output_dir',
            'BATCH_SIZE': 'training.per_device_train_batch_size',
            'LEARNING_RATE': 'training.learning_rate',
            'NUM_EPOCHS': 'training.num_train_epochs'
        }
    
    def load_config(self, config_path: Union[str, Path], 
                   model_config: Optional[str] = None,
                   sweep_config: Optional[str] = None) -> Dict[str, Any]:
        """
        설정 파일 로딩 (기존/신규 형식 자동 감지)
        
        Args:
            config_path: 메인 설정 파일 경로
            model_config: 모델별 특화 설정 파일명 (예: 'kobart')
            sweep_config: Sweep 설정 파일명 (예: 'hyperparameter_sweep')
            
        Returns:
            병합된 설정 딕셔너리
        """
        config_path = Path(config_path)
        self.loaded_files.append(str(config_path))
        
        # 메인 설정 파일 로딩
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        self.logger.info(f"Loaded main config from: {config_path}")
        
        # 기존 config.yaml 형식 감지 및 변환
        if self._is_legacy_format(config):
            self.logger.info("Legacy config format detected, migrating...")
            config = self._migrate_legacy_config(config)
            self.is_legacy = True
        
        # 모델별 설정 병합
        if model_config:
            config = self._merge_model_config(config, model_config)
        
        # Sweep 설정 병합
        if sweep_config:
            config = self._merge_sweep_config(config, sweep_config)
        
        # 환경변수 오버라이드 적용
        config = self._apply_env_overrides(config)
        
        # 설정 검증
        if self.validate:
            self._validate_config(config)
        
        # 기본값 처리
        config = self._apply_defaults(config)
        
        self.config = config
        self.logger.info("Configuration loaded and validated successfully")
        
        return config
    
    def merge_sweep_params(self, sweep_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        WandB Sweep 파라미터를 동적으로 병합 (6조 방식)
        
        Args:
            sweep_params: WandB config에서 받은 파라미터들
            
        Returns:
            Sweep 파라미터가 적용된 설정
        """
        if not self.config:
            raise ConfigValidationError("Base config must be loaded first")
        
        config = copy.deepcopy(self.config)
        
        # WandB 파라미터를 설정 구조에 맞게 매핑
        param_mapping = {
            # 학습 관련
            'learning_rate': 'training.learning_rate',
            'per_device_train_batch_size': 'training.per_device_train_batch_size',
            'per_device_eval_batch_size': 'training.per_device_eval_batch_size',
            'num_train_epochs': 'training.num_train_epochs',
            'warmup_ratio': 'training.warmup_ratio',
            'weight_decay': 'training.weight_decay',
            'lr_scheduler_type': 'training.lr_scheduler_type',
            'gradient_accumulation_steps': 'training.gradient_accumulation_steps',
            'label_smoothing': 'training.label_smoothing',
            'early_stopping_patience': 'training.early_stopping_patience',
            'optim': 'training.optim',
            
            # 토크나이저 관련
            'encoder_max_len': 'tokenizer.encoder_max_len',
            'decoder_max_len': 'tokenizer.decoder_max_len',
            
            # 생성 관련 (NLP 특화)
            'num_beams': 'generation.num_beams',
            'length_penalty': 'generation.length_penalty',
            'no_repeat_ngram_size': 'generation.no_repeat_ngram_size',
            'generation_max_length': 'generation.max_length',
            
            # 모델 관련
            'model_architecture': 'model.architecture',
            'model_checkpoint': 'model.checkpoint'
        }
        
        # 파라미터 적용
        for param_name, param_value in sweep_params.items():
            if param_name in param_mapping:
                config_path = param_mapping[param_name]
                self._set_nested_value(config, config_path, param_value)
                self.logger.debug(f"Applied sweep param: {param_name} = {param_value}")
        
        # 조건부 제약 적용 (메모리 최적화)
        config = self._apply_constraints(config)
        
        return config
    
    def save_config(self, config: Dict[str, Any], save_path: Union[str, Path]) -> None:
        """
        설정을 YAML 파일로 저장
        
        Args:
            config: 저장할 설정 딕셔너리
            save_path: 저장 경로
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 메타데이터 추가
        config_to_save = copy.deepcopy(config)
        if 'meta' not in config_to_save:
            config_to_save['meta'] = {}
        
        config_to_save['meta'].update({
            'saved_at': datetime.now().isoformat(),
            'loaded_files': self.loaded_files,
            'is_legacy_migrated': self.is_legacy
        })
        
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_to_save, f, default_flow_style=False, 
                     allow_unicode=True, indent=2)
        
        self.logger.info(f"Configuration saved to: {save_path}")
    
    def get_model_configs(self) -> List[str]:
        """
        사용 가능한 모델 설정 파일 목록 반환
        
        Returns:
            모델 설정 파일명 리스트 (확장자 제외)
        """
        models_dir = self.base_dir / "config" / "models"
        if not models_dir.exists():
            return []
        
        return [f.stem for f in models_dir.glob("*.yaml")]
    
    def get_sweep_configs(self) -> List[str]:
        """
        사용 가능한 Sweep 설정 파일 목록 반환
        
        Returns:
            Sweep 설정 파일명 리스트 (확장자 제외)
        """
        sweep_dir = self.base_dir / "config" / "sweep"
        if not sweep_dir.exists():
            return []
        
        return [f.stem for f in sweep_dir.glob("*.yaml")]
    
    def validate_config_file(self, config_path: Union[str, Path]) -> bool:
        """
        설정 파일 유효성 검사
        
        Args:
            config_path: 검사할 설정 파일 경로
            
        Returns:
            유효하면 True, 그렇지 않으면 False
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if self._is_legacy_format(config):
                config = self._migrate_legacy_config(config)
            
            self._validate_config(config)
            return True
        
        except Exception as e:
            self.logger.error(f"Config validation failed: {e}")
            return False
    
    def _is_legacy_format(self, config: Dict[str, Any]) -> bool:
        """
        기존 config.yaml 형식인지 판단
        
        Args:
            config: 설정 딕셔너리
            
        Returns:
            기존 형식이면 True
        """
        # 새로운 형식은 meta 섹션이 있음
        if "meta" in config:
            return False
        
        # 기존 형식은 general 섹션이 최상위에 있음
        legacy_indicators = ["general", "tokenizer", "training", "wandb", "inference"]
        return any(key in config for key in legacy_indicators)
    
    def _migrate_legacy_config(self, legacy_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        기존 config.yaml을 새로운 형식으로 변환
        
        Args:
            legacy_config: 기존 형식 설정
            
        Returns:
            새로운 형식으로 변환된 설정
        """
        migrated = {
            'meta': {
                'experiment_name': 'migrated_experiment',
                'version': '1.0',
                'description': 'Migrated from legacy config.yaml',
                'migrated_at': datetime.now().isoformat()
            }
        }
        
        # 기존 섹션들을 새로운 구조에 맞게 매핑
        section_mapping = {
            'general': 'general',
            'tokenizer': 'tokenizer', 
            'training': 'training',
            'wandb': 'wandb',
            'inference': 'inference'
        }
        
        for old_key, new_key in section_mapping.items():
            if old_key in legacy_config:
                migrated[new_key] = legacy_config[old_key]
        
        # 새로운 섹션들 추가 (기본값)
        if 'model' not in migrated:
            migrated['model'] = {
                'architecture': 'kobart',
                'checkpoint': legacy_config.get('general', {}).get('model_name', 'digit82/kobart-summarization'),
                'load_pretrained': True
            }
        
        if 'generation' not in migrated:
            # training에서 생성 관련 설정 추출
            training = legacy_config.get('training', {})
            migrated['generation'] = {
                'max_length': training.get('generation_max_length', 100),
                'num_beams': legacy_config.get('inference', {}).get('num_beams', 4),
                'no_repeat_ngram_size': legacy_config.get('inference', {}).get('no_repeat_ngram_size', 2),
                'early_stopping': legacy_config.get('inference', {}).get('early_stopping', True),
                'length_penalty': 1.0,
                'do_sample': False
            }
        
        if 'evaluation' not in migrated:
            migrated['evaluation'] = {
                'metrics': ['rouge1', 'rouge2', 'rougeL'],
                'multi_reference': True,
                'rouge_use_stemmer': True,
                'rouge_tokenize_korean': True
            }
        
        return migrated
    
    def _merge_model_config(self, base_config: Dict[str, Any], 
                          model_name: str) -> Dict[str, Any]:
        """
        모델별 특화 설정 병합
        
        Args:
            base_config: 기본 설정
            model_name: 모델명 (kobart, kogpt2, t5, solar_api)
            
        Returns:
            모델 설정이 병합된 설정
        """
        model_config_path = self.base_dir / "config" / "models" / f"{model_name}.yaml"
        
        if not model_config_path.exists():
            self.logger.warning(f"Model config not found: {model_config_path}")
            return base_config
        
        with open(model_config_path, 'r', encoding='utf-8') as f:
            model_config = yaml.safe_load(f)
        
        self.loaded_files.append(str(model_config_path))
        
        # 모델 설정을 기본 설정에 병합 (딥 머지)
        merged_config = self._deep_merge(base_config, model_config)
        
        self.logger.info(f"Merged model config: {model_name}")
        return merged_config
    
    def _merge_sweep_config(self, base_config: Dict[str, Any], 
                          sweep_name: str) -> Dict[str, Any]:
        """
        Sweep 설정 정보 추가 (실제 병합은 merge_sweep_params에서)
        
        Args:
            base_config: 기본 설정
            sweep_name: Sweep 설정명
            
        Returns:
            Sweep 정보가 추가된 설정
        """
        sweep_config_path = self.base_dir / "config" / "sweep" / f"{sweep_name}.yaml"
        
        if not sweep_config_path.exists():
            self.logger.warning(f"Sweep config not found: {sweep_config_path}")
            return base_config
        
        with open(sweep_config_path, 'r', encoding='utf-8') as f:
            sweep_config = yaml.safe_load(f)
        
        self.loaded_files.append(str(sweep_config_path))
        
        # Sweep 정보를 메타데이터에 추가
        base_config['meta']['sweep_config'] = sweep_config
        
        self.logger.info(f"Added sweep config info: {sweep_name}")
        return base_config
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        환경변수 기반 설정 오버라이드 적용
        
        Args:
            config: 기본 설정
            
        Returns:
            환경변수가 적용된 설정
        """
        for env_var, config_path in self._env_mapping.items():
            env_value = os.getenv(env_var)
            if env_value:
                # 타입 변환 시도
                try:
                    # 숫자 변환
                    if env_value.replace('.', '').replace('-', '').isdigit():
                        env_value = float(env_value) if '.' in env_value else int(env_value)
                    # 불린 변환
                    elif env_value.lower() in ['true', 'false']:
                        env_value = env_value.lower() == 'true'
                except:
                    pass  # 문자열 그대로 사용
                
                self._set_nested_value(config, config_path, env_value)
                self.logger.info(f"Applied env override: {env_var} -> {config_path} = {env_value}")
        
        return config
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        설정 검증
        
        Args:
            config: 검증할 설정 딕셔너리
            
        Raises:
            ConfigValidationError: 검증 실패 시
        """
        errors = []
        
        # 필수 섹션 검증
        required_sections = ['general', 'model', 'tokenizer', 'training']
        for section in required_sections:
            if section not in config:
                errors.append(f"Missing required section: {section}")
        
        # 타입 검증
        if 'training' in config:
            training = config['training']
            
            # 학습률 범위 검증
            lr = training.get('learning_rate')
            if lr and (lr <= 0 or lr > 1):
                errors.append(f"Invalid learning_rate: {lr} (must be 0 < lr <= 1)")
            
            # 배치 크기 검증
            batch_size = training.get('per_device_train_batch_size')
            if batch_size and (not isinstance(batch_size, int) or batch_size <= 0):
                errors.append(f"Invalid batch_size: {batch_size} (must be positive integer)")
            
            # 에폭 수 검증
            epochs = training.get('num_train_epochs')
            if epochs and (not isinstance(epochs, int) or epochs <= 0):
                errors.append(f"Invalid num_train_epochs: {epochs} (must be positive integer)")
        
        # 토크나이저 길이 검증
        if 'tokenizer' in config:
            tokenizer = config['tokenizer']
            
            encoder_len = tokenizer.get('encoder_max_len')
            if encoder_len and (not isinstance(encoder_len, int) or encoder_len <= 0):
                errors.append(f"Invalid encoder_max_len: {encoder_len}")
            
            decoder_len = tokenizer.get('decoder_max_len')
            if decoder_len and (not isinstance(decoder_len, int) or decoder_len <= 0):
                errors.append(f"Invalid decoder_max_len: {decoder_len}")
        
        # 생성 파라미터 검증
        if 'generation' in config:
            generation = config['generation']
            
            num_beams = generation.get('num_beams')
            if num_beams and (not isinstance(num_beams, int) or num_beams <= 0):
                errors.append(f"Invalid num_beams: {num_beams}")
            
            length_penalty = generation.get('length_penalty')
            if length_penalty and (not isinstance(length_penalty, (int, float)) or length_penalty <= 0):
                errors.append(f"Invalid length_penalty: {length_penalty}")
        
        if errors:
            raise ConfigValidationError(f"Configuration validation failed: {'; '.join(errors)}")
    
    def _apply_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        기본값 적용
        
        Args:
            config: 설정 딕셔너리
            
        Returns:
            기본값이 적용된 설정
        """
        # 기본값 정의
        defaults = {
            'general.seed': 42,
            'general.device': 'auto',
            'general.num_workers': 4,
            'training.fp16': True,
            'training.dataloader_num_workers': 4,
            'training.remove_unused_columns': False,
            'generation.early_stopping': True,
            'generation.do_sample': False,
            'evaluation.prediction_loss_only': False,
            'wandb.log_model': 'end',
            'wandb.save_code': True
        }
        
        for path, default_value in defaults.items():
            if self._get_nested_value(config, path) is None:
                self._set_nested_value(config, path, default_value)
        
        return config
    
    def _apply_constraints(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        조건부 제약 적용 (메모리 최적화)
        
        Args:
            config: 설정 딕셔너리
            
        Returns:
            제약이 적용된 설정
        """
        encoder_len = config.get('tokenizer', {}).get('encoder_max_len', 512)
        
        # 긴 시퀀스 사용 시 배치 크기 제한
        if encoder_len > 512:
            current_batch = config.get('training', {}).get('per_device_train_batch_size', 32)
            if current_batch > 16:
                config['training']['per_device_train_batch_size'] = 16
                self.logger.warning(f"Reduced batch size to 16 due to long sequence length: {encoder_len}")
        
        if encoder_len > 1024:
            current_batch = config.get('training', {}).get('per_device_train_batch_size', 16)
            if current_batch > 8:
                config['training']['per_device_train_batch_size'] = 8
                config['training']['gradient_accumulation_steps'] = max(
                    config.get('training', {}).get('gradient_accumulation_steps', 1), 2
                )
                self.logger.warning(f"Reduced batch size to 8 and increased accumulation steps due to very long sequence: {encoder_len}")
        
        return config
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        딕셔너리 딥 머지
        
        Args:
            base: 기본 딕셔너리
            override: 오버라이드할 딕셔너리
            
        Returns:
            병합된 딕셔너리
        """
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        
        return result
    
    def _get_nested_value(self, config: Dict[str, Any], path: str) -> Any:
        """
        중첩된 딕셔너리에서 값 조회
        
        Args:
            config: 설정 딕셔너리
            path: 점으로 구분된 경로 (예: 'training.learning_rate')
            
        Returns:
            조회된 값 또는 None
        """
        keys = path.split('.')
        current = config
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current
    
    def _set_nested_value(self, config: Dict[str, Any], path: str, value: Any) -> None:
        """
        중첩된 딕셔너리에 값 설정
        
        Args:
            config: 설정 딕셔너리
            path: 점으로 구분된 경로
            value: 설정할 값
        """
        keys = path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _load_schema(self) -> Dict[str, Any]:
        """
        설정 스키마 로딩 (향후 확장용)
        
        Returns:
            설정 검증 스키마
        """
        # 향후 JSON Schema 등을 활용한 상세 검증 구현 예정
        return {}


# 6조 방식의 get_yaml 함수를 NLP용으로 개선
def get_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """
    YAML 파일 로딩 (6조 방식 호환)
    
    Args:
        path: YAML 파일 경로
        
    Returns:
        로딩된 YAML 데이터
    """
    with open(path, "r", encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_config(config_path: Union[str, Path], 
               model_config: Optional[str] = None,
               validate: bool = True) -> Dict[str, Any]:
    """
    간편한 설정 로딩 함수 (기존 코드 호환용)
    
    Args:
        config_path: 메인 설정 파일 경로
        model_config: 모델 설정명 (선택사항)
        validate: 검증 활성화 여부
        
    Returns:
        로딩된 설정
    """
    manager = ConfigManager(validate=validate)
    return manager.load_config(config_path, model_config=model_config)
