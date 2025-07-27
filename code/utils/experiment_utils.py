"""
실험 관리 유틸리티

NLP 대화 요약 프로젝트를 위한 실험 추적, 모델 등록, 결과 분석 기능을 제공합니다.
WandB와 연동하여 체계적인 실험 관리를 지원합니다.
"""

import os
import json
import pickle
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
import logging
import pandas as pd
import numpy as np
from collections import defaultdict


@dataclass
class ExperimentInfo:
    """실험 정보 클래스"""
    experiment_id: str
    name: str
    description: str
    config: Dict[str, Any]
    model_type: str
    dataset_info: Dict[str, Any]
    start_time: str
    end_time: Optional[str] = None
    status: str = "실행중"  # 실행중, 완료, 실패
    best_metrics: Optional[Dict[str, float]] = None
    final_metrics: Optional[Dict[str, float]] = None
    model_path: Optional[str] = None
    wandb_run_id: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class ModelInfo:
    """모델 정보 클래스"""
    model_id: str
    name: str
    architecture: str
    checkpoint: str
    config: Dict[str, Any]
    performance: Dict[str, float]
    training_info: Dict[str, Any]
    file_path: str
    created_at: str
    experiment_id: Optional[str] = None
    tags: Optional[List[str]] = None


class ExperimentTracker:
    """
    실험 추적기
    
    실험 정보 저장, 로딩, 비교 등의 기능을 제공합니다.
    """
    
    def __init__(self, experiments_dir: Union[str, Path] = "./experiments"):
        """
        ExperimentTracker 초기화
        
        Args:
            experiments_dir: 실험 저장 디렉토리
        """
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.current_experiment = None
        
        # 실험 데이터베이스 (JSON 파일)
        self.db_path = self.experiments_dir / "experiments.json"
        self.experiments_db = self._load_experiments_db()
    
    def start_experiment(self, name: str, description: str, 
                        config: Dict[str, Any], model_type: str,
                        dataset_info: Optional[Dict[str, Any]] = None,
                        wandb_run_id: Optional[str] = None) -> str:
        """
        새로운 실험 시작
        
        Args:
            name: 실험명
            description: 실험 설명
            config: 실험 설정
            model_type: 모델 타입
            dataset_info: 데이터셋 정보
            wandb_run_id: WandB 실행 ID
            
        Returns:
            실험 ID
        """
        # 실험 ID 생성 (타임스탬프 + 설정 해시)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_hash = self._hash_config(config)
        experiment_id = f"{timestamp}_{config_hash[:8]}"
        
        # 실험 정보 생성
        experiment_info = ExperimentInfo(
            experiment_id=experiment_id,
            name=name,
            description=description,
            config=config,
            model_type=model_type,
            dataset_info=dataset_info or {},
            start_time=datetime.now().isoformat(),
            wandb_run_id=wandb_run_id
        )
        
        # 실험 디렉토리 생성
        exp_dir = self.experiments_dir / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 실험 정보 저장
        self._save_experiment_info(experiment_info)
        
        # 현재 실험으로 설정
        self.current_experiment = experiment_info
        
        self.logger.info(f"Started experiment: {experiment_id} - {name}")
        return experiment_id
    
    def complete_experiment(self, experiment_id: Optional[str] = None,
                          final_metrics: Optional[Dict[str, float]] = None,
                          model_path: Optional[str] = None,
                          notes: Optional[str] = None):
        """
        실험 완료 처리
        
        Args:
            experiment_id: 실험 ID
            final_metrics: 최종 메트릭
            model_path: 모델 저장 경로
            notes: 추가 노트
        """
        self.update_experiment(
            experiment_id=experiment_id,
            status="completed",
            end_time=datetime.now().isoformat(),
            final_metrics=final_metrics,
            model_path=model_path,
            notes=notes
        )
        
        exp_id = experiment_id or self.current_experiment.experiment_id
        self.logger.info(f"Completed experiment: {exp_id}")
    
    def update_experiment(self, experiment_id: Optional[str] = None, **kwargs):
        """실험 정보 업데이트"""
        if experiment_id is None:
            if self.current_experiment is None:
                raise ValueError("No current experiment and no experiment_id provided")
            experiment_info = self.current_experiment
        else:
            experiment_info = self._load_experiment_info(experiment_id)
        
        # 필드 업데이트
        for field, value in kwargs.items():
            if hasattr(experiment_info, field):
                setattr(experiment_info, field, value)
        
        # 저장
        self._save_experiment_info(experiment_info)
        
        if experiment_id is None or experiment_id == self.current_experiment.experiment_id:
            self.current_experiment = experiment_info
    
    def get_experiment_list(self, status: Optional[str] = None) -> List[ExperimentInfo]:
        """실험 리스트 조회"""
        experiments = []
        
        for exp_id in self.experiments_db:
            exp_info = self._load_experiment_info(exp_id)
            if status is None or exp_info.status == status:
                experiments.append(exp_info)
        
        # 시작 시간으로 정렬 (최신순)
        experiments.sort(key=lambda x: x.start_time, reverse=True)
        return experiments
    
    def get_best_experiments(self, metric: str = "rouge_combined_f1", 
                           top_k: int = 5) -> List[ExperimentInfo]:
        """최고 성능 실험들 조회"""
        experiments = self.get_experiment_list(status="completed")
        
        # 메트릭 기준으로 정렬
        experiments_with_metric = []
        for exp in experiments:
            if exp.best_metrics and metric in exp.best_metrics:
                experiments_with_metric.append((exp, exp.best_metrics[metric]))
        
        # 점수 기준 내림차순 정렬
        experiments_with_metric.sort(key=lambda x: x[1], reverse=True)
        
        return [exp for exp, _ in experiments_with_metric[:top_k]]
    
    def _load_experiments_db(self) -> Dict[str, Any]:
        """실험 데이터베이스 로딩"""
        if self.db_path.exists():
            with open(self.db_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_experiments_db(self):
        """실험 데이터베이스 저장"""
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(self.experiments_db, f, ensure_ascii=False, indent=2)
    
    def _save_experiment_info(self, experiment_info: ExperimentInfo):
        """실험 정보 저장"""
        exp_dir = self.experiments_dir / experiment_info.experiment_id
        info_file = exp_dir / "experiment_info.json"
        
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(experiment_info), f, ensure_ascii=False, indent=2)
        
        # 데이터베이스 업데이트
        self.experiments_db[experiment_info.experiment_id] = {
            'name': experiment_info.name,
            'status': experiment_info.status,
            'start_time': experiment_info.start_time,
            'model_type': experiment_info.model_type
        }
        self._save_experiments_db()
    
    def _load_experiment_info(self, experiment_id: str) -> ExperimentInfo:
        """실험 정보 로딩"""
        info_file = self.experiments_dir / experiment_id / "experiment_info.json"
        
        if not info_file.exists():
            raise FileNotFoundError(f"Experiment not found: {experiment_id}")
        
        with open(info_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return ExperimentInfo(**data)
    
    def _hash_config(self, config: Dict[str, Any]) -> str:
        """설정 해시 생성"""
        config_str = json.dumps(config, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(config_str.encode()).hexdigest()


class ModelRegistry:
    """모델 등록 및 관리 시스템"""
    
    def __init__(self, models_dir: Union[str, Path] = "./models"):
        """ModelRegistry 초기화"""
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # 모델 데이터베이스
        self.db_path = self.models_dir / "models.json"
        self.models_db = self._load_models_db()
    
    def register_model(self, name: str, architecture: str, checkpoint: str,
                      config: Dict[str, Any], performance: Dict[str, float],
                      training_info: Dict[str, Any], file_path: str,
                      experiment_id: Optional[str] = None,
                      tags: Optional[List[str]] = None) -> str:
        """모델 등록"""
        # 모델 ID 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_hash = hashlib.md5(name.encode()).hexdigest()[:8]
        model_id = f"{architecture}_{timestamp}_{name_hash}"
        
        # 모델 정보 생성
        model_info = ModelInfo(
            model_id=model_id,
            name=name,
            architecture=architecture,
            checkpoint=checkpoint,
            config=config,
            performance=performance,
            training_info=training_info,
            file_path=file_path,
            created_at=datetime.now().isoformat(),
            experiment_id=experiment_id,
            tags=tags or []
        )
        
        # 모델 정보 저장
        self._save_model_info(model_info)
        
        self.logger.info(f"Registered model: {model_id} - {name}")
        return model_id
    
    def _load_models_db(self) -> Dict[str, Any]:
        """모델 데이터베이스 로딩"""
        if self.db_path.exists():
            with open(self.db_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_models_db(self):
        """모델 데이터베이스 저장"""
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(self.models_db, f, ensure_ascii=False, indent=2)
    
    def _save_model_info(self, model_info: ModelInfo):
        """모델 정보 저장"""
        info_file = self.models_dir / f"{model_info.model_id}.json"
        
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(model_info), f, ensure_ascii=False, indent=2)
        
        # 데이터베이스 업데이트
        self.models_db[model_info.model_id] = {
            'name': model_info.name,
            'architecture': model_info.architecture,
            'created_at': model_info.created_at,
            'main_metric': model_info.performance.get('rouge_combined_f1', 0.0)
        }
        self._save_models_db()
    
    def _load_model_info(self, model_id: str) -> ModelInfo:
        """모델 정보 로딩"""
        info_file = self.models_dir / f"{model_id}.json"
        
        if not info_file.exists():
            raise FileNotFoundError(f"Model not found: {model_id}")
        
        with open(info_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return ModelInfo(**data)


# 편의 함수들
def create_experiment_tracker(experiments_dir: str = "./experiments") -> ExperimentTracker:
    """실험 추적기 생성 편의 함수"""
    return ExperimentTracker(experiments_dir)


def create_model_registry(models_dir: str = "./models") -> ModelRegistry:
    """모델 등록기 생성 편의 함수"""
    return ModelRegistry(models_dir)
