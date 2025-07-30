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
import pytz


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


# 한국 시간 기반 유틸리티 함수들
def get_korean_time_format(format_type: str = 'MMDDHHMM') -> str:
    """
    한국 시간 기준 시간 형식 생성
    
    Args:
        format_type: 시간 형식 타입
            - 'MMDDHHMM': 월일시분 (예: 07301455)
            - 'YYYYMMDD_HHMMSS': 년월일_시분초 (기존 형식)
    
    Returns:
        형식화된 시간 문자열
    """
    # 한국 시간 timezone 설정
    kst = pytz.timezone('Asia/Seoul')
    now = datetime.now(kst)
    
    if format_type == 'MMDDHHMM':
        return now.strftime('%m%d%H%M')  # 예: 0730 1455 -> 07301455
    elif format_type == 'YYYYMMDD_HHMMSS':
        return now.strftime('%Y%m%d_%H%M%S')  # 기존 형식 유지
    else:
        return now.strftime('%m%d%H%M')  # 기본값은 MMDDHHMM


def generate_experiment_id_with_korean_time() -> str:
    """
    한국 시간 기반 실험 ID 생성
    
    Returns:
        한국 시간 + 해시를 포함한 실험 ID
    """
    korean_time = get_korean_time_format('MMDDHHMM')
    # 간단한 해시 생성 (시분초 마이크로초 기반)
    import random
    hash_suffix = str(hash(datetime.now().isoformat()))[-4:].replace('-', '0')
    return f"{korean_time}_{hash_suffix}"


def get_wandb_run_name_with_korean_time(model_name: str = None, prefix: str = None) -> str:
    """
    한국 시간 기반 WandB run name 생성
    
    Args:
        model_name: 모델명 (선택사항)
        prefix: 접두사 (선택사항)
    
    Returns:
        WandB run name 문자열
    """
    korean_time = get_korean_time_format('MMDDHHMM')
    
    parts = []
    if prefix:
        parts.append(prefix)
    if model_name:
        parts.append(model_name)
    parts.append(korean_time)
    
    return '_'.join(parts)



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
    
    def end_experiment(self, experiment_id: Optional[str] = None,
                      final_metrics: Optional[Dict[str, float]] = None,
                      model_path: Optional[str] = None,
                      status: str = "completed",
                      notes: Optional[str] = None,
                      best_metrics: Optional[Dict[str, float]] = None):
        """
        실험 종료 (컴플리트 실험의 에일리어스)
        
        Args:
            experiment_id: 실험 ID (없으면 현재 실험 사용)
            final_metrics: 최종 메트릭
            model_path: 모델 저장 경로
            status: 실험 상태
            notes: 추가 노트
            best_metrics: 최고 성능 메트릭
        """
        # update_experiment를 직접 호출하여 status 처리
        self.update_experiment(
            experiment_id=experiment_id,
            status=status,
            end_time=datetime.now().isoformat(),
            final_metrics=final_metrics,
            model_path=model_path,
            notes=notes,
            best_metrics=best_metrics
        )
        
        exp_id = experiment_id or self.current_experiment.experiment_id
        self.logger.info(f"Ended experiment: {exp_id}")
    
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
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        메트릭 로깅
        
        Args:
            metrics: 메트릭 딕셔너리
            step: 단계 번호 (선택사항)
        """
        if not self.current_experiment:
            self.logger.warning("⚠️  현재 실행 중인 실험이 없습니다.")
            return
        
        # 메트릭 로깅 (간단한 정보만)
        if step is not None:
            self.logger.info(f"📊 Step {step} metrics logged")
        else:
            self.logger.info("📊 Metrics logged")
        
        # best_metrics 업데이트 (주요 메트릭만)
        rouge_combined = metrics.get('eval_rouge_combined_f1', 0) or metrics.get('rouge_combined_f1', 0)
        if rouge_combined > 0:
            current_best = self.current_experiment.best_metrics or {}
            if rouge_combined > current_best.get('rouge_combined_f1', 0):
                self.current_experiment.best_metrics = {
                    'rouge_combined_f1': rouge_combined,
                    'rouge1_f1': metrics.get('eval_rouge1_f1', 0) or metrics.get('rouge1_f1', 0),
                    'rouge2_f1': metrics.get('eval_rouge2_f1', 0) or metrics.get('rouge2_f1', 0),
                    'rougeL_f1': metrics.get('eval_rougeL_f1', 0) or metrics.get('rougeL_f1', 0)
                }
                self._save_experiment_info(self.current_experiment)
                self.logger.info(f"🏆 New best ROUGE-F1: {rouge_combined:.4f}")
    
    def get_experiment_list(self, status: Optional[str] = None) -> List[ExperimentInfo]:
        """실험 리스트 조회"""
        experiments = []
        for exp_id in self.experiments_db.keys():
            try:
                exp_info = self._load_experiment_info(exp_id)
                if status is None or exp_info.status == status:
                    experiments.append(exp_info)
            except FileNotFoundError:
                # 실험 정보 파일이 없는 경우 스킵
                continue
        
        # 시작 시간으로 정렬 (최신순)
        experiments.sort(key=lambda x: x.start_time, reverse=True)
        return experiments
    
    def get_best_experiments(self, metric: str = "rouge_combined_f1", 
                           top_k: int = 5) -> List[ExperimentInfo]:
        """최고 성능 실험들 조회"""
        experiments = self.get_experiment_list(status="completed")
        
        # 메트릭 기준으로 정렬
        valid_experiments = []
        for exp in experiments:
            if exp.best_metrics and metric in exp.best_metrics:
                valid_experiments.append(exp)
        
        valid_experiments.sort(
            key=lambda x: x.best_metrics[metric], 
            reverse=True
        )
        
        return valid_experiments[:top_k]
    
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
    
    def log_experiment(self, experiment_name: str, config: Dict[str, Any], 
                      results: Dict[str, Any], **kwargs):
        """
        실험 결과를 로그에 기록
        
        Args:
            experiment_name: 실험명
            config: 실험 설정
            results: 실험 결과
            **kwargs: 추가 파라미터
        """
        try:
            # 기본적으로 로그에 기록
            self.logger.info(f"📊 실험 로그: {experiment_name}")
            
            # 결과가 성공적이면 간단한 메트릭 로그
            if results.get('status') == 'success' and 'metrics' in results:
                metrics = results['metrics']
                self.logger.info(f"   ROUGE-1: {metrics.get('eval_rouge1_f1', 0):.4f}")
                self.logger.info(f"   ROUGE-2: {metrics.get('eval_rouge2_f1', 0):.4f}")
                self.logger.info(f"   ROUGE-L: {metrics.get('eval_rougeL_f1', 0):.4f}")
            
            # 실험이 이미 시작되었다면 업데이트
            if self.current_experiment is not None:
                if results.get('status') == 'success':
                    self.complete_experiment(
                        final_metrics=results.get('metrics', {})
                    )
                elif results.get('status') == 'error':
                    self.update_experiment(status="failed", notes=results.get('error', 'Unknown error'))
            
        except Exception as e:
            self.logger.warning(f"실험 로그 기록 중 오류: {e}")
            # 로그 실패가 전체 실행을 방해하지 않도록 pass
            pass
    
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
            'end_time': experiment_info.end_time,
            'best_metrics': experiment_info.best_metrics
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
    """
    모델 등록 및 관리
    
    학습된 모델들의 정보를 등록하고 관리하는 클래스입니다.
    """
    
    def __init__(self, registry_dir: Union[str, Path] = "./models"):
        """
        ModelRegistry 초기화
        
        Args:
            registry_dir: 모델 등록 정보 저장 디렉토리
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # 모델 데이터베이스 (JSON 파일)
        self.db_path = self.registry_dir / "models.json"
        self.models_db = self._load_models_db()
    
    def register_model(self, name: str, architecture: str, checkpoint: str,
                      config: Dict[str, Any], performance: Dict[str, float],
                      training_info: Dict[str, Any], file_path: str,
                      experiment_id: Optional[str] = None,
                      tags: Optional[List[str]] = None) -> str:
        """
        모델 등록
        
        Args:
            name: 모델명
            architecture: 모델 아키텍처
            checkpoint: 체크포인트 경로
            config: 모델 설정
            performance: 성능 메트릭
            training_info: 학습 정보
            file_path: 모델 파일 경로
            experiment_id: 실험 ID
            tags: 태그 리스트
            
        Returns:
            모델 ID
        """
        # 모델 ID 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = f"{name}_{timestamp}"
        
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
    
    def get_model_list(self, architecture: Optional[str] = None,
                      tag: Optional[str] = None) -> List[ModelInfo]:
        """모델 리스트 조회"""
        models = []
        for model_id in self.models_db.keys():
            try:
                model_info = self._load_model_info(model_id)
                
                # 필터링
                if architecture and model_info.architecture != architecture:
                    continue
                if tag and tag not in model_info.tags:
                    continue
                    
                models.append(model_info)
            except FileNotFoundError:
                continue
        
        # 생성 시간으로 정렬 (최신순)
        models.sort(key=lambda x: x.created_at, reverse=True)
        return models
    
    def get_best_models(self, metric: str = "rouge_combined_f1",
                       top_k: int = 5) -> List[ModelInfo]:
        """최고 성능 모델들 조회"""
        models = self.get_model_list()
        
        # 메트릭 기준으로 정렬
        valid_models = []
        for model in models:
            if metric in model.performance:
                valid_models.append(model)
        
        valid_models.sort(
            key=lambda x: x.performance[metric],
            reverse=True
        )
        
        return valid_models[:top_k]
    
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
        info_file = self.registry_dir / f"{model_info.model_id}.json"
        
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(model_info), f, ensure_ascii=False, indent=2)
        
        # 데이터베이스 업데이트
        self.models_db[model_info.model_id] = {
            'name': model_info.name,
            'architecture': model_info.architecture,
            'performance': model_info.performance,
            'created_at': model_info.created_at,
            'tags': model_info.tags
        }
        self._save_models_db()
    
    def _load_model_info(self, model_id: str) -> ModelInfo:
        """모델 정보 로딩"""
        info_file = self.registry_dir / f"{model_id}.json"
        
        if not info_file.exists():
            raise FileNotFoundError(f"Model not found: {model_id}")
        
        with open(info_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return ModelInfo(**data)
