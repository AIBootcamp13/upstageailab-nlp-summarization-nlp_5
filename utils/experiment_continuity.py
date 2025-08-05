"""
실험 연속성 보장 체크포인트 시스템

컨테이너 재시작이나 예상치 못한 중단 상황에서 실험을 자동으로 복구할 수 있는
체크포인트 및 상태 추적 시스템입니다. 실험 단계별 메타데이터 저장과 재시작 가능성을 판단합니다.
"""

import os
import json
import time
import hashlib
import logging
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
import torch
import psutil

logger = logging.getLogger(__name__)


@dataclass
class ExperimentCheckpoint:
    """실험 체크포인트 정보"""
    experiment_id: str
    experiment_name: str
    stage: str  # 'init', 'model_loaded', 'training_started', 'epoch_completed', 'completed', 'failed'
    timestamp: str
    config_hash: str
    system_info: Dict[str, Any]
    progress_info: Dict[str, Any]
    model_info: Dict[str, Any]
    training_metrics: Dict[str, Any]
    checksum: str
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentCheckpoint':
        """딕셔너리에서 생성"""
        return cls(**data)


@dataclass
class SystemSnapshot:
    """시스템 상태 스냅샷"""
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    gpu_memory_used: Optional[float]
    gpu_utilization: Optional[float]
    container_id: Optional[str]
    python_version: str
    torch_version: str
    cuda_version: Optional[str]
    hostname: str
    timestamp: str


class ExperimentContinuityManager:
    """
    실험 연속성 보장 관리자
    
    컨테이너 재시작이나 예상치 못한 중단 상황에서 실험을 자동으로 복구할 수 있는
    체크포인트 및 상태 추적 시스템을 제공합니다.
    """
    
    def __init__(self, 
                 checkpoint_dir: str = "./checkpoints/continuity",
                 auto_save_interval: int = 300,  # 5분
                 max_checkpoints: int = 10):
        """
        Args:
            checkpoint_dir: 체크포인트 저장 디렉토리
            auto_save_interval: 자동 저장 간격 (초)
            max_checkpoints: 최대 보관할 체크포인트 수
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.auto_save_interval = auto_save_interval
        self.max_checkpoints = max_checkpoints
        
        self.current_experiment: Optional[ExperimentCheckpoint] = None
        self.auto_save_thread: Optional[threading.Thread] = None
        self.auto_save_running = False
        
        logger.info(f"🗂️ 실험 연속성 관리자 초기화: {self.checkpoint_dir}")
    
    def start_experiment(self, 
                        experiment_id: str,
                        experiment_name: str, 
                        config: Dict[str, Any]) -> ExperimentCheckpoint:
        """
        새로운 실험 시작 및 초기 체크포인트 생성
        
        Args:
            experiment_id: 실험 고유 ID
            experiment_name: 실험명
            config: 실험 설정
            
        Returns:
            초기 체크포인트
        """
        logger.info(f"🚀 실험 연속성 추적 시작: {experiment_name} (ID: {experiment_id})")
        
        # 설정 해시 생성 (재시작 시 설정 변경 감지용)
        config_hash = self._calculate_config_hash(config)
        
        # 시스템 정보 수집
        system_info = self._collect_system_info()
        
        # 초기 체크포인트 생성
        checkpoint = ExperimentCheckpoint(
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            stage='init',
            timestamp=datetime.now().isoformat(),
            config_hash=config_hash,
            system_info=system_info,
            progress_info={
                'total_epochs': config.get('training', {}).get('num_train_epochs', 0),
                'completed_epochs': 0,
                'current_step': 0,
                'total_steps': 0
            },
            model_info={
                'architecture': config.get('model', {}).get('architecture', 'unknown'),
                'checkpoint': config.get('model', {}).get('checkpoint', 'unknown'),
                'model_loaded': False
            },
            training_metrics={},
            checksum=''
        )
        
        # 체크섬 계산
        checkpoint.checksum = self._calculate_checkpoint_checksum(checkpoint)
        
        # 체크포인트 저장
        self._save_checkpoint(checkpoint)
        
        self.current_experiment = checkpoint
        
        # 자동 저장 스레드 시작
        self._start_auto_save()
        
        return checkpoint
    
    def update_experiment_stage(self, 
                               stage: str,
                               progress_info: Optional[Dict[str, Any]] = None,
                               model_info: Optional[Dict[str, Any]] = None,
                               training_metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        실험 단계 업데이트
        
        Args:
            stage: 현재 단계
            progress_info: 진행 정보
            model_info: 모델 정보
            training_metrics: 학습 메트릭
        """
        if not self.current_experiment:
            logger.warning("진행 중인 실험이 없어 단계 업데이트를 건너뜁니다")
            return
        
        logger.debug(f"📝 실험 단계 업데이트: {stage}")
        
        # 현재 체크포인트 복사 및 업데이트
        updated_checkpoint = ExperimentCheckpoint(
            experiment_id=self.current_experiment.experiment_id,
            experiment_name=self.current_experiment.experiment_name,
            stage=stage,
            timestamp=datetime.now().isoformat(),
            config_hash=self.current_experiment.config_hash,
            system_info=self._collect_system_info(),
            progress_info={**self.current_experiment.progress_info, **(progress_info or {})},
            model_info={**self.current_experiment.model_info, **(model_info or {})},
            training_metrics={**self.current_experiment.training_metrics, **(training_metrics or {})},
            checksum=''
        )
        
        # 체크섬 재계산
        updated_checkpoint.checksum = self._calculate_checkpoint_checksum(updated_checkpoint)
        
        # 체크포인트 저장
        self._save_checkpoint(updated_checkpoint)
        
        self.current_experiment = updated_checkpoint
    
    def save_experiment_checkpoint(self, 
                                  stage: str,
                                  **kwargs) -> bool:
        """
        실험 체크포인트 저장 (편의 함수)
        
        Args:
            stage: 현재 단계
            **kwargs: 추가 정보
            
        Returns:
            저장 성공 여부
        """
        try:
            self.update_experiment_stage(stage, **kwargs)
            return True
        except Exception as e:
            logger.error(f"체크포인트 저장 실패: {e}")
            return False
    
    def can_resume_experiment(self, experiment_id: str) -> Tuple[bool, Optional[ExperimentCheckpoint]]:
        """
        실험 재시작 가능성 판단
        
        Args:
            experiment_id: 실험 ID
            
        Returns:
            (재시작 가능 여부, 최신 체크포인트)
        """
        try:
            # 최신 체크포인트 찾기
            latest_checkpoint = self._find_latest_checkpoint(experiment_id)
            
            if not latest_checkpoint:
                logger.info(f"실험 {experiment_id}의 체크포인트를 찾을 수 없습니다")
                return False, None
            
            # 체크포인트 무결성 검증
            if not self._verify_checkpoint_integrity(latest_checkpoint):
                logger.error(f"체크포인트 무결성 검증 실패: {experiment_id}")
                return False, None
            
            # 재시작 가능한 단계인지 확인
            resumable_stages = ['model_loaded', 'training_started', 'epoch_completed']
            
            if latest_checkpoint.stage in resumable_stages:
                logger.info(f"✅ 실험 {experiment_id} 재시작 가능 (단계: {latest_checkpoint.stage})")
                return True, latest_checkpoint
            elif latest_checkpoint.stage == 'completed':
                logger.info(f"실험 {experiment_id}는 이미 완료되었습니다")
                return False, latest_checkpoint
            elif latest_checkpoint.stage == 'failed':
                logger.warning(f"실험 {experiment_id}는 실패 상태입니다")
                return False, latest_checkpoint
            else:
                logger.info(f"실험 {experiment_id}는 재시작하기에는 너무 초기 단계입니다 (단계: {latest_checkpoint.stage})")
                return False, latest_checkpoint
            
        except Exception as e:
            logger.error(f"재시작 가능성 판단 중 오류: {e}")
            return False, None
    
    def resume_experiment(self, experiment_id: str) -> Optional[ExperimentCheckpoint]:
        """
        실험 자동 복구
        
        Args:
            experiment_id: 실험 ID
            
        Returns:
            복구된 체크포인트 또는 None
        """
        can_resume, checkpoint = self.can_resume_experiment(experiment_id)
        
        if not can_resume or not checkpoint:
            logger.error(f"실험 {experiment_id} 복구 불가능")
            return None
        
        logger.info(f"🔄 실험 {experiment_id} 자동 복구 시작")
        logger.info(f"   복구 지점: {checkpoint.stage}")
        logger.info(f"   마지막 저장: {checkpoint.timestamp}")
        
        # 현재 실험으로 설정
        self.current_experiment = checkpoint
        
        # 자동 저장 스레드 재시작
        self._start_auto_save()
        
        # 복구 정보 로깅
        if checkpoint.progress_info.get('completed_epochs', 0) > 0:
            logger.info(f"   완료된 에포크: {checkpoint.progress_info['completed_epochs']}/{checkpoint.progress_info['total_epochs']}")
        
        if checkpoint.training_metrics:
            latest_metrics = list(checkpoint.training_metrics.keys())[-1] if checkpoint.training_metrics else 'None'
            logger.info(f"   최신 메트릭: {latest_metrics}")
        
        logger.info(f"✅ 실험 {experiment_id} 복구 완료")
        
        return checkpoint
    
    def finish_experiment(self, 
                         success: bool = True,
                         final_metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        실험 종료 처리
        
        Args:
            success: 성공 여부
            final_metrics: 최종 메트릭
        """
        if not self.current_experiment:
            logger.warning("진행 중인 실험이 없어 종료 처리를 건너뜁니다")
            return
        
        stage = 'completed' if success else 'failed'
        
        logger.info(f"🏁 실험 종료 처리: {self.current_experiment.experiment_name} ({'성공' if success else '실패'})")
        
        self.update_experiment_stage(
            stage=stage,
            training_metrics=final_metrics or {}
        )
        
        # 자동 저장 중지
        self._stop_auto_save()
        
        # 오래된 체크포인트 정리
        self._cleanup_old_checkpoints(self.current_experiment.experiment_id)
        
        self.current_experiment = None
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """
        저장된 실험 목록 반환
        
        Returns:
            실험 정보 리스트
        """
        experiments = {}
        
        try:
            for checkpoint_file in self.checkpoint_dir.glob("*.json"):
                try:
                    with open(checkpoint_file, 'r', encoding='utf-8') as f:
                        checkpoint_data = json.load(f)
                    
                    exp_id = checkpoint_data['experiment_id']
                    
                    # 최신 체크포인트만 유지
                    if exp_id not in experiments or checkpoint_data['timestamp'] > experiments[exp_id]['timestamp']:
                        experiments[exp_id] = {
                            'experiment_id': exp_id,
                            'experiment_name': checkpoint_data['experiment_name'],
                            'stage': checkpoint_data['stage'],
                            'timestamp': checkpoint_data['timestamp'],
                            'progress': checkpoint_data.get('progress_info', {}),
                            'model': checkpoint_data.get('model_info', {}).get('architecture', 'unknown')
                        }
                        
                except Exception as e:
                    logger.debug(f"체크포인트 파일 읽기 실패: {checkpoint_file} - {e}")
                    continue
            
            return list(experiments.values())
            
        except Exception as e:
            logger.error(f"실험 목록 조회 실패: {e}")
            return []
    
    def _calculate_config_hash(self, config: Dict[str, Any]) -> str:
        """설정 해시 계산"""
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _calculate_checkpoint_checksum(self, checkpoint: ExperimentCheckpoint) -> str:
        """체크포인트 체크섬 계산"""
        # checksum 필드를 제외한 나머지 데이터로 체크섬 계산
        checkpoint_dict = checkpoint.to_dict()
        checkpoint_dict.pop('checksum', None)
        
        checkpoint_str = json.dumps(checkpoint_dict, sort_keys=True, default=str)
        return hashlib.sha256(checkpoint_str.encode()).hexdigest()
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """시스템 정보 수집"""
        try:
            # 기본 시스템 정보
            system_info = {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage_percent': psutil.disk_usage('.').percent,
                'hostname': os.uname().nodename if hasattr(os, 'uname') else 'unknown',
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'torch_version': torch.__version__,
                'timestamp': datetime.now().isoformat()
            }
            
            # GPU 정보 (가능한 경우)
            if torch.cuda.is_available():
                try:
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    gpu_memory_used = torch.cuda.memory_allocated(0) / (1024**3)
                    system_info.update({
                        'cuda_version': torch.version.cuda,
                        'gpu_memory_total_gb': gpu_memory,
                        'gpu_memory_used_gb': gpu_memory_used,
                        'gpu_memory_percent': (gpu_memory_used / gpu_memory) * 100
                    })
                except:
                    pass
            
            # 컨테이너 정보 (가능한 경우)
            try:
                if os.path.exists('/.dockerenv'):
                    with open('/proc/self/cgroup', 'r') as f:
                        cgroup_content = f.read()
                        if 'docker' in cgroup_content:
                            # Docker 컨테이너 ID 추출 시도
                            for line in cgroup_content.split('\n'):
                                if 'docker' in line and '/' in line:
                                    container_id = line.split('/')[-1][:12]  # 처음 12자리만
                                    system_info['container_id'] = container_id
                                    break
            except:
                pass
            
            return system_info
            
        except Exception as e:
            logger.debug(f"시스템 정보 수집 중 오류: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _save_checkpoint(self, checkpoint: ExperimentCheckpoint) -> bool:
        """체크포인트 저장"""
        try:
            # 파일명: experiment_id_timestamp.json
            timestamp_str = checkpoint.timestamp.replace(':', '-').replace('.', '-')
            filename = f"{checkpoint.experiment_id}_{timestamp_str}.json"
            filepath = self.checkpoint_dir / filename
            
            # JSON 저장
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(checkpoint.to_dict(), f, indent=2, ensure_ascii=False, default=str)
            
            logger.debug(f"체크포인트 저장됨: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"체크포인트 저장 실패: {e}")
            return False
    
    def _find_latest_checkpoint(self, experiment_id: str) -> Optional[ExperimentCheckpoint]:
        """최신 체크포인트 찾기"""
        try:
            latest_checkpoint = None
            latest_timestamp = None
            
            pattern = f"{experiment_id}_*.json"
            for checkpoint_file in self.checkpoint_dir.glob(pattern):
                try:
                    with open(checkpoint_file, 'r', encoding='utf-8') as f:
                        checkpoint_data = json.load(f)
                    
                    timestamp = checkpoint_data['timestamp']
                    
                    if latest_timestamp is None or timestamp > latest_timestamp:
                        latest_timestamp = timestamp
                        latest_checkpoint = ExperimentCheckpoint.from_dict(checkpoint_data)
                        
                except Exception as e:
                    logger.debug(f"체크포인트 파일 읽기 실패: {checkpoint_file} - {e}")
                    continue
            
            return latest_checkpoint
            
        except Exception as e:
            logger.error(f"최신 체크포인트 찾기 실패: {e}")
            return None
    
    def _verify_checkpoint_integrity(self, checkpoint: ExperimentCheckpoint) -> bool:
        """체크포인트 무결성 검증"""
        try:
            # 체크섬 재계산 및 비교
            calculated_checksum = self._calculate_checkpoint_checksum(checkpoint)
            
            if calculated_checksum != checkpoint.checksum:
                logger.error(f"체크포인트 체크섬 불일치: 예상 {checkpoint.checksum}, 실제 {calculated_checksum}")
                return False
            
            # 필수 필드 확인
            required_fields = ['experiment_id', 'experiment_name', 'stage', 'timestamp']
            for field in required_fields:
                if not getattr(checkpoint, field, None):
                    logger.error(f"체크포인트 필수 필드 누락: {field}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"체크포인트 무결성 검증 실패: {e}")
            return False
    
    def _start_auto_save(self) -> None:
        """자동 저장 스레드 시작"""
        if self.auto_save_thread and self.auto_save_thread.is_alive():
            return
        
        self.auto_save_running = True
        self.auto_save_thread = threading.Thread(target=self._auto_save_worker, daemon=True)
        self.auto_save_thread.start()
        
        logger.debug(f"자동 저장 스레드 시작 (간격: {self.auto_save_interval}초)")
    
    def _stop_auto_save(self) -> None:
        """자동 저장 스레드 중지"""
        self.auto_save_running = False
        
        if self.auto_save_thread and self.auto_save_thread.is_alive():
            self.auto_save_thread.join(timeout=5)
        
        logger.debug("자동 저장 스레드 중지")
    
    def _auto_save_worker(self) -> None:
        """자동 저장 워커"""
        while self.auto_save_running:
            try:
                time.sleep(self.auto_save_interval)
                
                if self.current_experiment and self.auto_save_running:
                    # 현재 상태로 체크포인트 업데이트 (단계는 유지)
                    self.update_experiment_stage(
                        stage=self.current_experiment.stage + '_auto_save',
                    )
                    
            except Exception as e:
                logger.debug(f"자동 저장 중 오류: {e}")
    
    def _cleanup_old_checkpoints(self, experiment_id: str) -> None:
        """오래된 체크포인트 정리"""
        try:
            pattern = f"{experiment_id}_*.json"
            checkpoint_files = list(self.checkpoint_dir.glob(pattern))
            
            # 파일을 타임스탬프 순으로 정렬 (최신순)
            checkpoint_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            
            # 최대 개수를 초과하는 파일들 삭제
            if len(checkpoint_files) > self.max_checkpoints:
                files_to_delete = checkpoint_files[self.max_checkpoints:]
                
                for file_to_delete in files_to_delete:
                    try:
                        file_to_delete.unlink()
                        logger.debug(f"오래된 체크포인트 삭제: {file_to_delete}")
                    except Exception as e:
                        logger.debug(f"체크포인트 삭제 실패: {file_to_delete} - {e}")
            
        except Exception as e:
            logger.debug(f"체크포인트 정리 중 오류: {e}")


# 전역 ExperimentContinuityManager 인스턴스
_experiment_continuity_manager = ExperimentContinuityManager()


def get_continuity_manager() -> ExperimentContinuityManager:
    """전역 연속성 관리자 반환"""
    return _experiment_continuity_manager


def save_experiment_checkpoint(stage: str, **kwargs) -> bool:
    """
    실험 체크포인트 저장 (전역 함수)
    
    Args:
        stage: 현재 단계
        **kwargs: 추가 정보
        
    Returns:
        저장 성공 여부
    """
    return _experiment_continuity_manager.save_experiment_checkpoint(stage, **kwargs)


def can_resume_experiment(experiment_id: str) -> Tuple[bool, Optional[ExperimentCheckpoint]]:
    """
    실험 재시작 가능성 판단 (전역 함수)
    
    Args:
        experiment_id: 실험 ID
        
    Returns:
        (재시작 가능 여부, 최신 체크포인트)
    """
    return _experiment_continuity_manager.can_resume_experiment(experiment_id)


def resume_experiment(experiment_id: str) -> Optional[ExperimentCheckpoint]:
    """
    실험 자동 복구 (전역 함수)
    
    Args:
        experiment_id: 실험 ID
        
    Returns:
        복구된 체크포인트 또는 None
    """
    return _experiment_continuity_manager.resume_experiment(experiment_id)
