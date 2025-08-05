"""
WandB 통합 유틸리티

실험별로 고유한 WandB run 설정 및 모델 레지스트리 연동
견고한 WandB 관리 시스템으로 네트워크 연결 실패 시에도 실험 연속성 보장

통합 에러 처리 시스템 적용:
- 네트워크 에러: 자동 재시도 및 오프라인 모드 전환
- 구조화된 로깅으로 WandB 연결 상태 추적
- 에러 발생 시에도 실험 진행 중단 없이 로컬 로깅으로 대체
"""

import os
import wandb
from pathlib import Path
import logging
import time
import socket
from datetime import datetime
from typing import Dict, Any, Optional

# 통합 에러 처리 시스템 import
from .error_handling import (
    handle_error, log_structured, log_performance_metric,
    safe_execute, get_logging_manager
)

logger = logging.getLogger(__name__)


class RobustWandbManager:
    """
    견고한 WandB 관리 시스템
    
    네트워크 연결 실패 시에도 실험이 중단되지 않도록 하는 Fail-Safe WandB 관리자.
    연결 테스트, 재시도 로직, 오프라인 모드 자동 전환을 통해 실험 연속성을 보장합니다.
    """
    
    def __init__(self, fallback_mode: bool = True, max_retries: int = 3):
        """
        Args:
            fallback_mode: 연결 실패 시 오프라인 모드로 전환할지 여부
            max_retries: 최대 재시도 횟수
        """
        self.fallback_mode = fallback_mode
        self.max_retries = max_retries
        self.connection_status = "unknown"
        self.offline_mode = False
        
    def safe_setup_wandb_for_experiment(self, 
                                       config: Dict[str, Any], 
                                       experiment_name: str,
                                       sweep_mode: bool = False) -> Dict[str, Any]:
        """
        안전한 WandB 실험 설정 (기존 setup_wandb_for_experiment 래핑)
        
        통합 에러 처리 시스템을 사용하여 네트워크 에러를 자동으로 처리하고
        오프라인 모드로 안전하게 대체합니다.
        
        Args:
            config: 실험 설정
            experiment_name: 실험명
            sweep_mode: Sweep 모드 여부
            
        Returns:
            WandB 설정 정보 (오프라인 모드 정보 포함)
        """
        # 통합 에러 처리로 전체 프로세스 래핑
        return safe_execute(
            func=self._execute_wandb_setup,
            config=config,
            experiment_name=experiment_name,
            sweep_mode=sweep_mode,
            error_category="wandb_errors",
            default_return=self._create_offline_fallback_result(config, experiment_name)
        )
    
    def _execute_wandb_setup(self, 
                            config: Dict[str, Any], 
                            experiment_name: str,
                            sweep_mode: bool = False) -> Dict[str, Any]:
        """
        실제 WandB 설정 실행 (내부 메서드)
        """
        log_structured(
            level="INFO",
            message=f"WandB 실험 설정 시작: {experiment_name}",
            component="wandb_utils",
            function="safe_setup_wandb_for_experiment",
            metadata={"experiment_name": experiment_name, "sweep_mode": sweep_mode}
        )
        
        start_time = time.time()
        
        try:
            # 1단계: 네트워크 연결 사전 테스트
            if not self._test_wandb_connectivity():
                logger.warning("WandB 연결 테스트 실패 - 오프라인 모드로 전환")
                return self._handle_offline_mode(config, experiment_name)
            
            # 2단계: 기존 함수로 WandB 설정 생성
            wandb_config = setup_wandb_for_experiment(config, experiment_name, sweep_mode)
            
            # 3단계: 안전한 WandB 초기화 시도
            result = self._safe_wandb_init(wandb_config, experiment_name)
            
            # 성공 시 성능 메트릭 로깅
            log_performance_metric(
                metric_name="wandb_setup_duration",
                value=time.time() - start_time,
                unit="seconds",
                component="wandb_utils"
            )
            
            return result
            
        except Exception as e:
            # 기존 예외 처리를 통합 에러 처리로 대체
            error_context = handle_error(
                error=e,
                component="wandb_utils",
                function="safe_setup_wandb_for_experiment",
                error_category="wandb_errors",
                context={"experiment_name": experiment_name, "sweep_mode": sweep_mode}
            )
            
            # 에러 전략에 따른 처리
            if error_context.strategy.value == "fallback":
                logger.warning(f"WandB 설정 실패, 오프라인 모드로 대체: {e}")
                return self._handle_offline_mode(config, experiment_name)
            else:
                # 기타 전략의 경우 에러 재발생
                raise e
    
    def _create_offline_fallback_result(self, config: Dict[str, Any], experiment_name: str) -> Dict[str, Any]:
        """
        오프라인 모드 폴백 결과 생성
        
        Args:
            config: 실험 설정
            experiment_name: 실험명
            
        Returns:
            오프라인 모드 설정 정보
        """
        log_structured(
            level="WARNING",
            message=f"WandB 오프라인 모드 활성화: {experiment_name}",
            component="wandb_utils",
            function="_create_offline_fallback_result",
            metadata={"experiment_name": experiment_name, "offline_mode": True}
        )
        
        self.offline_mode = True
        
        return {
            'status': 'offline',
            'offline_mode': True,
            'experiment_name': experiment_name,
            'message': 'WandB 연결 실패로 오프라인 모드로 전환',
            'fallback_logging': {
                'local_log_dir': './logs/wandb_offline',
                'experiment_id': f'offline_{int(time.time())}_{experiment_name}'
            }
        }
    
    def _test_wandb_connectivity(self) -> bool:
        """
        WandB 서비스 연결 테스트
        
        Returns:
            연결 가능 여부
        """
        try:
            # DNS 해결 테스트
            socket.gethostbyname('api.wandb.ai')
            
            # 포트 연결 테스트 (HTTPS)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)  # 5초 타임아웃
            result = sock.connect_ex(('api.wandb.ai', 443))
            sock.close()
            
            if result == 0:
                self.connection_status = "connected"
                logger.debug("WandB 연결 테스트 성공")
                return True
            else:
                logger.debug(f"WandB 포트 연결 실패: {result}")
                return False
                
        except socket.gaierror as e:
            logger.debug(f"WandB DNS 해결 실패: {e}")
            return False
        except Exception as e:
            logger.debug(f"WandB 연결 테스트 예외: {e}")
            return False
    
    def _safe_wandb_init(self, wandb_config: Dict[str, Any], experiment_name: str) -> Dict[str, Any]:
        """
        지수 백오프 재시도 로직으로 안전한 WandB 초기화
        
        Args:
            wandb_config: WandB 설정
            experiment_name: 실험명
            
        Returns:
            초기화 결과
        """
        for attempt in range(self.max_retries):
            try:
                # WandB 초기화 시도
                run = wandb.init(**wandb_config)
                
                if run is not None:
                    self.connection_status = "initialized"
                    logger.info(f"WandB 초기화 성공: {run.name}")
                    
                    return {
                        'status': 'success',
                        'run_id': run.id,
                        'run_name': run.name,
                        'run_url': run.get_url(),
                        'offline_mode': False
                    }
                
            except Exception as e:
                wait_time = (2 ** attempt) + 1  # 지수 백오프: 1, 3, 5초
                logger.warning(f"WandB 초기화 시도 {attempt + 1}/{self.max_retries} 실패: {e}")
                
                if attempt < self.max_retries - 1:
                    logger.info(f"{wait_time}초 후 재시도...")
                    time.sleep(wait_time)
                else:
                    # 최대 재시도 초과 시 오프라인 모드
                    if self.fallback_mode:
                        logger.warning("모든 재시도 실패, 오프라인 모드로 전환")
                        return self._handle_offline_mode(wandb_config, experiment_name)
                    else:
                        raise e
        
        return self._handle_offline_mode(wandb_config, experiment_name)
    
    def _handle_offline_mode(self, config: Dict[str, Any], experiment_name: str) -> Dict[str, Any]:
        """
        오프라인 모드 처리
        
        Args:
            config: 원본 설정
            experiment_name: 실험명
            
        Returns:
            오프라인 모드 설정 정보
        """
        self.offline_mode = True
        self.connection_status = "offline"
        
        # 오프라인 WandB 초기화
        try:
            offline_run = wandb.init(
                mode="offline",
                project=config.get('project', 'offline_project'),
                name=f"offline_{experiment_name}_{int(time.time())}",
                config=config.get('config', {}),
                dir=config.get('dir', './wandb_offline')
            )
            
            return {
                'status': 'offline',
                'offline_mode': True,
                'run_id': offline_run.id if offline_run else 'offline_run',
                'run_name': offline_run.name if offline_run else f'offline_{experiment_name}',
                'local_dir': config.get('dir', './wandb_offline')
            }
            
        except Exception as e:
            logger.error(f"오프라인 WandB 초기화도 실패: {e}")
            return self._create_offline_fallback_result(config, experiment_name)
    
    def get_connection_status(self) -> Dict[str, Any]:
        """연결 상태 정보 반환"""
        return {
            'status': self.connection_status,
            'offline_mode': self.offline_mode,
            'fallback_enabled': self.fallback_mode
        }


# 전역 RobustWandbManager 인스턴스
_robust_wandb_manager = RobustWandbManager()


def safe_setup_wandb_for_experiment(config: Dict[str, Any], 
                                   experiment_name: str,
                                   sweep_mode: bool = False) -> Dict[str, Any]:
    """
    견고한 WandB 실험 설정 (전역 함수)
    
    기존 setup_wandb_for_experiment()의 안전한 버전.
    네트워크 연결 실패 시 오프라인 모드로 자동 전환하여 실험 연속성을 보장합니다.
    
    Args:
        config: 실험 설정
        experiment_name: 실험명
        sweep_mode: Sweep 모드 여부
        
    Returns:
        WandB 설정 정보 (오프라인 모드 정보 포함)
    """
    return _robust_wandb_manager.safe_setup_wandb_for_experiment(config, experiment_name, sweep_mode)


def setup_wandb_for_experiment(config: Dict[str, Any], 
                             experiment_name: str,
                             sweep_mode: bool = False) -> Dict[str, Any]:
    """
    실험별 WandB 설정 초기화 (원본 함수 - 조장님 패턴 유지)
    
    Args:
        config: 실험 설정
        experiment_name: 실험명
        sweep_mode: Sweep 모드 여부
        
    Returns:
        WandB 설정 정보
    """
    # 한국 시간 기반 타임스탬프
    from utils.experiment_utils import get_korean_time_format
    korean_time = get_korean_time_format('MMDDHHMM')
    
    # WandB 설정 가져오기
    wandb_config = config.get('wandb', {})
    
    # 실험별 고유한 run name 생성
    # 조장님 패턴: b_automodel_{current_time}
    model_name = config.get('model', {}).get('architecture', 'unknown')
    model_short = model_name[:1] if model_name != 'unknown' else 'x'  # 첫 글자만
    run_name = f"{model_short}_{experiment_name}_{korean_time}"
    
    # Job type 설정
    job_type = "sweep" if sweep_mode else "experiment"
    
    # Tags 설정
    tags = wandb_config.get('tags', []).copy()
    tags.extend([
        model_name,
        experiment_name,
        f"date_{datetime.now().strftime('%Y%m%d')}",
        f"time_{korean_time}"
    ])
    
    # Notes 생성
    notes = f"""
실험명: {experiment_name}
모델: {config.get('model', {}).get('checkpoint', 'N/A')}
시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (KST)
학습 에포크: {config.get('training', {}).get('num_train_epochs', 'N/A')}
배치 크기: {config.get('training', {}).get('per_device_train_batch_size', 'N/A')}
학습률: {config.get('training', {}).get('learning_rate', 'N/A')}
"""
    
    # WandB 초기화 설정
    wandb_init_config = {
        'project': wandb_config.get('project', 'nlp-5'),
        'entity': wandb_config.get('entity', 'lyjune37-juneictlab'),
        'name': run_name,
        'notes': notes,
        'tags': tags,
        'group': wandb_config.get('group', experiment_name),
        'job_type': job_type,
        'config': config,
        'save_code': wandb_config.get('save_code', True),
    }
    
    # 환경 변수 설정
    if wandb_config.get('log_model', 'end') == 'end':
        os.environ["WANDB_LOG_MODEL"] = "end"
    
    return wandb_init_config


def log_model_to_wandb(model_path: str, 
                      model_name: str,
                      metrics: Dict[str, float],
                      config: Dict[str, Any],
                      aliases: Optional[list] = None):
    """
    학습된 모델을 WandB Model Registry에 등록 (원본 함수)
    
    Args:
        model_path: 모델 저장 경로
        model_name: 모델명
        metrics: 성능 메트릭
        config: 모델 설정
        aliases: 모델 별칭 (예: ["best", "latest"])
    """
    if wandb.run is None:
        logger.warning("WandB run이 활성화되지 않아 모델을 등록할 수 없습니다.")
        return
    
    # 모델 크기 확인
    model_size_mb = get_directory_size_mb(model_path)
    size_threshold = config.get('wandb', {}).get('log_model_size_threshold', 2000)  # 기본 2GB
    
    # 모델 크기가 임계값을 초과하면 로컬만 저장
    if model_size_mb > size_threshold:
        logger.warning(f"모델 크기({model_size_mb:.1f}MB)가 임계값({size_threshold}MB)을 초과하여 WandB에 업로드하지 않습니다.")
        logger.info(f"모델은 로컬에만 저장되었습니다: {model_path}")
        # WandB에 메타데이터만 기록
        wandb.run.summary.update({
            "model_saved_locally": True,
            "model_size_mb": model_size_mb,
            "model_path": model_path,
            **metrics
        })
        return
    
    try:
        # 모델 아티팩트 생성
        from utils.experiment_utils import get_korean_time_format
        model_artifact = wandb.Artifact(
            name=model_name,
            type="model",
            description=f"Dialogue summarization model: {model_name}",
            metadata={
                "rouge1_f1": metrics.get('rouge1_f1', 0),
                "rouge2_f1": metrics.get('rouge2_f1', 0),
                "rougeL_f1": metrics.get('rougeL_f1', 0),
                "rouge_combined_f1": metrics.get('rouge_combined_f1', 0),
                "architecture": config.get('model', {}).get('architecture', 'unknown'),
                "base_model": config.get('model', {}).get('checkpoint', 'unknown'),
                "training_epochs": config.get('training', {}).get('num_train_epochs', 0),
                "korean_time": get_korean_time_format('MMDDHHMM'),
                "model_size_mb": model_size_mb
            }
        )
        
        # 모델 파일 추가
        model_artifact.add_dir(model_path)
        
        # 별칭 설정
        if aliases is None:
            aliases = ["latest"]
            # 최고 성능 모델인 경우 best 태그 추가
            if metrics.get('rouge_combined_f1', 0) > 0.3:  # 임계값 조정 가능
                aliases.append("best")
        
        # WandB에 로그
        wandb.run.log_artifact(model_artifact, aliases=aliases)
        
        logger.info(f"모델이 WandB에 등록되었습니다: {model_name} ({model_size_mb:.1f}MB)")
        
    except Exception as e:
        logger.error(f"WandB 모델 등록 실패: {e}")


def get_directory_size_mb(directory_path: str) -> float:
    """
    디렉토리의 전체 크기를 MB 단위로 계산
    
    Args:
        directory_path: 크기를 계산할 디렉토리 경로
        
    Returns:
        디렉토리 크기 (MB)
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    
    return total_size / (1024 * 1024)  # bytes to MB


def finish_wandb_run(summary_metrics: Dict[str, Any] = None):
    """
    WandB run 종료 및 요약 정보 저장
    
    Args:
        summary_metrics: 최종 요약 메트릭
    """
    if wandb.run is not None:
        # 요약 메트릭 업데이트
        if summary_metrics:
            wandb.run.summary.update(summary_metrics)
        
        # run 종료
        wandb.finish()
        logger.info("WandB run이 종료되었습니다.")


def get_wandb_run_url() -> Optional[str]:
    """현재 WandB run의 URL 반환"""
    if wandb.run is not None:
        return wandb.run.get_url()
    return None
