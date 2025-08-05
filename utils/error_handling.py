"""
통합 에러 처리 및 로깅 시스템

모든 개선 사항을 통합하여 일관된 에러 처리 패턴과 상세한 로깅 시스템을 구축합니다.
문제 발생 시 신속한 진단과 대응이 가능하도록 구조화된 로그 및 알림 시스템을 제공합니다.
"""

import os
import sys
import json
import time
import traceback
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import inspect

# 표준 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """에러 심각도 분류"""
    LOW = "low"          # 경고 수준, 실험 진행에 영향 없음
    MEDIUM = "medium"    # 주의 필요, 성능 저하 가능성
    HIGH = "high"        # 중요, 기능 제한 발생
    CRITICAL = "critical" # 치명적, 실험 중단 위험


class ErrorStrategy(Enum):
    """에러 처리 전략"""
    RETRY = "retry"          # 재시도 가능
    FALLBACK = "fallback"    # 대안 방법 사용  
    CONTINUE = "continue"    # 무시하고 진행
    ABORT = "abort"          # 실험 중단


@dataclass
class ErrorContext:
    """에러 컨텍스트 정보"""
    error_id: str
    timestamp: str
    severity: ErrorSeverity
    strategy: ErrorStrategy
    component: str          # 발생 모듈/클래스
    function: str          # 발생 함수
    error_type: str        # 에러 타입
    error_message: str     # 에러 메시지
    stack_trace: str       # 스택 트레이스
    system_info: Dict[str, Any]
    retry_count: int = 0
    resolved: bool = False
    resolution_method: Optional[str] = None


@dataclass
class LogEntry:
    """구조화된 로그 엔트리"""
    timestamp: str
    level: str
    component: str
    function: str
    message: str
    metadata: Dict[str, Any]
    experiment_id: Optional[str] = None
    session_id: Optional[str] = None


class LoggingManager:
    """
    구조화된 로깅 관리자
    
    JSON 형태의 구조화된 로그를 생성하여 자동 분석과 모니터링을 지원합니다.
    """
    
    def __init__(self, log_dir: str = "./logs", enable_structured_logging: bool = True):
        """
        Args:
            log_dir: 로그 저장 디렉토리
            enable_structured_logging: 구조화된 로깅 활성화 여부
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_structured_logging = enable_structured_logging
        self.session_id = f"session_{int(time.time())}"
        self.experiment_id = None
        
        # 구조화된 로그 파일 핸들러 설정
        if enable_structured_logging:
            self._setup_structured_logging()
    
    def _setup_structured_logging(self):
        """구조화된 로깅 설정"""
        # 구조화된 로그 파일 (JSON Lines 형태)
        structured_log_file = self.log_dir / f"structured_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        # 기존 로거에 JSON 핸들러 추가
        json_handler = logging.FileHandler(structured_log_file)
        json_handler.setFormatter(logging.Formatter('%(message)s'))  # JSON만 저장
        
        # 별도 JSON 로거 생성
        self.json_logger = logging.getLogger(f'{__name__}.structured')
        self.json_logger.addHandler(json_handler)
        self.json_logger.setLevel(logging.DEBUG)
        self.json_logger.propagate = False  # 부모 로거에 전파 방지
    
    def set_experiment_id(self, experiment_id: str):
        """실험 ID 설정"""
        self.experiment_id = experiment_id
    
    def log_structured(self, 
                      level: str,
                      message: str, 
                      component: str = None,
                      function: str = None,
                      metadata: Dict[str, Any] = None) -> None:
        """
        구조화된 로그 생성
        
        Args:
            level: 로그 레벨
            message: 로그 메시지
            component: 컴포넌트명
            function: 함수명
            metadata: 추가 메타데이터
        """
        if not self.enable_structured_logging:
            return
        
        try:
            # 호출자 정보 자동 추출
            if not component or not function:
                frame = inspect.currentframe().f_back
                if frame:
                    component = component or frame.f_globals.get('__name__', 'unknown')
                    function = function or frame.f_code.co_name
            
            # 로그 엔트리 생성
            log_entry = LogEntry(
                timestamp=datetime.now().isoformat(),
                level=level,
                component=component or 'unknown',
                function=function or 'unknown',
                message=message,
                metadata=metadata or {},
                experiment_id=self.experiment_id,
                session_id=self.session_id
            )
            
            # JSON Lines 형태로 저장
            self.json_logger.info(json.dumps(asdict(log_entry), ensure_ascii=False, default=str))
            
        except Exception as e:
            # 로깅 실패가 본래 작업에 영향을 주지 않도록
            logger.debug(f"구조화된 로깅 실패: {e}")
    
    def log_error_context(self, error_context: ErrorContext) -> None:
        """에러 컨텍스트 로깅"""
        self.log_structured(
            level="ERROR",
            message=f"Error occurred: {error_context.error_message}",
            component=error_context.component,
            function=error_context.function,
            metadata={
                'error_id': error_context.error_id,
                'severity': error_context.severity.value,
                'strategy': error_context.strategy.value,
                'error_type': error_context.error_type,
                'retry_count': error_context.retry_count,
                'resolved': error_context.resolved,
                'resolution_method': error_context.resolution_method,
                'stack_trace': error_context.stack_trace,
                'system_info': error_context.system_info
            }
        )
    
    def log_performance_metric(self, 
                              metric_name: str, 
                              value: Union[int, float], 
                              unit: str = None,
                              component: str = None) -> None:
        """성능 메트릭 로깅"""
        self.log_structured(
            level="INFO",
            message=f"Performance metric: {metric_name}",
            component=component,
            metadata={
                'metric_name': metric_name,
                'value': value,
                'unit': unit,
                'metric_type': 'performance'
            }
        )
    
    def log_experiment_event(self, 
                            event_type: str, 
                            event_data: Dict[str, Any],
                            component: str = None) -> None:
        """실험 이벤트 로깅"""
        self.log_structured(
            level="INFO",
            message=f"Experiment event: {event_type}",
            component=component,
            metadata={
                'event_type': event_type,
                'event_data': event_data,
                'metric_type': 'experiment'
            }
        )


class AlertManager:
    """
    심각한 에러 알림 관리자
    
    Critical 및 High 심각도 에러에 대해 알림을 발송합니다.
    """
    
    def __init__(self, enable_alerts: bool = True):
        """
        Args:
            enable_alerts: 알림 활성화 여부
        """
        self.enable_alerts = enable_alerts
        self.alert_history = []
        self.alert_cooldown = {}  # 중복 알림 방지
        self.cooldown_period = 300  # 5분
    
    def send_alert(self, error_context: ErrorContext) -> bool:
        """
        에러 알림 발송
        
        Args:
            error_context: 에러 컨텍스트
            
        Returns:
            알림 발송 성공 여부
        """
        if not self.enable_alerts:
            return False
        
        # Critical 및 High 심각도만 알림
        if error_context.severity not in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]:
            return False
        
        # 중복 알림 방지 (같은 에러 타입에 대해 쿨다운 적용)
        alert_key = f"{error_context.component}:{error_context.error_type}"
        current_time = time.time()
        
        if alert_key in self.alert_cooldown:
            if current_time - self.alert_cooldown[alert_key] < self.cooldown_period:
                logger.debug(f"알림 쿨다운 중: {alert_key}")
                return False
        
        try:
            # 알림 메시지 생성
            alert_message = self._create_alert_message(error_context)
            
            # 알림 발송 (현재는 로그로만 출력, 향후 Slack/Email 연동 가능)
            logger.critical(f"🚨 ALERT: {alert_message}")
            
            # 알림 기록
            self.alert_history.append({
                'timestamp': error_context.timestamp,
                'error_id': error_context.error_id,
                'severity': error_context.severity.value,
                'component': error_context.component,
                'message': alert_message
            })
            
            # 쿨다운 업데이트
            self.alert_cooldown[alert_key] = current_time
            
            return True
            
        except Exception as e:
            logger.error(f"알림 발송 실패: {e}")
            return False
    
    def _create_alert_message(self, error_context: ErrorContext) -> str:
        """알림 메시지 생성"""
        return (
            f"[{error_context.severity.value.upper()}] "
            f"{error_context.component}.{error_context.function}: "
            f"{error_context.error_message} "
            f"(Retry: {error_context.retry_count})"
        )


class ErrorHandler:
    """
    통합 에러 처리 관리자
    
    에러 분류, 처리 전략 결정, 복구 시도를 담당합니다.
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 logging_manager: Optional[LoggingManager] = None,
                 alert_manager: Optional[AlertManager] = None):
        """
        Args:
            config_path: 에러 처리 설정 파일 경로
            logging_manager: 로깅 관리자
            alert_manager: 알림 관리자
        """
        self.logging_manager = logging_manager or LoggingManager()
        self.alert_manager = alert_manager or AlertManager()
        
        # 에러 처리 정책 로드
        self.error_policies = self._load_error_policies(config_path)
        
        # 에러 기록
        self.error_history = []
        self.retry_strategies = {}
        
        # 시스템 정보
        self.system_info = self._collect_system_info()
    
    def _load_error_policies(self, config_path: Optional[str]) -> Dict[str, Any]:
        """에러 처리 정책 로드"""
        default_policies = {
            "network_errors": {
                "severity": "medium",
                "strategy": "retry",
                "max_retries": 3,
                "retry_delay": 5,
                "fallback_action": "offline_mode"
            },
            "gpu_errors": {
                "severity": "high", 
                "strategy": "fallback",
                "fallback_action": "cpu_mode"
            },
            "model_loading_errors": {
                "severity": "high",
                "strategy": "retry",
                "max_retries": 2,
                "fallback_action": "alternative_model"
            },
            "checkpoint_errors": {
                "severity": "medium",
                "strategy": "continue",
                "fallback_action": "skip_checkpoint"
            },
            "wandb_errors": {
                "severity": "low",
                "strategy": "fallback", 
                "fallback_action": "offline_logging"
            },
            "general_errors": {
                "severity": "medium",
                "strategy": "continue",
                "max_retries": 1
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    custom_policies = json.load(f)
                default_policies.update(custom_policies)
                logger.info(f"에러 처리 정책 로드됨: {config_path}")
            except Exception as e:
                logger.warning(f"에러 처리 정책 로드 실패, 기본값 사용: {e}")
        
        return default_policies
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """시스템 정보 수집"""
        try:
            import psutil
            import torch
            
            return {
                'python_version': sys.version,
                'torch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'cpu_count': os.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / (1024**3),
                'hostname': os.uname().nodename if hasattr(os, 'uname') else 'unknown'
            }
        except Exception as e:
            logger.debug(f"시스템 정보 수집 실패: {e}")
            return {'error': 'system_info_collection_failed'}
    
    def handle_error(self, 
                    error: Exception,
                    component: str = None,
                    function: str = None,
                    error_category: str = "general_errors",
                    context: Dict[str, Any] = None) -> ErrorContext:
        """
        에러 처리 메인 함수
        
        Args:
            error: 발생한 예외
            component: 컴포넌트명
            function: 함수명  
            error_category: 에러 카테고리
            context: 추가 컨텍스트
            
        Returns:
            에러 컨텍스트
        """
        try:
            # 호출자 정보 자동 추출
            if not component or not function:
                frame = inspect.currentframe().f_back
                if frame:
                    component = component or frame.f_globals.get('__name__', 'unknown')
                    function = function or frame.f_code.co_name
            
            # 에러 ID 생성
            error_id = f"err_{int(time.time())}_{hash(str(error)) % 10000:04d}"
            
            # 에러 정책 가져오기
            policy = self.error_policies.get(error_category, self.error_policies["general_errors"])
            
            # 에러 컨텍스트 생성
            error_context = ErrorContext(
                error_id=error_id,
                timestamp=datetime.now().isoformat(),
                severity=ErrorSeverity(policy.get("severity", "medium")),
                strategy=ErrorStrategy(policy.get("strategy", "continue")),
                component=component or 'unknown',
                function=function or 'unknown',
                error_type=type(error).__name__,
                error_message=str(error),
                stack_trace=traceback.format_exc(),
                system_info=self.system_info,
                retry_count=0
            )
            
            # 에러 기록
            self.error_history.append(error_context)
            
            # 구조화된 로깅
            self.logging_manager.log_error_context(error_context)
            
            # 알림 발송
            self.alert_manager.send_alert(error_context)
            
            # 에러 처리 전략 실행
            self._execute_error_strategy(error_context, policy, context)
            
            return error_context
            
        except Exception as handler_error:
            # 에러 핸들러 자체에서 에러 발생 시 최소한의 로깅
            logger.critical(f"에러 핸들러 실패: {handler_error}")
            logger.critical(f"원본 에러: {error}")
            raise error  # 원본 에러를 다시 발생시킴
    
    def _execute_error_strategy(self, 
                               error_context: ErrorContext, 
                               policy: Dict[str, Any],
                               context: Dict[str, Any] = None) -> None:
        """에러 처리 전략 실행"""
        strategy = error_context.strategy
        
        if strategy == ErrorStrategy.RETRY:
            self._handle_retry(error_context, policy, context)
        elif strategy == ErrorStrategy.FALLBACK:
            self._handle_fallback(error_context, policy, context)
        elif strategy == ErrorStrategy.CONTINUE:
            self._handle_continue(error_context, policy, context)
        elif strategy == ErrorStrategy.ABORT:
            self._handle_abort(error_context, policy, context)
    
    def _handle_retry(self, error_context: ErrorContext, policy: Dict[str, Any], context: Dict[str, Any]) -> None:
        """재시도 처리"""
        max_retries = policy.get("max_retries", 3)
        retry_delay = policy.get("retry_delay", 1)
        
        if error_context.retry_count < max_retries:
            error_context.retry_count += 1
            logger.info(f"에러 재시도 {error_context.retry_count}/{max_retries}: {error_context.error_id}")
            
            if retry_delay > 0:
                time.sleep(retry_delay)
            
            # 재시도 로직은 호출자가 구현해야 함
            error_context.resolution_method = f"retry_{error_context.retry_count}"
        else:
            # 최대 재시도 초과 시 폴백
            logger.warning(f"최대 재시도 초과, 폴백 실행: {error_context.error_id}")
            fallback_action = policy.get("fallback_action", "continue")
            self._execute_fallback_action(error_context, fallback_action, context)
    
    def _handle_fallback(self, error_context: ErrorContext, policy: Dict[str, Any], context: Dict[str, Any]) -> None:
        """폴백 처리"""
        fallback_action = policy.get("fallback_action", "continue")
        logger.info(f"폴백 실행: {fallback_action} for {error_context.error_id}")
        self._execute_fallback_action(error_context, fallback_action, context)
    
    def _handle_continue(self, error_context: ErrorContext, policy: Dict[str, Any], context: Dict[str, Any]) -> None:
        """계속 진행 처리"""
        logger.info(f"에러 무시하고 계속 진행: {error_context.error_id}")
        error_context.resolved = True
        error_context.resolution_method = "continue"
    
    def _handle_abort(self, error_context: ErrorContext, policy: Dict[str, Any], context: Dict[str, Any]) -> None:
        """중단 처리"""
        logger.critical(f"치명적 에러로 인한 실험 중단: {error_context.error_id}")
        error_context.resolution_method = "abort"
        # 실제 중단은 호출자가 결정
    
    def _execute_fallback_action(self, error_context: ErrorContext, action: str, context: Dict[str, Any]) -> None:
        """폴백 액션 실행"""
        logger.info(f"폴백 액션 실행: {action}")
        
        if action == "offline_mode":
            logger.info("오프라인 모드로 전환")
        elif action == "cpu_mode":
            logger.info("CPU 모드로 전환")
        elif action == "alternative_model":
            logger.info("대안 모델 사용")
        elif action == "skip_checkpoint":
            logger.info("체크포인트 건너뛰기")
        elif action == "offline_logging":
            logger.info("오프라인 로깅으로 전환")
        
        error_context.resolved = True
        error_context.resolution_method = f"fallback_{action}"
    
    def get_error_stats(self) -> Dict[str, Any]:
        """에러 통계 반환"""
        if not self.error_history:
            return {"total_errors": 0}
        
        total_errors = len(self.error_history)
        severity_counts = {}
        strategy_counts = {}
        resolved_count = 0
        
        for error in self.error_history:
            # 심각도별 집계
            severity = error.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # 전략별 집계
            strategy = error.strategy.value
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            
            # 해결 여부
            if error.resolved:
                resolved_count += 1
        
        return {
            "total_errors": total_errors,
            "resolved_errors": resolved_count,
            "resolution_rate": resolved_count / total_errors if total_errors > 0 else 0,
            "severity_distribution": severity_counts,
            "strategy_distribution": strategy_counts,
            "recent_errors": [
                {
                    "error_id": err.error_id,
                    "timestamp": err.timestamp,
                    "component": err.component,
                    "error_type": err.error_type,
                    "severity": err.severity.value,
                    "resolved": err.resolved
                }
                for err in self.error_history[-10:]  # 최근 10개
            ]
        }


# 전역 인스턴스
_logging_manager = LoggingManager()
_alert_manager = AlertManager()
_error_handler = ErrorHandler(logging_manager=_logging_manager, alert_manager=_alert_manager)


def get_error_handler() -> ErrorHandler:
    """전역 에러 핸들러 반환"""
    return _error_handler


def get_logging_manager() -> LoggingManager:
    """전역 로깅 관리자 반환"""
    return _logging_manager


def handle_error(error: Exception, 
                component: str = None,
                function: str = None,
                error_category: str = "general_errors",
                context: Dict[str, Any] = None) -> ErrorContext:
    """
    에러 처리 전역 함수
    
    Args:
        error: 발생한 예외
        component: 컴포넌트명
        function: 함수명
        error_category: 에러 카테고리
        context: 추가 컨텍스트
        
    Returns:
        에러 컨텍스트
    """
    return _error_handler.handle_error(error, component, function, error_category, context)


def log_structured(level: str, 
                  message: str,
                  component: str = None,
                  function: str = None,
                  metadata: Dict[str, Any] = None) -> None:
    """
    구조화된 로깅 전역 함수
    
    Args:
        level: 로그 레벨
        message: 로그 메시지
        component: 컴포넌트명
        function: 함수명
        metadata: 추가 메타데이터
    """
    _logging_manager.log_structured(level, message, component, function, metadata)


def log_performance_metric(metric_name: str, 
                          value: Union[int, float],
                          unit: str = None,
                          component: str = None) -> None:
    """
    성능 메트릭 로깅 전역 함수
    
    Args:
        metric_name: 메트릭 이름
        value: 메트릭 값
        unit: 단위
        component: 컴포넌트명
    """
    _logging_manager.log_performance_metric(metric_name, value, unit, component)


def log_experiment_event(event_type: str, 
                        event_data: Dict[str, Any],
                        component: str = None) -> None:
    """
    실험 이벤트 로깅 전역 함수
    
    Args:
        event_type: 이벤트 타입
        event_data: 이벤트 데이터
        component: 컴포넌트명
    """
    _logging_manager.log_experiment_event(event_type, event_data, component)


def safe_execute(func: Callable, 
                *args,
                error_category: str = "general_errors",
                default_return: Any = None,
                **kwargs) -> Any:
    """
    안전한 함수 실행 (에러 처리 래퍼)
    
    Args:
        func: 실행할 함수
        *args: 함수 인자
        error_category: 에러 카테고리
        default_return: 에러 시 반환할 기본값
        **kwargs: 함수 키워드 인자
        
    Returns:
        함수 실행 결과 또는 기본값
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        error_context = handle_error(
            error=e,
            component=func.__module__ if hasattr(func, '__module__') else None,
            function=func.__name__ if hasattr(func, '__name__') else None,
            error_category=error_category
        )
        
        # 에러 전략에 따른 처리
        if error_context.strategy == ErrorStrategy.ABORT:
            raise e
        else:
            logger.warning(f"함수 실행 실패, 기본값 반환: {func.__name__ if hasattr(func, '__name__') else 'unknown'}")
            return default_return


def create_error_policy_config(config_path: str) -> None:
    """에러 처리 정책 설정 파일 생성"""
    default_config = {
        "network_errors": {
            "severity": "medium",
            "strategy": "retry", 
            "max_retries": 3,
            "retry_delay": 5,
            "fallback_action": "offline_mode"
        },
        "gpu_errors": {
            "severity": "high",
            "strategy": "fallback",
            "fallback_action": "cpu_mode"
        },
        "model_loading_errors": {
            "severity": "high",
            "strategy": "retry",
            "max_retries": 2,
            "fallback_action": "alternative_model"
        },
        "checkpoint_errors": {
            "severity": "medium",
            "strategy": "continue",
            "fallback_action": "skip_checkpoint"
        },
        "wandb_errors": {
            "severity": "low",
            "strategy": "fallback",
            "fallback_action": "offline_logging"
        }
    }
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(default_config, f, indent=2, ensure_ascii=False)
    
    logger.info(f"에러 처리 정책 설정 파일 생성됨: {config_path}")
