"""
HuggingFace 모델 로딩 안정성 강화 유틸리티

네트워크 연결 실패 시에도 모델 로딩이 실패하지 않도록 하는 견고한 모델 로더.
로컬 캐시 활용, 재시도 로직, 오프라인 모드 지원을 통해 실험 연속성을 보장합니다.

통합 에러 처리 시스템 적용:
- 네트워크 에러: 자동 재시도 및 로컬 캐시 활용
- 모델 로딩 에러: 대안 모델 또는 오프라인 모드로 대체
- 구조화된 로깅으로 모델 로딩 성능 및 상태 추적
"""

import os
import time
import logging
import socket
from pathlib import Path
from typing import Dict, Any, Optional, Union, Type
import torch
from transformers import (
    AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM,
    PreTrainedModel, PreTrainedTokenizer
)
# 통합 에러 처리 시스템 import
from .error_handling import (
    handle_error, log_structured, log_performance_metric,
    safe_execute, get_logging_manager
)

logger = logging.getLogger(__name__)


class RobustModelLoader:
    """
    견고한 HuggingFace 모델 로더
    
    네트워크 연결 실패 시에도 모델 로딩이 중단되지 않도록 하는 Fail-Safe 모델 로더.
    로컬 캐시 우선 확인, 재시도 로직, 오프라인 모드 자동 전환을 통해 실험 연속성을 보장합니다.
    """
    def __init__(self, 
                 cache_dir: str = "./hf_cache", 
                 offline_fallback: bool = True,
                 max_retries: int = 3,
                 retry_delay: float = 1.0):
        """
        Args:
            cache_dir: HuggingFace 캐시 디렉토리 경로
            offline_fallback: 네트워크 실패 시 오프라인 모드 사용 여부
            max_retries: 최대 재시도 횟수
            retry_delay: 재시도 간격 (초)
        """
        self.cache_dir = Path(cache_dir).resolve()
        self.offline_fallback = offline_fallback
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # 캐시 디렉토리 생성
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 환경변수 설정
        os.environ['TRANSFORMERS_CACHE'] = str(self.cache_dir)
        os.environ['HF_HOME'] = str(self.cache_dir)
        
        logger.info(f"🗂️ HuggingFace 캐시 디렉토리: {self.cache_dir}")
    
    def safe_from_pretrained(self, 
                           model_class: Type[PreTrainedModel], 
                           model_name: str, 
                           **kwargs) -> PreTrainedModel:
        """
        안전한 모델 로딩 (네트워크 실패 대응)
        
        Args:
            model_class: 모델 클래스 (AutoModelForSeq2SeqLM, AutoTokenizer 등)
            model_name: 모델 이름 또는 경로
            **kwargs: from_pretrained()에 전달할 추가 인자
            
        Returns:
            로딩된 모델 또는 토크나이저
        """
        # 캐시 디렉토리 설정
        kwargs.setdefault('cache_dir', str(self.cache_dir))
        
        # 1단계: 로컬 캐시 우선 확인
        if self._check_local_cache(model_name):
            logger.info(f"📦 로컬 캐시에서 {model_name} 로딩 시도")
            try:
                return self._try_load_from_cache(model_class, model_name, **kwargs)
            except Exception as e:
                logger.warning(f"캐시에서 로딩 실패: {e}")
        
        # 2단계: 네트워크 다운로드 시도 (재시도 로직)
        return self._try_network_download(model_class, model_name, **kwargs)
    
    def _check_local_cache(self, model_name: str) -> bool:
        """
        로컬 캐시에 모델이 존재하는지 확인
        
        Args:
            model_name: 모델 이름
            
        Returns:
            캐시 존재 여부
        """
        try:
            # HuggingFace 캐시 구조 확인
            cache_paths = [
                self.cache_dir / "models--" / model_name.replace("/", "--"),
                self.cache_dir / "hub" / f"models--{model_name.replace('/', '--')}",
                # 다양한 캐시 경로 패턴 지원
            ]
            
            for cache_path in cache_paths:
                if cache_path.exists() and any(cache_path.iterdir()):
                    logger.debug(f"✅ 로컬 캐시 발견: {cache_path}")
                    return True
            
            return False
            
        except Exception as e:
            logger.debug(f"캐시 확인 중 오류: {e}")
            return False
    
    def _try_load_from_cache(self, 
                           model_class: Type[PreTrainedModel], 
                           model_name: str, 
                           **kwargs) -> PreTrainedModel:
        """
        로컬 캐시에서 모델 로딩 시도
        
        Args:
            model_class: 모델 클래스
            model_name: 모델 이름
            **kwargs: 추가 인자
            
        Returns:
            로딩된 모델
        """
        # 오프라인 모드로 캐시에서만 로딩
        kwargs_offline = kwargs.copy()
        kwargs_offline['local_files_only'] = True
        
        try:
            model = model_class.from_pretrained(model_name, **kwargs_offline)
            logger.info(f"✅ 캐시에서 {model_name} 로딩 성공")
            return model
        except Exception as e:
            logger.warning(f"캐시 로딩 실패: {e}")
            raise
    
    def _try_network_download(self, 
                            model_class: Type[PreTrainedModel], 
                            model_name: str, 
                            **kwargs) -> PreTrainedModel:
        """
        네트워크를 통한 모델 다운로드 시도 (재시도 로직 포함)
        
        Args:
            model_class: 모델 클래스
            model_name: 모델 이름
            **kwargs: 추가 인자
            
        Returns:
            로딩된 모델
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"🌐 네트워크에서 {model_name} 다운로드 시도 ({attempt + 1}/{self.max_retries})")
                
                # 네트워크 연결 사전 테스트
                if not self._test_huggingface_connectivity():
                    raise ConnectionError("HuggingFace Hub 연결 실패")
                
                # 모델 다운로드 시도
                model = model_class.from_pretrained(model_name, **kwargs)
                logger.info(f"✅ 네트워크에서 {model_name} 다운로드 성공")
                return model
                
            except Exception as e:
                last_exception = e
                logger.warning(f"다운로드 실패 ({attempt + 1}/{self.max_retries}): {e}")
                
                # 네트워크 에러인지 확인
                if self._is_network_error(e):
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay * (2 ** attempt)  # 지수 백오프
                        logger.info(f"재시도 전 {wait_time:.1f}초 대기...")
                        time.sleep(wait_time)
                        continue
                else:
                    # 네트워크 에러가 아닌 경우 즉시 실패
                    logger.error(f"네트워크 외 에러로 즉시 실패: {e}")
                    break
        
        # 모든 재시도 실패 시 오프라인 모드 시도
        return self._handle_download_failure(model_class, model_name, last_exception, **kwargs)
    
    def _test_huggingface_connectivity(self) -> bool:
        """
        HuggingFace Hub 연결 테스트
        
        Returns:
            연결 가능 여부
        """
        try:
            # DNS 해결 테스트
            socket.gethostbyname('huggingface.co')
            
            # 포트 연결 테스트
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex(('huggingface.co', 443))
            sock.close()
            
            if result == 0:
                logger.debug("HuggingFace Hub 연결 테스트 성공")
                return True
            else:
                logger.debug(f"HuggingFace Hub 포트 연결 실패: {result}")
                return False
                
        except Exception as e:
            logger.debug(f"HuggingFace Hub 연결 테스트 실패: {e}")
            return False
    
    def _is_network_error(self, exception: Exception) -> bool:
        """
        네트워크 관련 에러인지 판단
        
        Args:
            exception: 발생한 예외
            
        Returns:
            네트워크 에러 여부
        """
        error_str = str(exception).lower()
        network_keywords = [
            'connection', 'timeout', 'network', 'resolve', 'unreachable',
            'offline', 'internet', 'dns', 'socket', 'ssl', 'certificate',
            'http', 'https', '404', '503', '502', '500', 'requests'
        ]
        
        return any(keyword in error_str for keyword in network_keywords)
    
    def _handle_download_failure(self, 
                                model_class: Type[PreTrainedModel], 
                                model_name: str, 
                                last_exception: Exception,
                                **kwargs) -> PreTrainedModel:
        """
        다운로드 실패 시 처리 (오프라인 모드 시도)
        
        Args:
            model_class: 모델 클래스
            model_name: 모델 이름
            last_exception: 마지막 예외
            **kwargs: 추가 인자
            
        Returns:
            로딩된 모델
        """
        if not self.offline_fallback:
            logger.error(f"다운로드 실패 및 오프라인 모드 비활성화: {last_exception}")
            raise last_exception
        
        logger.warning("🔄 오프라인 모드로 전환하여 캐시에서 로딩 시도")
        
        try:
            return self._try_load_from_cache(model_class, model_name, **kwargs)
        except Exception as cache_error:
            logger.error(f"❌ 캐시에서도 로딩 실패: {cache_error}")
            logger.error(f"원본 다운로드 에러: {last_exception}")
            
            # 가장 구체적인 에러 메시지 제공
            error_msg = f"""
모델 로딩 완전 실패: {model_name}
1. 네트워크 다운로드 실패: {last_exception}
2. 로컬 캐시 로딩 실패: {cache_error}

해결 방법:
- 네트워크 연결 확인
- HuggingFace Hub 접근성 확인
- 캐시 디렉토리 권한 확인: {self.cache_dir}
"""
            raise ConnectionError(error_msg)
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        캐시 디렉토리 정보 반환
        
        Returns:
            캐시 정보
        """
        try:
            cache_size = sum(f.stat().st_size for f in self.cache_dir.rglob('*') if f.is_file())
            cache_size_mb = cache_size / (1024 * 1024)
            
            model_count = len(list((self.cache_dir / "models--").glob("*"))) if (self.cache_dir / "models--").exists() else 0
            
            return {
                'cache_dir': str(self.cache_dir),
                'cache_size_mb': round(cache_size_mb, 2),
                'cached_models': model_count,
                'exists': self.cache_dir.exists()
            }
        except Exception as e:
            logger.warning(f"캐시 정보 조회 실패: {e}")
            return {
                'cache_dir': str(self.cache_dir),
                'error': str(e)
            }


# 전역 RobustModelLoader 인스턴스
_robust_model_loader = RobustModelLoader()


def safe_load_model(model_class: Type[PreTrainedModel], 
                   model_name: str, 
                   **kwargs) -> PreTrainedModel:
    """
    안전한 모델 로딩 (전역 함수)
    
    네트워크 연결 실패 시 로컬 캐시를 활용하여 모델 로딩.
    기존 from_pretrained() 호출을 이 함수로 교체하여 견고성 확보.
    
    통합 에러 처리 시스템을 사용하여 모델 로딩 에러를 자동으로 처리하고
    대안 모델 또는 오프라인 모드로 안전하게 대체합니다.
    
    Args:
        model_class: 모델 클래스 (AutoModelForSeq2SeqLM, AutoTokenizer 등)
        model_name: 모델 이름 또는 경로
        **kwargs: from_pretrained()에 전달할 추가 인자
        
    Returns:
        로딩된 모델 또는 토크나이저
        
    Example:
        # 기존 방식
        model = AutoModelForSeq2SeqLM.from_pretrained("gogamza/kobart-base-v2")
        
        # 안전한 방식  
        model = safe_load_model(AutoModelForSeq2SeqLM, "gogamza/kobart-base-v2")
    """
    # 통합 에러 처리로 모델 로딩 래핑
    return safe_execute(
        func=_execute_model_loading,
        model_class=model_class,
        model_name=model_name,
        kwargs=kwargs,
        error_category="model_loading_errors",
        default_return=None  # 모델 로딩 실패 시 None 반환
    )


def _execute_model_loading(model_class: Type[PreTrainedModel], 
                          model_name: str, 
                          kwargs: Dict[str, Any]) -> PreTrainedModel:
    """
    실제 모델 로딩 실행 (내부 함수)
    """
    start_time = time.time()
    
    log_structured(
        level="INFO",
        message=f"모델 로딩 시작: {model_name}",
        component="model_loading_utils",
        function="safe_load_model",
        metadata={"model_name": model_name, "model_class": model_class.__name__}
    )
    
    try:
        # RobustModelLoader를 통한 안전한 로딩
        result = _robust_model_loader.safe_from_pretrained(model_class, model_name, **kwargs)
        
        # 성공 시 성능 메트릭 로깅
        loading_duration = time.time() - start_time
        log_performance_metric(
            metric_name="model_loading_duration",
            value=loading_duration,
            unit="seconds",
            component="model_loading_utils"
        )
        
        log_structured(
            level="INFO",
            message=f"모델 로딩 완료: {model_name}",
            component="model_loading_utils",
            function="safe_load_model",
            metadata={
                "model_name": model_name,
                "loading_duration": loading_duration,
                "success": True
            }
        )
        
        return result
        
    except Exception as e:
        # 에러 발생 시 통합 에러 처리로 전달
        loading_duration = time.time() - start_time
        
        log_structured(
            level="ERROR",
            message=f"모델 로딩 실패: {model_name}",
            component="model_loading_utils",
            function="safe_load_model",
            metadata={
                "model_name": model_name,
                "loading_duration": loading_duration,
                "success": False,
                "error": str(e)
            }
        )
        
        # 원본 에러 재발생 (safe_execute에서 처리됨)
        raise e


def safe_load_tokenizer(model_name: str, **kwargs) -> PreTrainedTokenizer:
    """
    안전한 토크나이저 로딩 (편의 함수)
    
    Args:
        model_name: 모델 이름 또는 경로
        **kwargs: from_pretrained()에 전달할 추가 인자
        
    Returns:
        로딩된 토크나이저
    """
    return safe_load_model(AutoTokenizer, model_name, **kwargs)


def get_model_cache_info() -> Dict[str, Any]:
    """
    모델 캐시 정보 조회 (전역 함수)
    
    Returns:
        캐시 정보
    """
    return _robust_model_loader.get_cache_info()


def check_model_availability(model_name: str) -> Dict[str, bool]:
    """
    모델 사용 가능성 확인
    
    Args:
        model_name: 모델 이름
        
    Returns:
        사용 가능성 정보
    """
    return {
        'local_cache': _robust_model_loader._check_local_cache(model_name),
        'network': _robust_model_loader._test_huggingface_connectivity()
    }
