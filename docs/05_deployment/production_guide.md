# NLP 대화 요약 시스템 - 프로덕션 배포 가이드

## 목차
1. [배포 개요](#배포-개요)
2. [시스템 요구사항](#시스템-요구사항)
3. [기본 배포 설정](#기본-배포-설정)
4. [FastAPI 서버 구성](#fastapi-서버-구성)
5. [메트릭 모니터링](#메트릭-모니터링)
6. [로깅 시스템](#로깅-시스템)
7. [헬스체크 구성](#헬스체크-구성)
8. [성능 최적화](#성능-최적화)
9. [보안 설정](#보안-설정)
10. [문제 해결](#문제-해결)

---

## 배포 개요

이 가이드는 NLP 대화 요약 시스템을 프로덕션 환경에 안전하고 안정적으로 배포하기 위한 완전한 가이드입니다. Docker 컨테이너화, 모니터링, 로깅, 보안 설정을 포함한 모든 배포 과정을 다룹니다.

### 주요 특징
- 🚀 **FastAPI 기반 고성능 REST API**
- 📊 **Prometheus 메트릭 모니터링**
- 📝 **구조화된 JSON 로깅**
- 💚 **종합적인 헬스체크**
- 🔒 **API 키 기반 인증**
- ⚡ **자동 배치 처리 최적화**

---

## 시스템 요구사항

### 최소 요구사항
- **CPU**: 4 cores (Intel/AMD x64 또는 Apple Silicon)
- **RAM**: 8GB (권장: 16GB+)
- **Storage**: 20GB 여유 공간
- **OS**: Ubuntu 20.04+, macOS 12+, Windows 10+

### 권장 요구사항
- **GPU**: NVIDIA RTX 3070+ (CUDA 11.8+) 또는 Apple M1/M2
- **RAM**: 32GB
- **Storage**: SSD 50GB+

### 필수 소프트웨어
```bash
# Docker
docker --version  # Docker 20.0+

# Python
python --version  # Python 3.8+

# Git
git --version
```

---

## 기본 배포 설정

### 1. 프로젝트 준비

```bash
# 프로젝트 클론
git clone <your-repo-url>
cd nlp-sum-lyj

# 환경 설정
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정

```bash
# .env 파일 생성
cat > .env << EOF
# API 설정
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# 모델 설정
MODEL_PATH=outputs/best_model
MAX_BATCH_SIZE=16
DEVICE=auto

# 보안 설정
API_KEYS=your-secret-key-1,your-secret-key-2
RATE_LIMIT_PER_MINUTE=100

# 모니터링 설정
METRICS_PORT=9090
LOG_LEVEL=INFO
ENABLE_PROMETHEUS=true

# 데이터베이스 (선택사항)
DATABASE_URL=sqlite:///./app.db
EOF
```

### 3. Docker 이미지 빌드

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# 시스템 의존성
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드
COPY . .

# 포트 노출
EXPOSE 8000 9090

# 헬스체크
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 실행
CMD ["python", "deployment/main.py"]
```

```bash
# 이미지 빌드
docker build -t nlp-summarization-api:latest .

# 컨테이너 실행
docker run -d \
    --name nlp-api \
    -p 8000:8000 \
    -p 9090:9090 \
    --env-file .env \
    --restart unless-stopped \
    nlp-summarization-api:latest
```

---

## FastAPI 서버 구성

### 1. 메인 애플리케이션 구조

```python
# deployment/main.py
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import uvicorn
import os
from contextlib import asynccontextmanager

from api.routes import router
from api.middleware import setup_middleware
from monitoring.metrics import setup_metrics
from core.inference import InferenceEngine
from utils.logging import setup_logging

# 전역 상태 관리
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작 시 초기화
    setup_logging()
    logger.info("🚀 NLP 요약 API 시작")
    
    # 모델 로딩
    model_path = os.getenv("MODEL_PATH", "outputs/best_model")
    app_state["inference_engine"] = InferenceEngine(model_path)
    logger.info(f"✅ 모델 로딩 완료: {model_path}")
    
    # 메트릭 서버 시작
    if os.getenv("ENABLE_PROMETHEUS", "true").lower() == "true":
        metrics_port = int(os.getenv("METRICS_PORT", "9090"))
        setup_metrics(metrics_port)
        logger.info(f"📊 메트릭 서버 시작: 포트 {metrics_port}")
    
    yield
    
    # 종료 시 정리
    logger.info("🛑 NLP 요약 API 종료")

# FastAPI 앱 생성
app = FastAPI(
    title="NLP 대화 요약 API",
    description="대화 텍스트를 요약하는 REST API",
    version="1.0.0",
    lifespan=lifespan
)

# 미들웨어 설정
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 제한 필요
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 커스텀 미들웨어
setup_middleware(app)

# 라우터 등록
app.include_router(router, prefix="/api/v1")

# 기본 엔드포인트
@app.get("/")
async def root():
    return {
        "service": "NLP 대화 요약 API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    workers = int(os.getenv("API_WORKERS", "1"))
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        workers=workers,
        reload=False,
        access_log=True
    )
```

### 2. API 라우터 구성

```python
# deployment/api/routes.py
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional
import time

from .auth import get_api_key, rate_limiter
from .models import SummarizeRequest, SummarizeResponse, BatchSummarizeRequest
from ..monitoring.metrics import REQUEST_COUNT, REQUEST_LATENCY

router = APIRouter()

class SummarizeRequest(BaseModel):
    dialogue: str = Field(..., min_length=10, max_length=10000, description="요약할 대화 텍스트")
    max_length: Optional[int] = Field(100, ge=10, le=512, description="요약 최대 길이")
    min_length: Optional[int] = Field(10, ge=1, le=100, description="요약 최소 길이")
    num_beams: Optional[int] = Field(4, ge=1, le=10, description="빔 서치 크기")

class SummarizeResponse(BaseModel):
    summary: str = Field(..., description="생성된 요약문")
    original_length: int = Field(..., description="원본 대화 길이")
    summary_length: int = Field(..., description="요약문 길이")
    processing_time: float = Field(..., description="처리 시간(초)")

@router.post("/summarize", response_model=SummarizeResponse)
async def summarize_dialogue(
    request: SummarizeRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(get_api_key)
):
    """단일 대화 요약"""
    
    # 속도 제한 확인
    await rate_limiter.check_rate_limit(api_key)
    
    start_time = time.time()
    
    try:
        # 추론 실행
        inference_engine = app_state["inference_engine"]
        summary = inference_engine.predict_single(
            request.dialogue,
            max_length=request.max_length,
            min_length=request.min_length,
            num_beams=request.num_beams
        )
        
        processing_time = time.time() - start_time
        
        # 메트릭 기록
        REQUEST_COUNT.labels(method="POST", endpoint="summarize", status="success").inc()
        REQUEST_LATENCY.observe(processing_time)
        
        return SummarizeResponse(
            summary=summary,
            original_length=len(request.dialogue),
            summary_length=len(summary),
            processing_time=round(processing_time, 3)
        )
        
    except Exception as e:
        REQUEST_COUNT.labels(method="POST", endpoint="summarize", status="error").inc()
        raise HTTPException(status_code=500, detail=f"요약 처리 중 오류: {str(e)}")

@router.post("/batch-summarize")
async def batch_summarize(
    request: BatchSummarizeRequest,
    api_key: str = Depends(get_api_key)
):
    """배치 대화 요약"""
    
    # 배치 크기 제한
    if len(request.dialogues) > 50:
        raise HTTPException(status_code=400, detail="배치 크기는 최대 50개까지 가능합니다")
    
    await rate_limiter.check_rate_limit(api_key, multiplier=len(request.dialogues))
    
    start_time = time.time()
    
    try:
        inference_engine = app_state["inference_engine"]
        summaries = inference_engine.predict_batch(
            request.dialogues,
            batch_size=request.batch_size or 8
        )
        
        processing_time = time.time() - start_time
        
        REQUEST_COUNT.labels(method="POST", endpoint="batch-summarize", status="success").inc()
        
        return {
            "summaries": summaries,
            "count": len(summaries),
            "processing_time": round(processing_time, 3)
        }
        
    except Exception as e:
        REQUEST_COUNT.labels(method="POST", endpoint="batch-summarize", status="error").inc()
        raise HTTPException(status_code=500, detail=f"배치 처리 중 오류: {str(e)}")
```

---

## 메트릭 모니터링

### 1. Prometheus 메트릭 설정

```python
# deployment/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import functools
import time

# 메트릭 정의
REQUEST_COUNT = Counter(
    'nlp_api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'nlp_api_request_duration_seconds',
    'Request duration in seconds',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, float('inf')]
)

MODEL_MEMORY_USAGE = Gauge(
    'nlp_api_model_memory_bytes',
    'Model memory usage in bytes'
)

ACTIVE_CONNECTIONS = Gauge(
    'nlp_api_active_connections',
    'Number of active connections'
)

def monitor_performance(func):
    """성능 모니터링 데코레이터"""
    
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        status = "success"
        
        try:
            result = await func(*args, **kwargs)
            return result
        except Exception as e:
            status = "error"
            raise
        finally:
            duration = time.time() - start_time
            REQUEST_LATENCY.observe(duration)
            REQUEST_COUNT.labels(
                method="POST", 
                endpoint=func.__name__, 
                status=status
            ).inc()
    
    return wrapper

def start_metrics_server(port=9090):
    """메트릭 서버 시작"""
    start_http_server(port)
    print(f"📊 메트릭 서버 시작: 포트 {port}")

def setup_metrics(port=9090):
    """메트릭 시스템 초기화"""
    start_metrics_server(port)
    
    # 메모리 사용량 모니터링 시작
    import threading
    import psutil
    import torch
    
    def update_memory_metrics():
        while True:
            try:
                # 시스템 메모리
                process = psutil.Process()
                memory_bytes = process.memory_info().rss
                MODEL_MEMORY_USAGE.set(memory_bytes)
                
                time.sleep(30)  # 30초마다 업데이트
            except:
                pass
    
    thread = threading.Thread(target=update_memory_metrics, daemon=True)
    thread.start()
```

### 2. Grafana 대시보드 설정

```json
{
  "dashboard": {
    "title": "NLP 요약 API 모니터링",
    "panels": [
      {
        "title": "초당 요청 수",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(nlp_api_requests_total[5m])",
            "legendFormat": "{{status}}"
          }
        ]
      },
      {
        "title": "응답 시간 분포",
        "type": "graph", 
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(nlp_api_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, rate(nlp_api_request_duration_seconds_bucket[5m]))",
            "legendFormat": "50th percentile"
          }
        ]
      },
      {
        "title": "메모리 사용량",
        "type": "graph",
        "targets": [
          {
            "expr": "nlp_api_model_memory_bytes / 1024 / 1024",
            "legendFormat": "Memory (MB)"
          }
        ]
      }
    ]
  }
}
```

---

## 로깅 시스템

### 1. 구조화된 로깅

```python
# deployment/utils/logging.py
import logging
import json
import time
from typing import Dict, Any
from datetime import datetime

class StructuredLogger:
    """구조화된 JSON 로깅 클래스"""
    
    def __init__(self, name: str, level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # JSON 포맷터 설정
        formatter = logging.Formatter('%(message)s')
        
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def _format_log(self, level: str, message: str, **kwargs) -> str:
        """로그 메시지를 JSON 형식으로 포맷"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "message": message,
            "service": "nlp-summarization-api",
            **kwargs
        }
        return json.dumps(log_entry, ensure_ascii=False)
    
    def info(self, message: str, **kwargs):
        self.logger.info(self._format_log("INFO", message, **kwargs))
    
    def error(self, message: str, **kwargs):
        self.logger.error(self._format_log("ERROR", message, **kwargs))
    
    def warning(self, message: str, **kwargs):
        self.logger.warning(self._format_log("WARNING", message, **kwargs))

# 전역 로거
logger = StructuredLogger("api")

def setup_logging():
    """로깅 시스템 초기화"""
    import os
    
    log_level = os.getenv("LOG_LEVEL", "INFO")
    
    # 로그 파일 설정
    file_handler = logging.FileHandler("logs/api.log")
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    
    logger.logger.addHandler(file_handler)
    logger.info("로깅 시스템 초기화 완료", log_level=log_level)

def log_request(request_id: str, endpoint: str, duration: float, status: str):
    """API 요청 로깅"""
    logger.info(
        "API 요청 완료",
        request_id=request_id,
        endpoint=endpoint,
        duration=duration,
        status=status
    )
```

### 2. 요청 추적 미들웨어

```python
# deployment/api/middleware.py
import uuid
import time
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

class RequestTrackingMiddleware(BaseHTTPMiddleware):
    """요청 추적 미들웨어"""
    
    async def dispatch(self, request: Request, call_next):
        # 요청 ID 생성
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        
        start_time = time.time()
        
        # 요청 로깅
        logger.info(
            "API 요청 시작",
            request_id=request_id,
            method=request.method,
            url=str(request.url),
            client_ip=request.client.host
        )
        
        # 응답 처리
        try:
            response = await call_next(request)
            status = "success"
            status_code = response.status_code
        except Exception as e:
            status = "error"
            status_code = 500
            logger.error(
                "API 요청 처리 중 오류",
                request_id=request_id,
                error=str(e)
            )
            raise
        finally:
            duration = time.time() - start_time
            
            # 응답 로깅
            logger.info(
                "API 요청 완료",
                request_id=request_id,
                status=status,
                status_code=status_code,
                duration=round(duration, 3)
            )
        
        # 응답 헤더에 요청 ID 추가
        response.headers["X-Request-ID"] = request_id
        return response

def setup_middleware(app):
    """미들웨어 설정"""
    app.add_middleware(RequestTrackingMiddleware)
```

---

## 헬스체크 구성

### 1. 종합 헬스체크

```python
# deployment/api/health.py
from fastapi import APIRouter
from typing import Dict, Any
import torch
import psutil
import time
from pathlib import Path

router = APIRouter()

class HealthChecker:
    """종합적인 헬스체크 클래스"""
    
    def __init__(self, inference_engine=None):
        self.inference_engine = inference_engine
        self.startup_time = time.time()
    
    async def comprehensive_health_check(self) -> Dict[str, Any]:
        """종합 헬스체크 수행"""
        checks = {
            "system": await self._check_system_health(),
            "model": await self._check_model_health(),
            "dependencies": await self._check_dependencies(),
            "performance": await self._check_performance()
        }
        
        # 전체 상태 결정
        overall_status = "healthy"
        for check_name, check_result in checks.items():
            if not check_result.get("healthy", True):
                overall_status = "unhealthy"
                break
        
        return {
            "status": overall_status,
            "timestamp": time.time(),
            "uptime": time.time() - self.startup_time,
            "checks": checks
        }
    
    async def _check_system_health(self) -> Dict[str, Any]:
        """시스템 리소스 체크"""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            memory_healthy = memory.percent < 90
            disk_healthy = disk.percent < 90
            
            return {
                "healthy": memory_healthy and disk_healthy,
                "memory_usage_percent": memory.percent,
                "disk_usage_percent": disk.percent,
                "available_memory_gb": round(memory.available / (1024**3), 2)
            }
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    async def _check_model_health(self) -> Dict[str, Any]:
        """모델 상태 체크"""
        try:
            if self.inference_engine is None:
                return {"healthy": False, "error": "Model not loaded"}
            
            # 간단한 추론 테스트
            test_dialogue = "화자1: 안녕하세요\n화자2: 안녕하세요"
            start_time = time.time()
            result = self.inference_engine.predict_single(test_dialogue)
            inference_time = time.time() - start_time
            
            # GPU 메모리 체크 (사용 중인 경우)
            gpu_info = {}
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated(0)
                gpu_total = torch.cuda.get_device_properties(0).total_memory
                gpu_usage_percent = (gpu_memory / gpu_total) * 100
                
                gpu_info = {
                    "gpu_memory_usage_percent": round(gpu_usage_percent, 2),
                    "gpu_memory_used_gb": round(gpu_memory / (1024**3), 2),
                    "gpu_memory_total_gb": round(gpu_total / (1024**3), 2)
                }
            
            return {
                "healthy": True,
                "inference_time_seconds": round(inference_time, 3),
                "test_result_length": len(result) if result else 0,
                **gpu_info
            }
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    async def _check_dependencies(self) -> Dict[str, Any]:
        """의존성 라이브러리 체크"""
        try:
            import torch
            import transformers
            import fastapi
            
            return {
                "healthy": True,
                "torch_version": torch.__version__,
                "transformers_version": transformers.__version__,
                "fastapi_version": fastapi.__version__,
                "cuda_available": torch.cuda.is_available()
            }
        except ImportError as e:
            return {"healthy": False, "error": f"Missing dependency: {e}"}
    
    async def _check_performance(self) -> Dict[str, Any]:
        """성능 지표 체크"""
        # 실제 구현에서는 메트릭 저장소에서 데이터 조회
        return {
            "healthy": True,
            "note": "Performance metrics collection needed"
        }

# 전역 헬스체커
health_checker = None

@router.get("/health")
async def basic_health_check():
    """기본 헬스체크"""
    return {"status": "healthy", "timestamp": time.time()}

@router.get("/health/detailed")
async def detailed_health_check():
    """상세 헬스체크"""
    global health_checker
    if health_checker is None:
        # app_state에서 inference_engine 가져오기
        from ..main import app_state
        health_checker = HealthChecker(app_state.get("inference_engine"))
    
    return await health_checker.comprehensive_health_check()

@router.get("/health/ready")
async def readiness_check():
    """준비 상태 체크 (Kubernetes용)"""
    global health_checker
    if health_checker is None:
        from ..main import app_state
        health_checker = HealthChecker(app_state.get("inference_engine"))
    
    result = await health_checker.comprehensive_health_check()
    
    if result["status"] == "healthy":
        return {"status": "ready"}
    else:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Service not ready")
```

---

## 성능 최적화

### 1. 모델 최적화

```bash
# 모델 양자화 (옵션)
python scripts/quantize_model.py \
    --model-path outputs/best_model \
    --output-path outputs/quantized_model \
    --quantization-type int8

# 최적화된 모델로 배포
export MODEL_PATH=outputs/quantized_model
```

### 2. 배치 처리 최적화

```python
# deployment/optimization/batch_processor.py
import asyncio
from typing import List
from collections import deque
import time

class AdaptiveBatchProcessor:
    """동적 배치 크기 조정 프로세서"""
    
    def __init__(self, max_batch_size: int = 16, max_wait_time: float = 0.1):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.pending_requests = deque()
        self.processing = False
    
    async def add_request(self, dialogue: str, response_future: asyncio.Future):
        """요청을 배치에 추가"""
        self.pending_requests.append((dialogue, response_future, time.time()))
        
        if not self.processing:
            asyncio.create_task(self._process_batch())
    
    async def _process_batch(self):
        """배치 처리 실행"""
        if self.processing:
            return
        
        self.processing = True
        
        try:
            # 대기 시간 또는 최대 크기까지 기다림
            await asyncio.sleep(self.max_wait_time)
            
            # 배치 구성
            batch = []
            futures = []
            
            while self.pending_requests and len(batch) < self.max_batch_size:
                dialogue, future, timestamp = self.pending_requests.popleft()
                batch.append(dialogue)
                futures.append(future)
            
            if batch:
                # 실제 추론 실행
                from ..main import app_state
                inference_engine = app_state["inference_engine"]
                results = inference_engine.predict_batch(batch)
                
                # 결과 전달
                for future, result in zip(futures, results):
                    if not future.cancelled():
                        future.set_result(result)
                        
        except Exception as e:
            # 에러를 모든 future에 전파
            for future in futures:
                if not future.cancelled():
                    future.set_exception(e)
        finally:
            self.processing = False
            
            # 남은 요청이 있으면 다음 배치 처리
            if self.pending_requests:
                asyncio.create_task(self._process_batch())
```

### 3. 캐싱 시스템

```python
# deployment/caching/cache_manager.py
import hashlib
import time
from typing import Optional
import redis

class CacheManager:
    """Redis 기반 캐시 매니저"""
    
    def __init__(self, redis_url: Optional[str] = None, ttl_seconds: int = 3600):
        self.ttl_seconds = ttl_seconds
        
        if redis_url:
            self.redis_client = redis.from_url(redis_url)
        else:
            self.redis_client = None
    
    def _generate_key(self, dialogue: str) -> str:
        """캐시 키 생성"""
        return f"summary:{hashlib.md5(dialogue.encode()).hexdigest()}"
    
    async def get(self, dialogue: str) -> Optional[str]:
        """캐시에서 조회"""
        if not self.redis_client:
            return None
        
        key = self._generate_key(dialogue)
        result = self.redis_client.get(key)
        
        return result.decode('utf-8') if result else None
    
    async def set(self, dialogue: str, summary: str):
        """캐시에 저장"""
        if not self.redis_client:
            return
        
        key = self._generate_key(dialogue)
        self.redis_client.setex(key, self.ttl_seconds, summary)
```

---

## 보안 설정

### 1. API 키 인증

```python
# deployment/api/auth.py
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import time
import os

security = HTTPBearer()

def get_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    """API 키 검증"""
    valid_keys = os.getenv("API_KEYS", "").split(",")
    
    if credentials.credentials not in valid_keys:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    return credentials.credentials

class RateLimiter:
    """속도 제한기"""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = {}  # {api_key: [timestamp, ...]}
    
    async def check_rate_limit(self, api_key: str, multiplier: int = 1):
        """속도 제한 확인"""
        current_time = time.time()
        minute_ago = current_time - 60
        
        # 이전 요청 기록 정리
        if api_key in self.requests:
            self.requests[api_key] = [
                req_time for req_time in self.requests[api_key]
                if req_time > minute_ago
            ]
        else:
            self.requests[api_key] = []
        
        # 현재 요청 수 확인
        if len(self.requests[api_key]) >= self.requests_per_minute:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded"
            )
        
        # 현재 요청 기록
        for _ in range(multiplier):
            self.requests[api_key].append(current_time)

# 전역 속도 제한기
rate_limit_per_minute = int(os.getenv("RATE_LIMIT_PER_MINUTE", "100"))
rate_limiter = RateLimiter(rate_limit_per_minute)
```

### 2. 입력 검증

```python
# deployment/api/validation.py
import re
from fastapi import HTTPException

def validate_dialogue_input(dialogue: str) -> str:
    """대화 입력 검증 및 정제"""
    
    # 길이 검증
    if len(dialogue) > 10000:
        raise HTTPException(
            status_code=400,
            detail="대화 텍스트가 너무 깁니다 (최대 10,000자)"
        )
    
    if len(dialogue.strip()) < 10:
        raise HTTPException(
            status_code=400,
            detail="대화 텍스트가 너무 짧습니다 (최소 10자)"
        )
    
    # 악성 패턴 검사
    malicious_patterns = [
        r'<script.*?>.*?</script>',
        r'javascript:',
        r'data:text/html',
        r'vbscript:',
    ]
    
    for pattern in malicious_patterns:
        if re.search(pattern, dialogue, re.IGNORECASE | re.DOTALL):
            raise HTTPException(
                status_code=400,
                detail="잘못된 입력이 감지되었습니다"
            )
    
    # 기본 정리
    dialogue = re.sub(r'<[^>]+>', '', dialogue)  # HTML 태그 제거
    dialogue = re.sub(r'\s+', ' ', dialogue).strip()  # 공백 정리
    
    return dialogue
```

---

## 문제 해결

### 일반적인 배포 문제

#### 1. 컨테이너 시작 실패
```bash
# 로그 확인
docker logs nlp-api

# 일반적인 원인과 해결책:
# - 모델 파일 없음: outputs/best_model 디렉토리 확인
# - 포트 충돌: 다른 포트 사용
# - 메모리 부족: 배치 크기 줄이기
```

#### 2. 메모리 부족
```yaml
# docker-compose.yml에서 메모리 제한
services:
  nlp-api:
    image: nlp-summarization-api:latest
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          memory: 4G
```

#### 3. 성능 이슈
```bash
# CPU/메모리 사용률 모니터링
docker stats nlp-api

# GPU 사용률 확인 (NVIDIA GPU)
nvidia-smi
```

### 로그 분석

```bash
# API 로그 확인
tail -f logs/api.log

# 에러 로그만 필터링
grep "ERROR" logs/api.log

# 특정 요청 ID 추적
grep "abc12345" logs/api.log
```

### 성능 튜닝

1. **배치 크기 최적화**
   - GPU 메모리에 맞게 조정
   - 메모리 사용량 vs 처리 속도 트레이드오프

2. **워커 프로세스 수 조정**
   - CPU 코어 수에 맞게 설정
   - 메모리 사용량 고려

3. **캐싱 활용**
   - 자주 요청되는 내용 캐싱
   - Redis 또는 메모리 캐시 사용

---

## 추가 리소스

### 모니터링 설정

```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9091:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana

volumes:
  grafana-storage:
```

### 배포 스크립트

```bash
#!/bin/bash
# scripts/deploy.sh

echo "🚀 NLP 요약 API 배포 시작"

# 환경 확인
if [ ! -f ".env" ]; then
    echo "❌ .env 파일이 없습니다"
    exit 1
fi

# 이미지 빌드
echo "🔨 Docker 이미지 빌드..."
docker build -t nlp-summarization-api:latest .

# 기존 컨테이너 정지
echo "🛑 기존 서비스 정지..."
docker stop nlp-api 2>/dev/null || true
docker rm nlp-api 2>/dev/null || true

# 새 컨테이너 시작
echo "🚀 새 서비스 시작..."
docker run -d \
    --name nlp-api \
    -p 8000:8000 \
    -p 9090:9090 \
    --env-file .env \
    --restart unless-stopped \
    -v $(pwd)/outputs:/app/outputs \
    nlp-summarization-api:latest

# 헬스체크
echo "💚 헬스체크 대기..."
sleep 10

if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ 배포 성공!"
    echo "📊 API 문서: http://localhost:8000/docs"
    echo "📈 메트릭: http://localhost:9090"
else
    echo "❌ 배포 실패 - 로그 확인 필요"
    docker logs nlp-api
    exit 1
fi
```

이 프로덕션 배포 가이드를 통해 안정적이고 확장 가능한 NLP 요약 서비스를 운영할 수 있습니다. 각 섹션의 코드는 실제 운영 환경에서 사용할 수 있도록 구성되었습니다.
