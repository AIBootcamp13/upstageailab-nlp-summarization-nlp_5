# 추론 최적화 가이드

이 섹션은 `core/inference.py`의 InferenceEngine을 효율적으로 활용하기 위한 고급 기능과 최적화 기법을 다룹니다.

## 문서 구성

### [고급 추론 엔진 사용 가이드](./advanced_inference_guide.md)
InferenceEngine의 모든 고급 기능을 상세히 설명하는 포괄적인 가이드입니다.

**주요 내용**:
- 다중 입력 형식 지원 (string, list, DataFrame)
- 자동 디바이스 최적화 (CUDA, MPS, CPU)
- 배치 처리 최적화
- 캐시 시스템 활용
- 메모리 관리 및 성능 최적화
- 대회 제출 형식 처리

## 대상 독자

### 고급 사용자
- 대용량 데이터 처리가 필요한 사용자
- 성능 최적화를 추구하는 개발자
- 다양한 하드웨어 환경에서 작업하는 사용자

### 연구자 및 데이터 사이언티스트
- 효율적인 실험 환경 구축이 필요한 연구자
- 배치 처리를 통한 대규모 평가가 필요한 팀
- 메모리 제약 환경에서 작업하는 사용자

## 빠른 시작

기본적인 고급 기능 사용법:

```python
from core.inference import InferenceEngine, InferenceConfig

# 자동 최적화된 설정으로 초기화
config = InferenceConfig(
    model_path="gogamza/kobart-base-v2",
    device=None,      # 자동 감지
    batch_size=None,  # 자동 조정
    fp16=None         # 자동 설정
)

engine = InferenceEngine(config)

# 다양한 입력 형식 지원
summary = engine("단일 대화 텍스트")
summaries = engine(["대화1", "대화2", "대화3"])
result_df = engine(dataframe)
```

## 성능 특징

### 자동 최적화
- **플랫폼별 자동 감지**: CUDA, MPS (Apple Silicon), CPU
- **메모리 기반 배치 크기 조정**: 하드웨어 성능에 맞춤
- **Mixed Precision 자동 설정**: 성능과 정확도 균형

### 배치 처리 효율성
- **DataLoader 기반**: 메모리 효율적인 대용량 데이터 처리
- **진행률 추적**: 장시간 작업의 상태 모니터링
- **메모리 관리**: OOM 방지를 위한 적응적 처리

### 다양한 사용 사례 지원
- **실시간 단일 추론**: 즉시 응답이 필요한 서비스
- **대용량 배치 처리**: 연구 및 평가를 위한 대규모 데이터셋
- **구조화된 데이터 처리**: CSV/DataFrame 직접 처리

## 하드웨어별 최적화 가이드

### CUDA GPU 환경
- 큰 배치 크기 (32+)
- FP16 Mixed Precision
- 다중 워커 활용

### Apple Silicon (MPS) 환경
- 중간 배치 크기 (16)
- FP32 유지 (안정성)
- 메모리 효율성 중심

### CPU 환경
- 작은 배치 크기 (4-8)
- 단일 워커 사용
- 캐시 최적화

## 문제 해결

일반적인 문제와 해결책:

- **메모리 부족**: 배치 크기 자동 조정
- **플랫폼 호환성**: 자동 디바이스 감지
- **성능 최적화**: 하드웨어별 맞춤 설정

## 관련 문서

- **[API 참조](../../03_technical_docs/api_reference/core_modules.md)**: 전체 API 상세 정보
- **[성능 최적화](../../03_technical_docs/performance_optimization.md)**: 시스템 수준 최적화
- **[실험 관리](../experiment_management/)**: 체계적인 실험 설계

---

**다음 단계**: [고급 추론 엔진 사용 가이드](./advanced_inference_guide.md)를 통해 모든 고급 기능을 마스터하세요.
