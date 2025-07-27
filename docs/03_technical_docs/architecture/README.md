# 🏗️ 아키텍처

프로젝트의 전체 구조와 설계 철학을 설명하는 문서입니다.

## 📋 포함 문서

### 🗂️ [프로젝트 구조](./project_structure.md)
- 전체 프로젝트 디렉토리 구조 상세 분석
- 각 모듈의 역할과 책임 영역
- 파일 간 의존성 및 상호작용 패턴
- 설정 파일 및 데이터 플로우

### 📊 [코드-문서 일치성 분석](./code_documentation_analysis.md)
- 소스 코드와 문서 간의 체계적 일치성 분석
- 15개 핵심 파일과 49개 문서의 정밀 비교
- 개선 권장사항 및 3단계 실행 로드맵
- 프로젝트 품질 평가 및 향후 발전 방향

### 🔍 [코드 분석](./code_analysis.md)
- 개별 모듈의 상세 코드 분석
- 설계 패턴 및 최적화 기법
- 성능 특성 및 확장성 분석

## 🎯 아키텍처 핵심 원칙

### 1. 모듈화 설계
- **단일 책임 원칙**: 각 모듈은 명확한 하나의 책임
- **느슨한 결합**: 모듈 간 최소한의 의존성
- **높은 응집도**: 관련 기능들의 논리적 그룹핑

### 2. 상대 경로 기반
- **PathManager**: 프로젝트 루트 기준 상대 경로 강제
- **포터빌리티**: 환경에 관계없는 일관된 경로 처리
- **안전성**: 절대 경로 사용 금지로 보안 강화

### 3. 디바이스 추상화
- **자동 감지**: MPS/CUDA/CPU 자동 선택
- **최적화**: 디바이스별 성능 튜닝
- **호환성**: 모든 환경에서 동일한 API

## 📊 시스템 구성도

```
nlp-sum-lyj/
├── code/                    # 소스 코드
│   ├── core/               # 핵심 기능 모듈
│   ├── utils/              # 공통 유틸리티
│   ├── data_augmentation/  # 데이터 증강
│   ├── preprocessing/      # 전처리 파이프라인
│   ├── postprocessing/     # 후처리 파이프라인
│   ├── models/             # 모델 관련 (가중치 등)
│   └── ensemble/           # 앙상블 기법
├── config/                 # 설정 파일
├── data/                   # 데이터셋
├── docs/                   # 문서 (현재 위치)
└── outputs/                # 실험 결과
```

## 🔄 데이터 플로우

1. **데이터 로딩**: PathManager → DataProcessor
2. **전처리**: TextNormalizer → AugmentedDataLoader
3. **학습**: WeightedTrainer → ModelRegistry
4. **추론**: InferenceEngine → PostProcessingPipeline
5. **평가**: RougeCalculator → ExperimentTracker

## 🛡️ 설계 패턴

### 팩토리 패턴
- 디바이스별 최적화 설정 자동 생성
- 모델 아키텍처별 인스턴스 생성

### 옵저버 패턴
- 실험 진행 상황 실시간 추적
- WandB 통합 로깅

### 전략 패턴
- 다양한 앙상블 전략 선택
- 후처리 파이프라인 조합

## 🔗 관련 문서

- **API 참조**: [구현 세부사항](../api_reference/README.md)
- **알고리즘**: [핵심 알고리즘](../algorithms/README.md)
- **개발 가이드**: [기여 방법](../../07_development/README.md)

---
📍 **위치**: `docs/03_technical_docs/architecture/`
