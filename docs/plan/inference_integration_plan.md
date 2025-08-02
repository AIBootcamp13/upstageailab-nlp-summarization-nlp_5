# 추론 시스템 통합 계획

## 1. 개요

### 1.1 목적
현재 `run_main_5_experiments.sh` 시스템에서 누락된 추론(inference) 기능을 통합하여, baseline.py와 동일한 전체 워크플로우를 구현한다.

### 1.2 현재 상황
- **구현됨**: 데이터 로드, 전처리, 모델 학습, 평가
- **누락됨**: 학습 후 test.csv에 대한 자동 추론 및 제출 파일 생성
- **문제점**: auto_experiment_runner.py에서 `_run_test_inference()` 호출하지만 실제 구현 없음

### 1.3 목표
- baseline.py의 추론 기능을 완벽하게 재현
- 모든 모델 타입(mT5, T5, BART) 지원
- 자동화된 추론 및 결과 관리

## 2. 시스템 아키텍처

### 2.1 전체 흐름도
```
run_main_5_experiments.sh
    ↓
auto_experiment_runner.py
    ↓
trainer.py (학습)
    ↓
PostTrainingInference (추론)  ← 새로 구현
    ↓
CompetitionCSVManager (결과 관리)
    ↓
prediction/ 폴더에 결과 저장
```

### 2.2 컴포넌트 관계도
```
┌─────────────────────────────────────────────┐
│     run_main_5_experiments.sh               │
│  - GPU 메모리 관리                          │
│  - 실험 순차 실행                          │
└────────────────┬───────────────────────────┘
                 │
┌────────────────▼───────────────────────────┐
│     auto_experiment_runner.py              │
│  - 실험 설정 로드                          │
│  - trainer.py 실행                         │
│  - _run_test_inference() 호출              │ ← 구현 필요
└────────────────┬───────────────────────────┘
                 │
┌────────────────▼───────────────────────────┐
│     PostTrainingInference                  │ ← 새로 구현
│  - 체크포인트 로드                         │
│  - test.csv 추론                           │
│  - 결과 후처리                             │
└────────────────┬───────────────────────────┘
                 │
┌────────────────▼───────────────────────────┐
│     InferenceEngine (기존)                 │
│  - 모델 타입 자동 감지                     │
│  - 배치 추론 실행                          │
│  - 결과 생성                               │
└─────────────────────────────────────────────┘
```

## 3. 구현 상세 설계

### 3.1 auto_experiment_runner.py 수정

#### 3.1.1 _run_test_inference() 메서드 구현
```python
def _run_test_inference(self, 
                       experiment_id: str,
                       checkpoint_path: str,
                       config: Dict[str, Any]) -> Dict[str, Any]:
    """
    학습 완료 후 test.csv에 대한 추론 수행
    
    기능:
    1. 모델별 설정 적용
    2. PostTrainingInference 초기화
    3. test.csv 추론 실행
    4. 결과 파일 저장 및 인덱싱
    """
```

### 3.2 PostTrainingInference 클래스 구현

#### 3.2.1 주요 메서드
- `__init__()`: 초기화 및 설정 로드
- `load_model_and_tokenizer()`: 체크포인트에서 모델 로드
- `prepare_test_data()`: baseline.py의 prepare_test_dataset() 재현
- `run_inference()`: 실제 추론 실행
- `postprocess_results()`: 특수 토큰 제거 등 후처리
- `save_results()`: CSV 파일 저장

#### 3.2.2 baseline.py 호환성 보장
- BOS/EOS 토큰 처리 동일하게 구현
- DatasetForInference 클래스 동일하게 구현
- 특수 토큰 제거 로직 동일하게 구현

### 3.3 모델별 처리

#### 3.3.1 ModelSpecificHandler 클래스
```python
class ModelSpecificHandler:
    """모델별 특성 처리"""
    
    model_configs = {
        "mt5": {
            "model_class": "T5ForConditionalGeneration",
            "use_prefix": True,
            "prefix": "dialogue summarization in korean: "
        },
        "t5": {
            "model_class": "T5ForConditionalGeneration",
            "use_prefix": True,
            "prefix": "dialogue summarization in korean: "
        },
        "bart": {
            "model_class": "BartForConditionalGeneration",
            "use_prefix": False,
            "prefix": ""
        }
    }
```

## 4. 구현 계획

### 4.1 Phase 1: 기본 구현 (1일차)
- [ ] auto_experiment_runner.py의 `_run_test_inference()` 구현
- [ ] PostTrainingInference 클래스 기본 구조 구현
- [ ] 단위 테스트 작성

### 4.2 Phase 2: 모델별 지원 (2일차)
- [ ] ModelSpecificHandler 구현
- [ ] mT5, T5, BART 각각 테스트
- [ ] 모델별 최적화 설정 적용

### 4.3 Phase 3: 통합 테스트 (3일차)
- [ ] 전체 워크플로우 테스트
- [ ] 에러 처리 및 복구 메커니즘 테스트
- [ ] 성능 최적화

### 4.4 Phase 4: 문서화 및 배포 (4일차)
- [ ] 사용자 가이드 작성
- [ ] API 문서 작성
- [ ] 배포 및 검증

## 5. 테스트 계획

### 5.1 단위 테스트
- PostTrainingInference 각 메서드 테스트
- ModelSpecificHandler 테스트
- 에러 상황 테스트

### 5.2 통합 테스트
- 전체 실험 실행 후 추론까지 자동 실행 확인
- 다양한 모델로 테스트
- 결과 파일 형식 검증

### 5.3 성능 테스트
- GPU 메모리 사용량 모니터링
- 추론 속도 측정
- 배치 크기 최적화

## 6. 위험 요소 및 대응 방안

### 6.1 위험 요소
1. **메모리 부족**: 학습 직후 추론 시 GPU 메모리 부족
2. **모델 호환성**: 다양한 모델 아키텍처 지원
3. **체크포인트 탐색**: 최적 체크포인트 찾기 실패

### 6.2 대응 방안
1. **메모리 관리**: 
   - 학습 후 명시적 메모리 정리
   - 추론 전 GPU 메모리 확인
   
2. **모델 호환성**:
   - try-except로 여러 모델 클래스 시도
   - 모델별 설정 사전 정의
   
3. **체크포인트 탐색**:
   - 다양한 경로 패턴 시도
   - 폴백 메커니즘 구현

## 7. 성공 기준

### 7.1 기능적 요구사항
- [ ] baseline.py의 모든 추론 기능 재현
- [ ] 모든 모델 타입 지원
- [ ] 자동화된 실행

### 7.2 비기능적 요구사항
- [ ] 추론 시간 5분 이내
- [ ] GPU 메모리 사용량 22GB 이하
- [ ] 에러 발생 시 명확한 메시지

### 7.3 검증 방법
1. baseline.py 결과와 비교
2. 다양한 모델로 테스트
3. 제출 파일 형식 확인

## 8. 참고 자료

### 8.1 관련 파일
- `/code/baseline.py`: 원본 추론 로직
- `/code/core/inference.py`: 기존 추론 엔진
- `/code/auto_experiment_runner.py`: 통합 지점

### 8.2 설정 예시
```yaml
inference:
  batch_size: 32
  no_repeat_ngram_size: 2
  early_stopping: true
  generate_max_length: 100
  num_beams: 4
  remove_tokens: ['<usr>', '<s>', '</s>', '<pad>']
```
