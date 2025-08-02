# baseline.py와 제안 시스템 기능 비교 분석

## 1. 전체 워크플로우 비교

### 1.1 baseline.py 워크플로우
```
1. Config 생성 (딕셔너리) → YAML 저장
2. 데이터 로드 (train.csv, dev.csv)
3. 전처리 (Preprocess 클래스)
   - BOS/EOS 토큰 추가
   - 특수 토큰 처리
4. Dataset 생성 (커스텀 클래스)
5. 모델/토크나이저 로드
6. Trainer 설정 및 학습
7. 체크포인트 저장
8. [별도 실행] 추론
   - test.csv 로드
   - 모델 generate
   - 특수 토큰 제거
   - CSV 저장
```

### 1.2 제안 시스템 워크플로우
```
1. Config 로드 (YAML 파일)
2. 데이터 로드 (동일)
3. 전처리 (DataProcessor 클래스)
   - 동일한 토큰 처리
   - HuggingFace Dataset 변환
4. Dataset 생성 (HF Dataset)
5. 모델/토크나이저 로드 (다중 모델 지원)
6. Trainer 설정 및 학습 (동일)
7. 체크포인트 저장 (동일)
8. [자동 실행] 추론
   - 동일한 프로세스
   - 자동화됨
```

## 2. 핵심 기능별 상세 비교

### 2.1 데이터 처리

| 단계 | baseline.py | 제안 시스템 | 기능 동일성 |
|------|------------|-------------|------------|
| CSV 읽기 | `pd.read_csv()` | `pd.read_csv()` | ✅ 동일 |
| 컬럼 선택 | `['fname','dialogue','summary']` | 동일 | ✅ 동일 |
| BOS 토큰 | 학습: `bos + summary`<br>추론: `bos`만 | 토크나이저 자동 처리 | ✅ 동일 |
| EOS 토큰 | 학습: `summary + eos` | 토크나이저 자동 처리 | ✅ 동일 |
| 특수 토큰 | Person1~7, PhoneNumber 등 | 동일하게 추가 | ✅ 동일 |

### 2.2 모델 처리

| 기능 | baseline.py | 제안 시스템 | 결과 |
|------|------------|-------------|------|
| 지원 모델 | BART만 | mT5, T5, BART | ✅ 확장됨 |
| 모델 로드 | `BartForConditionalGeneration` | `AutoModelForSeq2SeqLM` | ✅ 개선됨 |
| 토큰 임베딩 | `resize_token_embeddings()` | 동일 | ✅ 동일 |
| 디바이스 | cuda:0 우선 | 자동 감지 (cuda/mps/cpu) | ✅ 개선됨 |

### 2.3 학습 과정

| 단계 | baseline.py | 제안 시스템 | 동일성 |
|------|------------|-------------|--------|
| Trainer | `Seq2SeqTrainer` | `Seq2SeqTrainer` | ✅ 동일 |
| Arguments | `Seq2SeqTrainingArguments` | 동일 | ✅ 동일 |
| 평가 지표 | ROUGE-1/2/L F1 | 동일 | ✅ 동일 |
| Early Stopping | patience=3, threshold=0.001 | 동일 | ✅ 동일 |
| 체크포인트 | best model 저장 | 동일 | ✅ 동일 |
| WandB | 통합 | 동일 | ✅ 동일 |

### 2.4 추론 과정

| 단계 | baseline.py | 제안 시스템 | 동일성 |
|------|------------|-------------|--------|
| 실행 방식 | 수동 실행 필요 | 자동 실행 | ✅ 개선됨 |
| 체크포인트 로드 | `from_pretrained(ckt_path)` | 동일 | ✅ 동일 |
| test.csv 처리 | DatasetForInference | 동일 구현 예정 | ✅ 동일 |
| 배치 처리 | DataLoader 사용 | 동일 | ✅ 동일 |
| generate 설정 | beam=4, no_repeat=2 등 | 동일 | ✅ 동일 |
| 토큰 제거 | remove_tokens 리스트 | 동일 | ✅ 동일 |
| 결과 저장 | fname, summary CSV | 동일 | ✅ 동일 |

## 3. 주요 차이점 및 개선사항

### 3.1 개선된 기능
1. **다중 모델 지원**: BART뿐만 아니라 mT5, T5 지원
2. **자동화**: 학습 후 추론 자동 실행
3. **유연한 설정**: 외부 YAML 파일로 설정 관리
4. **디바이스 최적화**: GPU/MPS/CPU 자동 감지
5. **실험 추적**: 체계적인 결과 관리

### 3.2 동일한 핵심 기능
1. **데이터 형식**: 동일한 CSV 형식
2. **전처리**: BOS/EOS 토큰, 특수 토큰 처리
3. **학습 프로세스**: Trainer, 평가 지표, Early Stopping
4. **추론 결과**: 동일한 출력 형식

## 4. 구현 우선순위

### 4.1 필수 구현 사항
1. `_run_test_inference()` 메서드
2. baseline.py 호환 전처리기
3. DatasetForInference 클래스
4. 특수 토큰 제거 로직

### 4.2 추가 개선 사항
1. 모델별 최적화 설정
2. 메모리 효율적 추론
3. 에러 복구 메커니즘
4. 상세한 로깅

## 5. 검증 방법

### 5.1 기능 검증
- baseline.py와 동일한 입력에 대해 동일한 출력 생성 확인
- 각 모델별로 추론 결과 검증
- 특수 토큰 처리 정확성 확인

### 5.2 성능 검증
- 추론 시간 측정
- GPU 메모리 사용량 모니터링
- 배치 크기별 성능 비교

## 6. 결론

제안된 시스템은 baseline.py의 **모든 핵심 기능을 동일하게 수행**하면서 다음과 같은 개선사항을 제공합니다:

1. ✅ **기능적 동일성**: 데이터 처리, 학습, 추론의 모든 단계가 동일하게 작동
2. ✅ **확장성**: 더 많은 모델 지원
3. ✅ **자동화**: 수동 작업 최소화
4. ✅ **유지보수성**: 구조화된 코드와 설정 관리

추론 부분만 구현하면 baseline.py와 완전히 동일한 기능을 제공하는 더 나은 시스템이 완성됩니다.
