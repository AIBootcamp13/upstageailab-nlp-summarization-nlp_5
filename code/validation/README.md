# 실험 실패 문제 해결 시스템

이 시스템은 NLP 대화 요약 실험에서 발생하는 토큰 인덱싱 오류와 메모리 부족 문제를 자동으로 감지하고 해결합니다.

## 🔍 해결된 문제들

### 1. 토큰 인덱싱 오류 (`srcIndex < srcSelectDimSize`)
- **원인**: 토크나이저가 생성하는 토큰 ID가 모델의 vocabulary 범위를 벗어남
- **해결책**: 
  - 토큰 범위 검증 시스템 구축
  - 안전한 시퀀스 길이로 자동 조정
  - 특수 토큰 호환성 검증

### 2. GPU 메모리 부족 오류 (`CUDA out of memory`)
- **원인**: 이전 실험의 GPU 메모리가 제대로 해제되지 않음
- **해결책**:
  - 실험간 완전한 GPU 메모리 정리
  - 메모리 요구사항 사전 추정
  - 배치 크기 자동 조정

## 🛠️ 구현된 모듈

### 1. 검증 시스템 (`code/validation/`)
```
code/validation/
├── __init__.py                  # 모듈 초기화
├── token_validation.py          # 토큰 호환성 검증
├── memory_validation.py         # 메모리 요구사항 검증
└── pre_experiment_check.py      # 실험 전 통합 검증
```

### 2. 핵심 기능

#### 토큰 검증 (`TokenValidator`)
```python
from validation.token_validation import validate_model_tokenizer_compatibility

# 모델-토크나이저 호환성 검증
result = validate_model_tokenizer_compatibility(model_name, config)
if not result["overall_valid"]:
    print("토큰 범위 문제 감지!")
```

#### 메모리 검증 (`MemoryValidator`)
```python
from validation.memory_validation import estimate_memory_requirements

# 메모리 요구사항 추정
memory_result = estimate_memory_requirements(config)
if not memory_result["memory_sufficient"]:
    print(f"메모리 부족: {memory_result['estimated_memory_gb']:.1f}GB 필요")
```

#### 실험 전 검증
```bash
# 자동 문제 해결과 함께 검증 실행
python3 code/validation/pre_experiment_check.py \
    --config config/experiments/baseline_kobart_rtx3090.yaml \
    --auto-fix \
    --cleanup
```

## 🔧 수정된 설정

### 1. KoBART 설정 안전화
```yaml
# config/experiments/baseline_kobart_rtx3090.yaml
tokenizer:
  encoder_max_len: 512  # 1280→512 (토큰 범위 문제 해결)
```

### 2. T5 설정 메모리 최적화
```yaml
# config/experiments/eenzeenee_t5_rtx3090.yaml
training:
  per_device_train_batch_size: 8   # 20→8 (메모리 부족 해결)
  per_device_eval_batch_size: 16   # 32→16
```

## 🚀 사용법

### 1. 개별 실험 검증
```bash
# 특정 실험 설정 검증
python3 code/validation/pre_experiment_check.py \
    --config config/experiments/baseline_kobart_rtx3090.yaml \
    --auto-fix
```

### 2. 전체 실험 실행 (자동 검증 포함)
```bash
# 검증이 통합된 실험 실행
bash run_main_5_experiments.sh
```

### 3. 수동 메모리 정리
```python
from validation.memory_validation import cleanup_between_experiments

# 실험간 GPU 메모리 완전 정리
success = cleanup_between_experiments()
```

## 📊 검증 결과 해석

### 성공적인 검증
```
🔍 실험 전 검증 결과
====================================
📝 토큰 호환성 검증:
   상태: ✅ 통과
   토크나이저 vocab: 30000
   모델 vocab: 30000

💾 메모리 요구사항 검증:
   상태: ✅ 충분
   예상 사용량: 8.5GB
   사용 가능: 23.7GB
   사용률: 35.9%

🎯 전체 검증 결과: ✅ 실험 실행 가능
```

### 문제 감지시
```
🔍 실험 전 검증 결과
====================================
📝 토큰 호환성 검증:
   상태: ❌ 실패
   권장사항:
     - 🚨 토큰 ID 범위 초과 감지!
     - 💡 특수 토큰을 모델 vocabulary에 추가 필요

💾 메모리 요구사항 검증:
   상태: ⚠️ 부족
   권장사항:
     - 🚨 메모리 부족: 5.2GB 초과
     - 💡 배치 크기를 20 → 8로 줄이세요

🎯 전체 검증 결과: ❌ 문제 해결 필요
🔧 설정이 자동으로 수정되었습니다
```

## 🎯 기대 효과

### 1. 실험 안정성 향상
- ✅ 토큰 범위 오류 방지
- ✅ 메모리 부족 오류 방지
- ✅ 실험 실패율 대폭 감소

### 2. 개발 효율성 증대
- ⚡ 자동 문제 감지 및 수정
- ⚡ 실험 재시작 횟수 감소
- ⚡ 디버깅 시간 단축

### 3. 리소스 최적화
- 🧹 실험간 완전한 메모리 정리
- 🧹 GPU 활용률 최적화
- 🧹 시스템 안정성 향상

## 📝 로그 및 디버깅

### 검증 로그 위치
```
validation_logs/
├── pre_experiment_check.log      # 실험 전 검증 로그
└── last_validation_result.json   # 마지막 검증 결과 (JSON)
```

### 상세 로그 확인
```bash
# 상세한 검증 로그 출력
python3 code/validation/pre_experiment_check.py \
    --config config/experiments/baseline_kobart_rtx3090.yaml \
    --verbose
```

## 🔄 문제 해결 플로우

1. **실험 시작** → 2. **자동 검증** → 3. **문제 감지** → 4. **자동 수정** → 5. **재검증** → 6. **실험 실행**

이 시스템을 통해 이제 모든 실험이 안정적으로 실행될 것입니다! 🎉
