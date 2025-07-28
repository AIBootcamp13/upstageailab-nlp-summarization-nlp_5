# 빠른 검증 테스트 가이드

1에포크 빠른 테스트로 전체 파이프라인이 에러 없이 동작하는지 검증할 수 있습니다.

## 🚀 빠른 테스트 방법들

### 1. 범용 빠른 테스트 스크립트

```bash
# 기본 사용 (eenzeenee 모델)
python quick_test.py

# 특정 모델 테스트
python quick_test.py --model-section eenzeenee
python quick_test.py --model-section xlsum_mt5
python quick_test.py --model-section baseline

# 샘플 수 조정
python quick_test.py --max-samples 50

# 상세 로깅
python quick_test.py --verbose
```

### 2. 셸 스크립트 사용

```bash
# 간편한 실행
./run_quick_test.sh

# 특정 모델 테스트
./run_quick_test.sh --model-section eenzeenee

# 다양한 옵션
./run_quick_test.sh --max-samples 30 --verbose
```

### 3. 모델별 전용 스크립트

```bash
# eenzeenee 모델 빠른 테스트
./run_eenzeenee_experiment.sh --quick-test
EENZEENEE_QUICK_TEST=true EENZEENEE_RUN_ACTUAL=true ./run_eenzeenee_experiment.sh

# 다른 모델들도 비슷하게 지원 예정
```

### 4. 통합 테스트 (모든 모델)

```bash
# 모든 모델 빠른 테스트
./run_all_quick_tests.sh --all

# 특정 모델만
./run_all_quick_tests.sh --model eenzeenee

# 상세 로깅
./run_all_quick_tests.sh --all --verbose
```

## ⚡ 빠른 테스트 특징

### 자동 설정 조정
- **에포크**: 1에포크만 실행
- **샘플 수**: 기본 50-100개 (조정 가능)
- **배치 크기**: 2로 축소 (메모리 절약)
- **입력 길이**: 256토큰으로 단축
- **출력 길이**: 64토큰으로 단축
- **평가 빈도**: 더 자주 (빠른 피드백)

### 시간 단축
- **일반 훈련**: 30분-2시간
- **빠른 테스트**: 5-15분
- **목적**: 에러 검증, 파이프라인 동작 확인

## 📋 사용 시나리오

### 1. 새로운 환경 설정 후
```bash
# 모든 의존성이 제대로 설치되었는지 확인
./run_all_quick_tests.sh --all
```

### 2. 코드 수정 후 검증
```bash
# 특정 모델만 빠르게 테스트
python quick_test.py --model-section eenzeenee --max-samples 30
```

### 3. GPU 메모리 확인
```bash
# 작은 배치로 메모리 사용량 확인
python quick_test.py --max-samples 20
```

### 4. 새로운 모델 추가 후
```bash
# 새 모델 설정이 올바른지 확인
python quick_test.py --model-section new_model --verbose
```

## 🛠️ 고급 사용법

### 커스텀 설정으로 테스트
```bash
# 특정 설정 파일 사용
python quick_test.py --config custom_config.yaml

# 더 많은 샘플로 테스트
python quick_test.py --max-samples 200 --model-section xlsum_mt5
```

### 결과 분석
```bash
# 상세 로그 보기
python quick_test.py --verbose 2>&1 | tee quick_test.log

# JSON 결과 확인
# 결과는 outputs/quick_test_*/ 디렉토리에 저장됨
```

## 🔧 트러블슈팅

### 메모리 부족 에러
```bash
# 더 작은 배치와 샘플 수 사용
python quick_test.py --max-samples 20
```

### 모델 로딩 실패
```bash
# 상세 로그로 문제 확인
python quick_test.py --verbose --model-section problematic_model
```

### 의존성 문제
```bash
# requirements.txt 재설치
pip install -r requirements.txt

# 그 후 빠른 테스트
python quick_test.py
```

## 📊 기대 결과

### 성공적인 빠른 테스트
```
✅ 빠른 테스트 성공!
📊 모델: eenzeenee/t5-base-korean-summarization
📊 훈련 샘플: 50
📊 평가 샘플: 12
🎯 ROUGE-L: 0.3456
```

### 전체 파이프라인 검증 완료
- 모델 로딩 ✅
- 데이터 로딩 ✅  
- 토크나이징 ✅
- 1에포크 훈련 ✅
- 평가 메트릭 계산 ✅
- 결과 저장 ✅

## 💡 팁

1. **첫 실행**: 항상 빠른 테스트로 시작
2. **코드 변경**: 수정 후 즉시 빠른 테스트
3. **실험 설계**: 전체 실험 전 빠른 테스트로 검증
4. **디버깅**: 에러 발생시 빠른 테스트로 문제 격리
5. **성능 확인**: ROUGE 점수가 0.2 이상이면 정상

이제 전체 훈련을 실행하기 전에 빠르게 파이프라인을 검증할 수 있습니다!
