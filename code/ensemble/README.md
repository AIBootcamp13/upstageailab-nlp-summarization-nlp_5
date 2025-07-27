# Solar API 앙상블 시스템
## 개요
미세 조정된 모델과 Solar API를 결합하여 최고 성능을 달성하는 앙상블 시스템입니다.


## 주요 특징 (안정성 강화 버전)

### 1. 가중치 기반 앙상블
- **정적 가중치**: 고정된 비율로 결합 (기본 0.7:0.3)
- **동적 가중치**: 입력 특성에 따라 가중치 자동 조정
  - 특수 토큰 보존도
  - 요약 길이 균형
  - 신뢰도 점수

### 2. 안정성 강화된 Solar API 연동
- **오류 처리 강화**:
  - 지수 백오프 재시도 메커니즘 (5회 시도, 최대 60초 대기)
  - 타임아웃 오류 시 자동 타임아웃 증가 (45→120초)
  - API 키 검증 및 연결 테스트
- **폴백 메커니즘**:
  - Solar API 실패 시 Fine-tuned 모델로 자동 전환
  - 부분 실패 시 완료된 결과 보존 및 재시작 지원
  - 신뢰도 기반 결과 검증
- **비용 최적화**:
  - 지능형 캐싱 메커니즘 (동일 입력 재요청 방지)
  - 동적 Rate limiting (80요청/분, 보수적 조정)
  - 연속 실패 모니터링 및 자동 비활성화
- **모니터링 및 로깅**:
  - API 호출 내역 상세 추적
  - 비용 추정 및 누적 계산
  - 성능 메트릭 모니터링

### 환경 설정
```bash
# API 키 설정 (필수)
export UPSTAGE_API_KEY="your-api-key-here"

# Python 환경
conda activate nlp-sum
```

### 기본 실행
```bash
# 동적 가중치 모드 (권장)
./run_solar_ensemble.sh

# 모든 모드 비교
./run_solar_ensemble.sh all

# 테스트 (100개 샘플)
./run_solar_ensemble.sh test
```

### 직접 실행
```python
from code.ensemble.solar_ensemble import WeightedEnsemble, EnsembleConfig

# 설정
config = EnsembleConfig(
    solar_api_key="your-key",
    fine_tuned_weight=0.7,
    solar_weight=0.3,
    dynamic_weights=True
)

# 앙상블 생성
ensemble = WeightedEnsemble(
    fine_tuned_model_path="path/to/model",
    ensemble_config=config
)

# 처리
# 처리
result = ensemble.process_single("대화 내용...")
print(result.ensemble_summary)
```

## 앙상블 전략

### 1. 정적 가중치
- 고정된 비율로 결합
### 2. 동적 가중치
- 입력에 따라 가중치 조정
- 특수 토큰이 많으면 Fine-tuned 가중치 증가
- 입력에 따라 가중치 조정
- 특수 토큰이 많으면 Fine-tuned 가중치 증가
- 복잡한 대화는 Solar 가중치 증가

### 3. 신뢰도 기반 선택
- 두 모델의 일치도가 높으면 신뢰도 상승
- 신뢰도가 낮으면 Fine-tuned 모델 우선

## 성능 분석

### 예상 성능
| 모드 | ROUGE-F1 | 처리 시간 | API 비용 |
|-----|----------|-----------|----------|
| Fine-tuned only | 54-55% | 빠름 | 없음 |
| Static ensemble | 56-57% | 중간 | 중간 |
| Dynamic ensemble | 57-58% | 중간 | 중간 |
| Solar dominant | 55-56% | 느림 | 높음 |

### 비용 분석
- API 호출당 약 $0.001 (예시)
- 전체 테스트셋: 약 $5-10
- 캐싱으로 20-30% 절감 가능

## 트러블슈팅

### 1. API 키 오류
```bash
ERROR: UPSTAGE_API_KEY 환경 변수가 설정되지 않았습니다.
```
해결: `export UPSTAGE_API_KEY="your-key"`

### 2. Rate Limit 초과
```
Rate limit reached. Sleeping for X seconds...
```
정상 동작입니다. 자동으로 대기 후 재시도합니다.

### 3. 메모리 부족
- batch_size 감소 (기본 8 → 4)
- use_async를 false로 설정

### 4. API 타임아웃
- timeout 증가 (기본 30초 → 60초)
- max_retries 증가 (기본 3 → 5)

## 최적화 팁

### 1. 캐싱 활용
- 개발 중에는 항상 캐싱 활성화
- 캐시 디렉토리 정기적 정리

### 2. 배치 크기 조정
- GPU 메모리에 따라 조정
- API rate limit 고려

### 3. Few-shot 예제
- 고품질 예제 3-5개 선택
- 도메인별 예제 준비

### 4. 후처리 최적화
- 특수 토큰 검증 필수
- 길이 최적화 신중히

## 평가 메트릭

### 주요 지표
1. **ROUGE 점수**: 전체적인 품질
2. **특수 토큰 재현율**: PII 정보 보존
3. **신뢰도 점수**: 예측 신뢰성
4. **개선율**: Fine-tuned 대비 향상

### 분석 도구
```bash
# 결과 분석
python scripts/analyze_solar_ensemble.py

# 비용 분석
python scripts/calculate_api_costs.py
| 정적 앙상블 | 56-57% | 중간 | 중간 |

## 제출 준비

1. **최고 성능 모드 선택**
   - 보통 dynamic_weights가 최고 성능

2. **전체 테스트셋 처리**
   ```bash
   ./run_solar_ensemble.sh
   ```

3. **제출 파일 확인**
   - 위치: `outputs/solar_ensemble/*/submission.csv`
   - 형식 검증 필수

4. **백업**
   - 모든 구성요소 요약 저장
   - API 호출 로그 보관

## 주의사항

1. **API 비용**
   - 실행 전 예상 비용 계산
   - 비용 제한 설정 확인

2. **처리 시간**
   - 전체 테스트셋: 2-3시간
   - Rate limit으로 인한 대기 시간 포함

3. **에러 처리**
   - API 실패 시 자동 폴백
   - 부분 실패 시 재실행 가능

4. **보안**
   - API 키를 코드에 하드코딩하지 마세요
   - 환경 변수 사용 필수

## 고급 설정

### 프롬프트 커스터마이징
```python
def custom_prompt(dialogue):
    return [{
        "role": "system",
        "content": "당신은 한국어 대화 요약 전문가입니다."
    }, {
        "role": "user", 
        "content": f"다음 대화를 요약하세요:\n{dialogue}"
    }]
```

### 가중치 조정 함수
```python
def custom_weight_function(features):
    if features['special_tokens'] > 5:
        return {'fine_tuned': 0.8, 'solar': 0.2}
    else:
        return {'fine_tuned': 0.6, 'solar': 0.4}
```

### 후처리 파이프라인
```python
def postprocess_ensemble(summary):
    # 특수 토큰 검증
    summary = validate_special_tokens(summary)
    # 길이 조정
    summary = optimize_length(summary)
    return summary
```

## 실험 결과 예시

### Validation Set 성능
```
Mode: dynamic_weights
ROUGE-1: 0.5821
ROUGE-2: 0.3456  
ROUGE-L: 0.5234
Average: 0.5504
Improvement: +3.2%
```

### API 사용 통계
```
Total API calls: 500
Cache hits: 150 (30%)
Average latency: 0.8s
Total cost: $0.35
```

### 신뢰도 분포
```
High confidence (>0.8): 65%
Medium confidence (0.6-0.8): 25%
Low confidence (<0.6): 10%
```

## 향후 개선 방향

1. **멀티 모델 앙상블**
   - Solar API 외 다른 LLM 추가
   - 3개 이상 모델 투표

2. **적응형 가중치**
   - 온라인 학습으로 가중치 최적화
   - 도메인별 가중치 프로파일

3. **프롬프트 자동 최적화**
   - 베이지안 최적화
   - 진화 알고리즘

4. **비용 최적화**
   - 선택적 API 호출
   - 신뢰도 기반 스킵

## 문제 해결 가이드

### Q: API 응답이 너무 느립니다
A: 
- 배치 크기를 줄이세요 (8 → 4)
- 비동기 모드가 활성화되어 있는지 확인
- 타임아웃을 늘리되, 재시도 횟수는 줄이세요

### Q: 캐시가 작동하지 않습니다
A: 
- 캐시 디렉토리 권한 확인
- 캐시 키 생성 로직 확인
- TTL이 만료되지 않았는지 확인

### Q: 메모리 오류가 발생합니다
A:
- GPU 메모리: 모델을 CPU로 이동
- RAM: 배치 크기 감소
- 캐시 크기 제한 설정

### Q: 특수 토큰이 누락됩니다
A:
- 후처리 파이프라인 확인
- 프롬프트에 명시적 지시 추가
- Fine-tuned 모델 가중치 증가

## 결론

Solar API 앙상블은 Fine-tuned 모델의 도메인 특화 능력과 Solar API의 일반화 능력을 결합하여 최고의 성능을 달성할 수 있습니다. 적절한 가중치 조정과 후처리를 통해 ROUGE-F1 57-58% 달성이 가능합니다.

---

마지막 업데이트: 2025-01-27
