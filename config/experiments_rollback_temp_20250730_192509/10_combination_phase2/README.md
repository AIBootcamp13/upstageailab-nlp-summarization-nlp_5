# 조합 실험 2차 - 고급 기능 통합

## 개요
1차 조합 실험의 최적 구성에 고급 기능들을 추가하여 최종 성능을 극대화합니다.

## 실험 구성

### 10a_phase1_plus_token_weight.yaml
- **목적**: 1차 최적 구성 + 특수 토큰 가중치
- **주요 특징**:
  - PII 토큰 가중치: 2.5
  - 화자 토큰 가중치: 2.0
  - 레이블 스무딩: 0.1
  - 정적 가중치 사용
- **예상 효과**: 
  - 특수 토큰 재현율 향상
  - PII 정보 보존 개선
  - ROUGE 점수 +1-2% 향상 예상

### 10b_phase1_plus_backtrans.yaml
- **목적**: 1차 최적 구성 + 백트랜슬레이션
- **주요 특징**:
  - Google Translate API 사용
  - 한국어→영어→한국어 변환
  - 품질 임계값: 0.7
  - 증강 비율: 40%
- **예상 효과**:
  - 데이터 다양성 증가
  - 과적합 방지
  - ROUGE 점수 +2-3% 향상 예상

### 10c_all_optimizations.yaml
- **목적**: 모든 최적화 기법 통합
- **주요 특징**:
  - 동적 토큰 가중치
  - 다중 언어 백트랜슬레이션 (영어, 일본어)
  - Diverse Beam Search
  - 강화된 후처리
  - 25 에폭 학습
- **예상 효과**:
  - 최고 성능 달성
  - ROUGE-F1 55% 목표
  - 학습 시간 증가 (약 8-10시간)

## 실행 방법

### 개별 실험 실행
```bash
# 토큰 가중치 실험
python code/auto_experiment_runner.py --config config/experiments/10_combination_phase2/10a_phase1_plus_token_weight.yaml

# 백트랜슬레이션 실험
python code/auto_experiment_runner.py --config config/experiments/10_combination_phase2/10b_phase1_plus_backtrans.yaml

# 전체 통합 실험
python code/auto_experiment_runner.py --config config/experiments/10_combination_phase2/10c_all_optimizations.yaml
```

### 순차 실행
```bash
# 모든 2차 조합 실험 순차 실행
./run_phase2_experiments.sh
```

## 성능 분석

### 주요 메트릭
- ROUGE-1/2/L F1 점수
- 특수 토큰 재현율
- 학습/추론 시간
- 메모리 사용량

### 분석 스크립트
```bash
python scripts/analyze_combination_phase2.py
```

## 최종 선택 기준

1. **성능**: ROUGE-F1 점수가 가장 높은 구성
2. **효율성**: 학습/추론 시간 대비 성능 향상
3. **안정성**: 검증 세트에서의 일관된 성능
4. **실용성**: 실제 배포 환경에서의 사용 가능성

## 주의사항

- 백트랜슬레이션 사용 시 Google Translate API 키 필요
- 전체 통합 실험은 GPU 메모리 16GB 이상 권장
- 캐시 디렉토리 용량 확인 (약 5GB 필요)

## 예상 결과

| 실험 | 예상 ROUGE-F1 | 학습 시간 | 메모리 |
|------|--------------|-----------|--------|
| 10a (토큰 가중치) | 52-53% | 4-5시간 | 12GB |
| 10b (백트랜슬레이션) | 53-54% | 6-7시간 | 14GB |
| 10c (전체 통합) | 54-56% | 8-10시간 | 16GB |

## 다음 단계

1. 실험 결과 분석
2. 최적 구성 선택
3. 최종 모델 학습
4. 테스트 세트 평가
5. 제출 준비
