# 🧮 알고리즘

프로젝트에서 사용된 핵심 알고리즘과 기법에 대한 상세 설명입니다.

## 📋 포함 문서

### 🌟 [Solar API 통합](./solar_api_integration.md)
- Solar API와 Fine-tuned 모델의 앙상블 기법
- 동적 가중치 조정 알고리즘
- 신뢰도 기반 결과 선택 메커니즘
- 비용 최적화 및 성능 분석

## 🎯 핵심 알고리즘

### 1. 앙상블 기법
#### 정적 가중치 앙상블
```python
# 고정 비율 결합 (예: 70:30)
ensemble_result = (
    fine_tuned_weight * fine_tuned_output + 
    solar_weight * solar_output
)
```

#### 동적 가중치 앙상블
```python
# 입력 특성에 따른 가중치 조정
def calculate_dynamic_weights(input_features):
    special_token_count = count_special_tokens(input_features)
    dialogue_complexity = analyze_complexity(input_features)
    
    if special_token_count > threshold:
        return {"fine_tuned": 0.8, "solar": 0.2}
    elif dialogue_complexity > threshold:
        return {"fine_tuned": 0.6, "solar": 0.4}
    else:
        return {"fine_tuned": 0.7, "solar": 0.3}
```

### 2. 특수 토큰 가중치 조정
#### TokenWeightedCrossEntropy
- **PII 토큰**: 2.5배 가중치
- **화자 토큰**: 2.0배 가중치
- **일반 토큰**: 1.0배 기본 가중치

#### 동적 가중치 스케줄링
```python
def dynamic_weight_schedule(epoch, total_epochs):
    if epoch < total_epochs * 0.3:  # 워밍업
        return min(3.0, epoch / (total_epochs * 0.3) * 3.0)
    elif epoch < total_epochs * 0.8:  # 감소
        return max(1.5, 3.0 - (epoch - total_epochs * 0.3) / (total_epochs * 0.5) * 1.5)
    else:  # 안정화
        return 1.5
```

### 3. 후처리 파이프라인
#### 규칙 기반 후처리
1. **중복 제거**: 연속된 동일 문장 제거
2. **길이 최적화**: 최적 요약 길이 조정
3. **특수 토큰 검증**: PII/화자 정보 보존 확인
4. **문법 교정**: 기본적인 문법 오류 수정

#### 품질 검증 알고리즘
```python
def validate_summary_quality(summary, original_dialogue):
    scores = {
        "special_token_preservation": check_special_tokens(summary, original_dialogue),
        "length_optimization": check_length_constraints(summary),
        "coherence_score": calculate_coherence(summary),
        "factual_consistency": check_factual_consistency(summary, original_dialogue)
    }
    return weighted_average(scores)
```

### 4. 데이터 증강 알고리즘
#### 동의어 치환 (SynonymReplacement)
- WordNet 기반 동의어 사전 활용
- 문맥 고려한 적절한 동의어 선택
- 특수 토큰 보존

#### 문장 순서 변경 (SentenceReorder)
- 화자 순서 보존하며 대화 재배열
- 의미적 일관성 유지
- 대화 흐름 자연스러움 보장

#### 역번역 (BackTranslation)
- 한국어 → 영어 → 한국어 변환
- Google Translate API 활용
- 의미 보존하며 표현 다양성 증대

## 📊 성능 최적화 기법

### 1. 빔 서치 개선
- **Diverse Beam Search**: 5개 그룹으로 다양성 확보
- **Length Penalty**: 1.2로 조정하여 적절한 길이 유도
- **No Repeat N-gram**: 2-gram 반복 방지

### 2. 학습률 스케줄링
- **Cosine Annealing with Warm Restarts**
- **초기 학습률**: 5e-5 (베이스라인 대비 증가)
- **워밍업**: 전체 학습의 10%

### 3. 메모리 최적화
- **Gradient Checkpointing**: 메모리 사용량 50% 감소
- **Mixed Precision (FP16)**: 학습 속도 2배 향상
- **Dynamic Batching**: GPU 메모리 효율 극대화

## 🔍 평가 알고리즘

### Multi-Reference ROUGE
```python
def compute_multi_reference_rouge(predictions, references_list):
    rouge_scores = {}
    for i, pred in enumerate(predictions):
        max_scores = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
        
        for ref in references_list[i]:
            scores = calculate_rouge(pred, ref)
            for metric in max_scores:
                max_scores[metric] = max(max_scores[metric], scores[metric])
        
        rouge_scores[i] = max_scores
    
    return aggregate_scores(rouge_scores)
```

### 특수 토큰 메트릭
- **재현율**: 원본 대화의 특수 토큰 보존 비율
- **정밀도**: 생성된 요약의 특수 토큰 정확도
- **F1 점수**: 재현율과 정밀도의 조화 평균

## 🔗 관련 문서

- **구현 세부사항**: [API 참조](../api_reference/README.md)
- **시스템 구조**: [아키텍처](../architecture/README.md)
- **실험 결과**: [성능 분석](../../04_experiments/README.md)

---
📍 **위치**: `docs/03_technical_docs/algorithms/`
