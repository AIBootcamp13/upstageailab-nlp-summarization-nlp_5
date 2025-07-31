# 빔 서치 파라미터 최적화

## 개요
추론 시 빔 서치 파라미터를 최적화하여 생성 품질을 향상시킵니다. 다양한 디코딩 전략을 체계적으로 실험하여 최적의 파라미터 조합을 찾습니다.

## 실험 구성

### 1. 빔 크기 최적화 (08a_beam_size_sweep.yaml)
- **목적**: 최적의 num_beams 값 찾기
- **실험값**: 2, 4, 6, 8
- **평가 기준**:
  - ROUGE-F1 점수
  - 추론 시간
  - 메모리 사용량
  - 반복률

### 2. 길이 패널티 최적화 (08b_length_penalty.yaml)
- **목적**: 적절한 요약 길이를 위한 length_penalty 조정
- **실험값**: 0.5, 0.8, 1.0, 1.2, 1.5, 2.0
- **평가 기준**:
  - 압축 비율 (목표: 30%)
  - 내용 보존율
  - 길이 일관성

### 3. 반복 방지 최적화 (08c_no_repeat_ngram.yaml)
- **목적**: 반복 문제 해결을 위한 no_repeat_ngram_size 조정
- **실험값**: 0, 2, 3, 4, 5
- **평가 기준**:
  - N-gram 반복률
  - 어휘 다양성
  - 유창성 점수

### 4. Diverse Beam Search (08d_diverse_beam.yaml)
- **목적**: 다양한 생성을 위한 그룹 빔 서치
- **실험 조합**:
  - 그룹 수: 1, 2, 4
  - 다양성 패널티: 0.0, 0.5, 1.0, 2.0
- **평가 기준**:
  - Self-BLEU (다양성)
  - Best-of-N vs Ensemble 성능

## 실행 방법

### 1. 개별 실험 실행
```bash
# 빔 크기 실험
python experiments/run_beam_search_experiment.py \
    --config config/experiments/08_beam_search_optimization/08a_beam_size_sweep.yaml

# 길이 패널티 실험
python experiments/run_beam_search_experiment.py \
    --config config/experiments/08_beam_search_optimization/08b_length_penalty.yaml

# 반복 방지 실험
python experiments/run_beam_search_experiment.py \
    --config config/experiments/08_beam_search_optimization/08c_no_repeat_ngram.yaml

# Diverse Beam 실험
python experiments/run_beam_search_experiment.py \
    --config config/experiments/08_beam_search_optimization/08d_diverse_beam.yaml
```

### 2. 전체 실험 실행
```bash
#!/bin/bash
# run_all_beam_experiments.sh

for config in config/experiments/08_beam_search_optimization/*.yaml; do
    echo "실험 실행: $config"
    python experiments/run_beam_search_experiment.py --config $config
    sleep 10  # GPU 메모리 정리
done
```

### 3. 빠른 테스트 (작은 데이터셋)
```bash
# 테스트 모드로 실행 (subset 사용)
python experiments/run_beam_search_experiment.py \
    --config config/experiments/08_beam_search_optimization/08a_beam_size_sweep.yaml \
    --test-mode
```

## 결과 분석

### 분석 스크립트 실행
```bash
python scripts/analyze_beam_search.py
```

### 생성되는 결과물
`outputs/analysis/beam_search_optimization/` 디렉토리:

1. **beam_size_analysis.png**: 빔 크기별 성능/속도/메모리 분석
2. **length_penalty_analysis.png**: 길이 패널티 효과 분석
3. **no_repeat_analysis.png**: 반복 방지 효과 분석
4. **diverse_beam_analysis.png**: Diverse Beam Search 결과
5. **beam_search_report.md**: 종합 분석 보고서
6. **optimal_beam_search.yaml**: 최적 파라미터 설정

## 주요 메트릭

### 1. 품질 메트릭
- **ROUGE-F1**: 전반적인 요약 품질
- **BERTScore**: 의미적 유사도
- **Content Coverage**: 핵심 내용 포함율

### 2. 효율성 메트릭
- **Inference Time**: 배치당 추론 시간
- **Memory Usage**: GPU 메모리 사용량
- **Throughput**: 초당 처리 샘플 수

### 3. 다양성 메트릭
- **Self-BLEU**: 생성 간 유사도 (낮을수록 다양)
- **Distinct N-grams**: 고유 n-gram 비율
- **Lexical Diversity**: 어휘 다양성

### 4. 반복 메트릭
- **N-gram Repetition Rate**: n-gram 반복률
- **Longest Repeated Substring**: 최장 반복 부분 문자열

## 예상 결과

### 빔 크기
- **2**: 빠르지만 품질 낮음
- **4**: 속도/품질 균형 (권장)
- **6-8**: 높은 품질, 느린 속도

### 길이 패널티
- **< 1.0**: 짧은 요약 생성
- **1.0-1.5**: 적절한 길이 (권장)
- **> 1.5**: 긴 요약 생성

### 반복 방지
- **0**: 반복 허용 (자연스러움)
- **2-3**: 적절한 반복 방지 (권장)
- **4-5**: 과도한 제약 (부자연스러움)

### Diverse Beam
- 대부분의 경우 표준 빔 서치로 충분
- 다양성이 중요한 경우에만 사용

## 구현 세부사항

### 실험 실행기 수정
```python
# experiments/run_beam_search_experiment.py

def run_beam_experiment(config):
    """빔 서치 실험 실행"""
    
    # 모델 로드 (학습된 체크포인트 사용)
    model = load_trained_model(config['checkpoint_path'])
    
    # 각 설정에 대해 실험
    for exp_config in config['generation']['experiments']:
        print(f"실험: {exp_config['name']}")
        
        # 생성 설정 업데이트
        generation_config = GenerationConfig(**exp_config)
        
        # 추론 실행
        results = run_inference(
            model, 
            test_dataset,
            generation_config,
            measure_time=True,
            measure_memory=True
        )
        
        # 메트릭 계산
        metrics = compute_metrics(results)
        
        # 결과 저장
        save_results(exp_config['name'], metrics, results)
```

### 메트릭 계산 함수
```python
def compute_beam_search_metrics(predictions, references):
    """빔 서치 관련 메트릭 계산"""
    
    metrics = {}
    
    # 기본 ROUGE
    metrics.update(compute_rouge(predictions, references))
    
    # 반복 메트릭
    metrics.update(compute_repetition_metrics(predictions))
    
    # 길이 메트릭
    metrics.update(compute_length_metrics(predictions, references))
    
    # 다양성 메트릭
    if len(predictions[0]) > 1:  # 다중 생성인 경우
        metrics.update(compute_diversity_metrics(predictions))
    
    return metrics
```

## 최적화 팁

### 1. 메모리 관리
```yaml
# 큰 빔 크기 사용 시
generation:
  num_beams: 8
  # 메모리 절약 옵션
  use_cache: true
  output_scores: false  # 점수 저장 안 함
  return_dict_in_generate: false
```

### 2. 속도 최적화
```python
# 배치 추론
with torch.no_grad():
    outputs = model.generate(
        input_ids,
        **generation_config,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
```

### 3. 품질 vs 속도 트레이드오프
- **실시간 서비스**: beam_size=2-3, 간단한 설정
- **배치 처리**: beam_size=4-6, 최적화된 설정
- **최고 품질**: beam_size=8, diverse beam, 앙상블

## 문제 해결

### CUDA Out of Memory
```bash
# 배치 크기 줄이기
infernce:
  batch_size: 4  # 8에서 감소

# 또는 빔 크기 줄이기
generation:
  num_beams: 4  # 8에서 감소
```

### 반복 문제 지속
```yaml
# 더 강한 반복 방지
generation:
  no_repeat_ngram_size: 3
  repetition_penalty: 1.2  # 추가 패널티
  encoder_repetition_penalty: 1.0
```

### 길이 제어 실패
```yaml
# 더 세밀한 제어
generation:
  min_length: 30
  max_length: 180  # 200에서 감소
  length_penalty: 1.5
  early_stopping: false  # 최소 길이 보장
```

## 고급 기법

### 1. Constrained Beam Search
```python
# 특정 단어 포함 강제
force_words_ids = tokenizer(['중요한', '핵심'], add_special_tokens=False).input_ids

generation_config = GenerationConfig(
    num_beams=4,
    force_words_ids=force_words_ids
)
```

### 2. 동적 파라미터 조정
```python
# 입력 길이에 따른 조정
def get_dynamic_config(input_length):
    if input_length < 100:
        return {'length_penalty': 0.8, 'max_length': 50}
    elif input_length < 300:
        return {'length_penalty': 1.2, 'max_length': 100}
    else:
        return {'length_penalty': 1.5, 'max_length': 150}
```

### 3. 앙상블 전략
```python
# 다양한 설정으로 생성 후 선택
configs = [
    {'num_beams': 4, 'length_penalty': 1.0},
    {'num_beams': 4, 'length_penalty': 1.5},
    {'num_beams': 6, 'no_repeat_ngram_size': 3}
]

all_outputs = []
for config in configs:
    outputs = model.generate(**config)
    all_outputs.append(outputs)

# 최고 ROUGE 선택
best_output = select_best_by_rouge(all_outputs, reference)
```

## 실험 결과 예시

### 최적 설정 (일반적인 경우)
```yaml
generation:
  num_beams: 4
  max_length: 200
  min_length: 30
  length_penalty: 1.2
  no_repeat_ngram_size: 3
  early_stopping: true
  temperature: 1.0
  do_sample: false
```

### 성능 향상
- 베이스라인: ROUGE-F1 47.12%
- 최적화 후: ROUGE-F1 48.0-48.5% (+0.9%)
- 추론 시간: 1.2x (허용 범위)
- 반복률: 5% → 1% 감소

## 참고 자료

- [HuggingFace Generation](https://huggingface.co/docs/transformers/generation_strategies)
- [Beam Search Decoding](https://arxiv.org/abs/1702.01806)
- [Diverse Beam Search](https://arxiv.org/abs/1610.02424)
- [Length Control in NLG](https://arxiv.org/abs/1909.09483)
