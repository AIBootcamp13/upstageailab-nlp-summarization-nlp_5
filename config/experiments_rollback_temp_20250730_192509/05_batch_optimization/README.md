# 배치 크기 및 Gradient Accumulation 최적화 실험

## 실험 목적
메모리 효율과 학습 성능의 최적점을 찾기 위해 다양한 배치 크기와 gradient accumulation 조합을 실험합니다.

## 실험 설정
효과적인 배치 크기를 64로 고정하고 3가지 구성을 테스트:

### 1. 05a_small_batch_high_accum.yaml
- **배치 크기**: 8
- **Gradient Accumulation**: 8
- **효과적 배치 크기**: 64
- **특징**: 메모리 사용량 최소화, 안정적인 학습

### 2. 05b_medium_batch_medium_accum.yaml  
- **배치 크기**: 16
- **Gradient Accumulation**: 4
- **효과적 배치 크기**: 64
- **특징**: 균형잡힌 구성, 적당한 속도와 메모리 사용

### 3. 05c_large_batch_no_accum.yaml
- **배치 크기**: 64
- **Gradient Accumulation**: 1
- **효과적 배치 크기**: 64
- **특징**: 최고 속도, 높은 메모리 사용량

## 실행 방법

### 1. 개별 실험 실행
```bash
# 작은 배치 + 높은 accumulation
python experiments/run_experiment.py --config config/experiments/05_batch_optimization/05a_small_batch_high_accum.yaml

# 중간 배치 + 중간 accumulation  
python experiments/run_experiment.py --config config/experiments/05_batch_optimization/05b_medium_batch_medium_accum.yaml

# 큰 배치 + accumulation 없음
python experiments/run_experiment.py --config config/experiments/05_batch_optimization/05c_large_batch_no_accum.yaml
```

### 2. 모든 실험 순차 실행
```bash
# 배치 실행 스크립트
for config in config/experiments/05_batch_optimization/*.yaml; do
    echo "Running experiment: $config"
    python experiments/run_experiment.py --config $config
done
```

### 3. 병렬 실행 (GPU가 여러 개인 경우)
```bash
# GPU 0에서 실행
CUDA_VISIBLE_DEVICES=0 python experiments/run_experiment.py --config config/experiments/05_batch_optimization/05a_small_batch_high_accum.yaml &

# GPU 1에서 실행
CUDA_VISIBLE_DEVICES=1 python experiments/run_experiment.py --config config/experiments/05_batch_optimization/05b_medium_batch_medium_accum.yaml &

# GPU 2에서 실행
CUDA_VISIBLE_DEVICES=2 python experiments/run_experiment.py --config config/experiments/05_batch_optimization/05c_large_batch_no_accum.yaml &
```

## 결과 분석

실험 완료 후 분석 스크립트 실행:
```bash
python scripts/analyze_batch_optimization.py
```

분석 결과는 `outputs/analysis/batch_optimization/` 디렉토리에 저장됩니다:
- `memory_vs_speed.png`: 메모리 사용량 대비 학습 속도 분석
- `training_curves.png`: 학습 곡선 비교
- `summary_table.png`: 종합 비교 테이블
- `batch_optimization_report.md`: 상세 분석 보고서
- `batch_optimization_results.csv`: 결과 데이터

## 주요 모니터링 메트릭

1. **GPU 메모리 사용량**
   - 평균 및 최대 메모리 사용량
   - OOM 에러 발생 여부

2. **학습 속도**
   - Steps/second
   - 전체 학습 시간

3. **성능**
   - ROUGE-F1 점수
   - 수렴 속도

4. **효율성**
   - 분당 ROUGE 향상율
   - 메모리 대비 성능

## 예상 결과

- **05a (작은 배치)**: 안정적이지만 느림, 최고 성능 가능성
- **05b (중간 배치)**: 균형잡힌 성능, 개발에 적합
- **05c (큰 배치)**: 빠른 학습, 메모리 사용량 높음

## 문제 해결

### OOM (Out of Memory) 에러
```bash
# gradient_checkpointing 활성화
# config 파일에서 gradient_checkpointing: true 설정

# 또는 배치 크기 줄이기
per_device_train_batch_size: 32  # 64에서 32로
```

### 느린 학습 속도
```bash
# 데이터로더 워커 수 증가
dataloader_num_workers: 8  # 4에서 8로

# 또는 gradient accumulation 줄이기
gradient_accumulation_steps: 2  # 4에서 2로
```

## 다음 단계

최적의 배치 구성을 찾은 후:
1. 선택된 구성을 후속 실험의 기본값으로 사용
2. 더 긴 학습 에폭으로 최종 성능 확인
3. 다른 하이퍼파라미터와 조합하여 추가 최적화
