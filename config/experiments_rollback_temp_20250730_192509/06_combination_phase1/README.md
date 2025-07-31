# 조합 실험 1차 - 최적 구성 찾기

## 실험 목적
지금까지 구현한 개선사항들 중 효과적인 것들을 조합하여 최적의 구성을 찾습니다. Ablation study 방식으로 각 구성요소의 기여도와 시너지 효과를 분석합니다.

## 실험 구성

### 1. 06a_aug_plus_post.yaml
- **조합**: 데이터 증강 + 후처리
- **목적**: 입력 데이터 품질 향상과 출력 품질 개선의 시너지
- **주요 설정**:
  - 동의어 치환 (15%) + 문장 재배열 (20%)
  - 중복 제거 + 길이 최적화 + PII 토큰 검증

### 2. 06b_aug_plus_lr.yaml
- **조합**: 데이터 증강 + 학습률 스케줄링
- **목적**: 데이터 다양성과 최적화 전략의 시너지
- **주요 설정**:
  - 동의어 치환 (15%) + 문장 재배열 (20%)
  - Cosine Annealing LR (3e-5 시작, warmup 10%)

### 3. 06c_all_simple.yaml
- **조합**: 모든 간단한 개선사항 통합
- **목적**: 전체 파이프라인 최적화
- **주요 설정**:
  - 텍스트 정규화 (전처리)
  - 데이터 증강 (학습)
  - Cosine Annealing LR (최적화)
  - 후처리 파이프라인 (추론)

## 실행 방법

### 1. 순차 실행 (권장)
```bash
# 실험 디렉토리로 이동
cd /path/to/nlp-sum-lyj

# 각 조합 실험 실행
python experiments/run_experiment.py --config config/experiments/06_combination_phase1/06a_aug_plus_post.yaml
python experiments/run_experiment.py --config config/experiments/06_combination_phase1/06b_aug_plus_lr.yaml
python experiments/run_experiment.py --config config/experiments/06_combination_phase1/06c_all_simple.yaml
```

### 2. 배치 실행 스크립트
```bash
#!/bin/bash
# run_combination_phase1.sh

EXPERIMENTS=(
    "06a_aug_plus_post"
    "06b_aug_plus_lr"
    "06c_all_simple"
)

for exp in "${EXPERIMENTS[@]}"; do
    echo "Running experiment: $exp"
    python experiments/run_experiment.py --config config/experiments/06_combination_phase1/${exp}.yaml
    
    # GPU 메모리 정리를 위한 대기
    sleep 30
done
```

### 3. 병렬 실행 (GPU 여러 개)
```bash
# 터미널 1 (GPU 0)
CUDA_VISIBLE_DEVICES=0 python experiments/run_experiment.py --config config/experiments/06_combination_phase1/06a_aug_plus_post.yaml

# 터미널 2 (GPU 1)
CUDA_VISIBLE_DEVICES=1 python experiments/run_experiment.py --config config/experiments/06_combination_phase1/06b_aug_plus_lr.yaml

# 터미널 3 (GPU 2)
CUDA_VISIBLE_DEVICES=2 python experiments/run_experiment.py --config config/experiments/06_combination_phase1/06c_all_simple.yaml
```

## 결과 분석

### 실험 완료 후 분석
```bash
python scripts/analyze_combination_phase1.py
```

### 분석 결과물
`outputs/analysis/combination_phase1/` 디렉토리에 생성:

1. **component_comparison.png**: 개별 구성요소 vs 조합 성능 비교
2. **synergy_analysis.png**: 시너지 효과 분석 (예상 vs 실제)
3. **efficiency_analysis.png**: 학습 시간 대비 성능 효율성
4. **ablation_study_table.png**: 구성요소별 Ablation Study
5. **phase1_combination_report.md**: 상세 분석 보고서
6. **06_best_combination.yaml**: 최적 조합 설정 파일

## 평가 기준

### 1. 성능
- 목표: ROUGE-F1 52% 이상
- 베이스라인 대비 향상폭

### 2. 시너지 효과
- 양의 시너지: 조합이 개별 효과의 합보다 좋음
- 음의 시너지: 구성요소 간 간섭 발생

### 3. 효율성
- 학습 시간 대비 성능 향상
- 추론 시간 및 메모리 사용량

## 예상 결과

1. **06a (증강+후처리)**: 
   - 예상: ~49-50%
   - 입출력 품질 동시 개선

2. **06b (증강+LR)**: 
   - 예상: ~48-49%
   - 학습 효율성 극대화

3. **06c (전체 통합)**: 
   - 예상: ~51-52%
   - 최고 성능, 긴 학습 시간

## 문제 해결

### OOM 에러
```yaml
# gradient_checkpointing 활성화
training:
  gradient_checkpointing: true
  
# 또는 배치 크기 줄이기
training:
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 8
```

### 느린 수렴
```yaml
# 학습률 조정
training:
  learning_rate: 5.0e-05  # 더 높게
  
# 또는 warmup 비율 증가
training:
  warmup_ratio: 0.15
```

### 과적합
```yaml
# dropout 증가 (모델별로 다름)
model:
  dropout: 0.2
  
# 또는 weight decay 증가
training:
  weight_decay: 0.05
```

## 다음 단계

1. **최적 조합 선정**: 
   - 가장 높은 성능의 조합을 06_best_combination.yaml로 저장
   
2. **2차 실험 준비**:
   - 최적 조합을 기반으로 고급 기능 추가
   - 특수 토큰 가중치, 빔 서치, 백트랜슬레이션 등

3. **추가 분석**:
   - 에러 분석: 어떤 유형의 요약에서 실패하는지
   - 정성적 평가: 생성된 요약의 품질 수동 검토

## 주의사항

1. **재현성**: 모든 실험은 seed=42로 고정
2. **공정한 비교**: 동일한 에폭 수와 조기 종료 설정
3. **메모리 관리**: 실험 간 GPU 메모리 정리 필수
4. **로깅**: WandB에 모든 메트릭 자동 기록

## 참고 자료

- 개별 구성요소 구현:
  - `code/data_augmentation/simple_augmentation.py`
  - `code/postprocessing/post_processor.py`
  - `code/preprocessing/text_normalizer.py`
- 분석 스크립트: `scripts/analyze_combination_phase1.py`
- 이전 실험 결과: `outputs/analysis/*/`
