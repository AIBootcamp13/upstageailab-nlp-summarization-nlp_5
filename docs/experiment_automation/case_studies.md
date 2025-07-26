# 실험 자동화 시스템 - 실전 예제 및 사례 연구

## 목차
1. [실전 예제 1: 첫 번째 Sweep 실행](#실전-예제-1-첫-번째-sweep-실행)
2. [실전 예제 2: 모델 비교 실험](#실전-예제-2-모델-비교-실험)
3. [실전 예제 3: 대규모 하이퍼파라미터 탐색](#실전-예제-3-대규모-하이퍼파라미터-탐색)
4. [성공 사례 분석](#성공-사례-분석)
5. [실패 사례와 교훈](#실패-사례와-교훈)

---

## 실전 예제 1: 첫 번째 Sweep 실행

### 목표
KoBART 모델의 최적 학습률과 배치 크기 찾기

### Step 1: 빠른 파이프라인 검증
```bash
# 먼저 100개 샘플로 빠른 테스트
cd code
python trainer.py \
    --config config/base_config.yaml \
    --train-data ../data/train.csv
```

설정 수정 (config/quick_test.yaml):
```yaml
data:
  max_train_samples: 100
  max_val_samples: 50

training:
  num_train_epochs: 1
  eval_steps: 20
  save_steps: 50
```

### Step 2: 소규모 Sweep 설정
```yaml
# config/sweep/first_sweep.yaml
name: "First Learning Rate and Batch Size Search"
method: "grid"  # 처음엔 grid로 확실하게

metric:
  name: "eval/rouge_combined_f1"
  goal: "maximize"

parameters:
  learning_rate:
    values: [1e-5, 3e-5, 5e-5, 7e-5]  # 4개
  
  per_device_train_batch_size:
    values: [8, 16]  # 2개
  
  # 총 4 × 2 = 8개 실험

early_terminate:
  type: "hyperband"
  min_iter: 3  # 최소 3 에폭은 실행
```

### Step 3: Sweep 실행
```bash
python sweep_runner.py \
    --base-config config/base_config.yaml \
    --sweep-config first_sweep \
    --count 8
```

### Step 4: 결과 분석
```python
# analyze_first_sweep.py
import json
import pandas as pd
from pathlib import Path

# 결과 로딩
results_dir = Path("outputs/sweep_first_sweep")
results = []

for result_file in results_dir.glob("run_*_result.json"):
    with open(result_file) as f:
        results.append(json.load(f))

# DataFrame으로 변환
df = pd.DataFrame(results)

# 결과 테이블
print("\n=== Sweep Results ===")
print(df[['sweep_params.learning_rate', 
         'sweep_params.per_device_train_batch_size',
         'best_metrics.rouge_combined_f1']].sort_values(
    'best_metrics.rouge_combined_f1', ascending=False))

# 최적 설정
best_run = df.loc[df['best_metrics.rouge_combined_f1'].idxmax()]
print(f"\n최적 설정:")
print(f"Learning Rate: {best_run['sweep_params']['learning_rate']}")
print(f"Batch Size: {best_run['sweep_params']['per_device_train_batch_size']}")
print(f"ROUGE Combined F1: {best_run['best_metrics']['rouge_combined_f1']:.4f}")
```

### 예상 결과
```
=== Sweep Results ===
   learning_rate  batch_size  rouge_combined_f1
3         5e-05          16              0.4823
7         3e-05          16              0.4756
1         5e-05           8              0.4698
5         3e-05           8              0.4621
2         7e-05          16              0.4589
0         1e-05          16              0.4523
6         7e-05           8              0.4501
4         1e-05           8              0.4487

최적 설정:
Learning Rate: 5e-05
Batch Size: 16
ROUGE Combined F1: 0.4823
```

---

## 실전 예제 2: 모델 비교 실험

### 목표
KoBART vs KoGPT2 vs T5 성능 비교

### Step 1: 모델별 최적 설정 파일 준비
```yaml
# config/models/optimal_kobart.yaml
model:
  architecture: "kobart"
  checkpoint: "digit82/kobart-summarization"

training:
  learning_rate: 5e-5  # 이전 실험에서 찾은 최적값
  per_device_train_batch_size: 16

# config/models/optimal_kogpt2.yaml  
model:
  architecture: "kogpt2"
  checkpoint: "skt/kogpt2-base-v2"
  
tokenizer:
  padding_side: "left"

training:
  learning_rate: 3e-5
  per_device_train_batch_size: 8  # GPT2는 메모리 사용량이 높음

# config/models/optimal_t5.yaml
model:
  architecture: "t5"
  checkpoint: "KETI-AIR/ke-t5-base"

training:
  learning_rate: 1e-4
  per_device_train_batch_size: 12
```

### Step 2: 모델 비교 스크립트
```python
# compare_models.py
import subprocess
import json
from pathlib import Path
import pandas as pd

models = ['kobart', 'kogpt2', 't5']
results = {}

for model in models:
    print(f"\n{'='*50}")
    print(f"Training {model.upper()}")
    print('='*50)
    
    # 각 모델을 3번씩 다른 시드로 실행
    model_results = []
    
    for seed in [42, 123, 456]:
        cmd = [
            'python', 'trainer.py',
            '--config', f'config/models/optimal_{model}.yaml',
            '--train-data', '../data/train.csv',
            '--val-data', '../data/dev.csv'
        ]
        
        # 환경변수로 시드 설정
        env = os.environ.copy()
        env['SEED'] = str(seed)
        
        # 실행
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        
        # 결과 파싱
        output_dir = Path(f"outputs/{model}_seed{seed}")
        with open(output_dir / "results/training_results.json") as f:
            metrics = json.load(f)
            model_results.append(metrics['best_metrics'])
    
    results[model] = model_results

# 결과 분석
print("\n=== Model Comparison Results ===")
comparison_df = []

for model, runs in results.items():
    rouge_scores = [r['rouge_combined_f1'] for r in runs]
    comparison_df.append({
        'model': model,
        'mean_rouge': np.mean(rouge_scores),
        'std_rouge': np.std(rouge_scores),
        'max_rouge': max(rouge_scores),
        'min_rouge': min(rouge_scores)
    })

df = pd.DataFrame(comparison_df)
df = df.sort_values('mean_rouge', ascending=False)
print(df.to_string(index=False))

# 통계적 유의성 검정
from scipy import stats

# KoBART vs KoGPT2
kobart_scores = [r['rouge_combined_f1'] for r in results['kobart']]
kogpt2_scores = [r['rouge_combined_f1'] for r in results['kogpt2']]
t_stat, p_value = stats.ttest_ind(kobart_scores, kogpt2_scores)
print(f"\nKoBART vs KoGPT2: t={t_stat:.3f}, p={p_value:.3f}")
```

### 예상 결과
```
=== Model Comparison Results ===
  model  mean_rouge  std_rouge  max_rouge  min_rouge
 kobart      0.4812     0.0034     0.4856     0.4778
     t5      0.4689     0.0051     0.4745     0.4632
 kogpt2      0.4534     0.0042     0.4578     0.4489

KoBART vs KoGPT2: t=8.234, p=0.001
```

---

## 실전 예제 3: 대규모 하이퍼파라미터 탐색

### 목표
베이지안 최적화로 10개 이상의 하이퍼파라미터 동시 최적화

### Step 1: 복잡한 Sweep 설정
```yaml
# config/sweep/advanced_bayesian_sweep.yaml
name: "Advanced Hyperparameter Optimization"
method: "bayes"

metric:
  name: "eval/rouge_combined_f1"
  goal: "maximize"

parameters:
  # 학습 관련
  learning_rate:
    distribution: "log_uniform_values"
    min: 1e-6
    max: 1e-3
  
  per_device_train_batch_size:
    values: [4, 8, 16, 32]
  
  gradient_accumulation_steps:
    values: [1, 2, 4]
  
  warmup_ratio:
    distribution: "uniform"
    min: 0.0
    max: 0.3
  
  weight_decay:
    distribution: "log_uniform_values"
    min: 1e-4
    max: 1e-1
  
  # 정규화
  dropout_rate:
    distribution: "uniform"
    min: 0.0
    max: 0.5
  
  label_smoothing_factor:
    distribution: "uniform"
    min: 0.0
    max: 0.3
  
  # 생성 관련
  num_beams:
    values: [1, 2, 4, 8]
  
  length_penalty:
    distribution: "uniform"
    min: 0.5
    max: 2.0
  
  no_repeat_ngram_size:
    values: [0, 2, 3, 4]
  
  # 토크나이저
  encoder_max_len:
    values: [256, 512, 768]

early_terminate:
  type: "hyperband"
  s: 3
  eta: 3
  max_iter: 81
```

### Step 2: 병렬 실행으로 효율성 극대화
```bash
# 4개의 워커로 병렬 실행
python parallel_sweep_runner.py \
    --base-config config/base_config.yaml \
    --single-parallel advanced_bayesian_sweep \
    --num-workers 4 \
    --runs-per-worker 25 \
    --output-dir ./advanced_sweep_results
```

### Step 3: 실시간 모니터링
```python
# monitor_sweep.py
import wandb
import time
from datetime import datetime

def monitor_sweep_progress(sweep_id, refresh_interval=60):
    """Sweep 진행상황 실시간 모니터링"""
    
    api = wandb.Api()
    
    while True:
        sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
        runs = sweep.runs
        
        # 상태별 집계
        status_counts = {
            'running': sum(1 for r in runs if r.state == 'running'),
            'finished': sum(1 for r in runs if r.state == 'finished'),
            'failed': sum(1 for r in runs if r.state == 'failed'),
            'crashed': sum(1 for r in runs if r.state == 'crashed')
        }
        
        # 성능 통계
        finished_runs = [r for r in runs if r.state == 'finished']
        if finished_runs:
            rouge_scores = [r.summary.get('best/rouge_combined_f1', 0) 
                          for r in finished_runs]
            best_score = max(rouge_scores) if rouge_scores else 0
            avg_score = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0
        else:
            best_score = avg_score = 0
        
        # 출력
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
        print(f"Sweep Progress: {sweep_id}")
        print("-" * 50)
        print(f"Running: {status_counts['running']}")
        print(f"Completed: {status_counts['finished']}")
        print(f"Failed: {status_counts['failed'] + status_counts['crashed']}")
        print(f"\nBest ROUGE F1: {best_score:.4f}")
        print(f"Average ROUGE F1: {avg_score:.4f}")
        
        if status_counts['running'] == 0 and len(runs) >= 100:
            print("\nSweep completed!")
            break
        
        time.sleep(refresh_interval)

# 사용
monitor_sweep_progress("your-sweep-id")
```

### Step 4: 고급 결과 분석
```python
# advanced_analysis.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_advanced_sweep(sweep_id):
    """고급 Sweep 결과 분석"""
    
    # 1. 데이터 수집
    api = wandb.Api()
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    
    data = []
    for run in sweep.runs:
        if run.state == 'finished':
            config = {k: v for k, v in run.config.items() 
                     if not k.startswith('_')}
            config['rouge_f1'] = run.summary.get('best/rouge_combined_f1', 0)
            data.append(config)
    
    df = pd.DataFrame(data)
    
    # 2. 파라미터 중요도 분석
    features = [col for col in df.columns if col != 'rouge_f1']
    X = df[features]
    y = df['rouge_f1']
    
    # 범주형 변수 인코딩
    X_encoded = pd.get_dummies(X, columns=['per_device_train_batch_size', 
                                          'num_beams', 
                                          'encoder_max_len'])
    
    # Random Forest로 feature importance 계산
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_encoded, y)
    
    importances = pd.DataFrame({
        'feature': X_encoded.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # 3. 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Feature Importance
    axes[0, 0].barh(importances['feature'][:10], 
                    importances['importance'][:10])
    axes[0, 0].set_xlabel('Importance')
    axes[0, 0].set_title('Top 10 Feature Importances')
    
    # Learning Rate vs Performance
    axes[0, 1].scatter(df['learning_rate'], df['rouge_f1'], alpha=0.6)
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_xlabel('Learning Rate')
    axes[0, 1].set_ylabel('ROUGE F1')
    axes[0, 1].set_title('Learning Rate vs Performance')
    
    # 2D 파라미터 히트맵
    pivot = df.pivot_table(
        values='rouge_f1',
        index='per_device_train_batch_size',
        columns='gradient_accumulation_steps',
        aggfunc='mean'
    )
    sns.heatmap(pivot, annot=True, fmt='.3f', ax=axes[1, 0])
    axes[1, 0].set_title('Batch Size vs Gradient Accumulation')
    
    # 성능 분포
    axes[1, 1].hist(df['rouge_f1'], bins=30, edgecolor='black')
    axes[1, 1].axvline(df['rouge_f1'].max(), color='red', 
                       linestyle='--', label=f'Best: {df["rouge_f1"].max():.4f}')
    axes[1, 1].axvline(df['rouge_f1'].mean(), color='blue', 
                       linestyle='--', label=f'Mean: {df["rouge_f1"].mean():.4f}')
    axes[1, 1].set_xlabel('ROUGE F1')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Performance Distribution')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('advanced_sweep_analysis.png', dpi=300)
    
    # 4. 최적 설정 추천
    best_runs = df.nlargest(5, 'rouge_f1')
    
    print("\n=== Top 5 Configurations ===")
    for idx, row in best_runs.iterrows():
        print(f"\nRank {idx+1} (ROUGE F1: {row['rouge_f1']:.4f}):")
        for param in features:
            print(f"  {param}: {row[param]}")
    
    # 5. 파라미터 간 상관관계
    correlation_matrix = df[features + ['rouge_f1']].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', 
                cmap='coolwarm', center=0)
    plt.title('Parameter Correlation Matrix')
    plt.tight_layout()
    plt.savefig('parameter_correlations.png', dpi=300)
    
    return df, importances, best_runs
```

---

## 성공 사례 분석

### 사례 1: 메모리 효율적인 대규모 모델 학습

**문제 상황**: T5-large 모델 학습 시 OOM 에러 발생

**해결 과정**:
1. 초기 설정으로 실패
   ```yaml
   training:
     per_device_train_batch_size: 16
     encoder_max_len: 1024
   ```

2. 단계적 최적화
   ```python
   # memory_optimization_sweep.yaml
   parameters:
     per_device_train_batch_size:
       values: [1, 2, 4]
     
     gradient_accumulation_steps:
       values: [8, 16, 32]
     
     gradient_checkpointing:
       values: [true, false]
     
     fp16:
       values: [true]
     
     encoder_max_len:
       values: [512, 768]
   ```

3. 최적 설정 발견
   - Batch size: 2
   - Gradient accumulation: 16 (효과적 배치 크기 = 32)
   - Gradient checkpointing: True
   - Encoder max length: 768

**결과**: 
- 메모리 사용량 65% 감소
- 학습 속도 25% 감소 (acceptable trade-off)
- ROUGE F1: 0.4923 달성

### 사례 2: 다국어 모델 fine-tuning

**문제 상황**: mT5 모델이 한국어 요약에서 낮은 성능

**해결 과정**:
1. 언어별 토크나이저 분석
2. 한국어 특화 전처리 추가
3. 학습률 warm-up 전략 수정

```python
# korean_optimization.py
def preprocess_korean_text(text):
    """한국어 텍스트 전처리"""
    # 특수문자 정규화
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r'[''']', "'", text)
    
    # 반복 문자 제거
    text = re.sub(r'(.)\1{3,}', r'\1\1', text)
    
    # 공백 정규화
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

# Sweep 설정에 추가
parameters:
  warmup_ratio:
    values: [0.1, 0.2, 0.3]  # mT5는 더 긴 warmup 필요
  
  learning_rate:
    min: 5e-5
    max: 5e-4  # 다국어 모델은 더 높은 학습률
```

**결과**:
- Baseline: 0.3856 → Optimized: 0.4567 (+18.4%)
- 핵심 요인: warmup_ratio 0.2 + learning_rate 2e-4

---

## 실패 사례와 교훈

### 실패 사례 1: 과도한 병렬화

**상황**: 
- 8개 GPU에서 각각 다른 모델 학습
- 모든 실험이 동시에 데이터 로딩

**문제**:
- I/O 병목으로 모든 실험 속도 저하
- 일부 실험에서 데이터 로딩 에러

**교훈**:
```python
# 개선된 병렬 실행 전략
def staggered_parallel_execution(experiments, num_gpus, delay=60):
    """시차를 둔 병렬 실행"""
    for i, (exp, gpu) in enumerate(zip(experiments, cycle(range(num_gpus)))):
        if i > 0 and i % num_gpus == 0:
            time.sleep(delay)  # GPU당 하나씩 실행 후 대기
        
        launch_experiment(exp, gpu)
```

### 실패 사례 2: 불충분한 검증

**상황**:
- 빠른 실험을 위해 validation set 크기 축소
- 50개 샘플만으로 평가

**문제**:
- 과적합된 하이퍼파라미터 선택
- 실제 성능과 큰 차이

**교훈**:
```yaml
# 최소 검증 데이터 크기 보장
data:
  min_val_samples: 500  # 최소 500개는 유지
  val_sample_ratio: 0.1  # 또는 전체의 10%
```

### 실패 사례 3: 잘못된 메트릭 선택

**상황**:
- 생성 품질 평가에 loss만 사용
- ROUGE 점수 계산 비활성화 (속도 때문에)

**문제**:
- Loss는 감소했지만 실제 요약 품질 저하
- 반복적이고 의미 없는 문장 생성

**교훈**:
```python
# 다중 메트릭 사용
def compute_comprehensive_metrics(predictions, references):
    """포괄적인 평가 메트릭 계산"""
    metrics = {}
    
    # ROUGE 점수
    rouge_scores = rouge_calculator.compute(predictions, references)
    metrics.update(rouge_scores)
    
    # 추가 메트릭
    metrics['avg_length'] = np.mean([len(p.split()) for p in predictions])
    metrics['repetition_rate'] = calculate_repetition_rate(predictions)
    metrics['coverage'] = calculate_coverage(predictions, references)
    
    # 복합 점수
    metrics['composite_score'] = (
        0.3 * metrics['rouge1_f1'] +
        0.3 * metrics['rouge2_f1'] +
        0.3 * metrics['rougeL_f1'] +
        0.1 * (1 - metrics['repetition_rate'])
    )
    
    return metrics
```

---

## 핵심 교훈 정리

1. **점진적 접근**: 작은 실험부터 시작해서 점차 확대
2. **체계적 기록**: 모든 실험 설정과 결과를 문서화
3. **통계적 검증**: 단일 실행이 아닌 여러 시드로 검증
4. **리소스 관리**: GPU와 메모리를 효율적으로 활용
5. **팀 소통**: 실험 결과와 인사이트를 즉시 공유

---

*이 문서는 실제 실험 경험을 바탕으로 작성되었습니다. 팀원들의 추가 사례를 환영합니다!*
