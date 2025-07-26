# 실험 자동화 베스트 프랙티스

## 목차
1. [실험 설계 가이드라인](#실험-설계-가이드라인)
2. [효율적인 하이퍼파라미터 튜닝](#효율적인-하이퍼파라미터-튜닝)
3. [성능 최적화 전략](#성능-최적화-전략)
4. [실험 관리 및 추적](#실험-관리-및-추적)
5. [팀 협업 가이드](#팀-협업-가이드)
6. [일반적인 실수와 해결책](#일반적인-실수와-해결책)

---

## 실험 설계 가이드라인

### 1. 체계적인 실험 계획 수립

#### ✅ 좋은 예: 단계별 접근
```python
# 1단계: 빠른 실행으로 파이프라인 검증
quick_test_config = {
    "data": {"max_train_samples": 1000},
    "training": {
        "num_train_epochs": 1,
        "eval_steps": 100
    }
}

# 2단계: 소규모 그리드 서치
small_grid_search = {
    "learning_rate": [1e-5, 5e-5, 1e-4],
    "batch_size": [8, 16]
}

# 3단계: 베이지안 최적화로 세밀한 튜닝
bayesian_optimization = {
    "method": "bayes",
    "learning_rate": {"min": 1e-5, "max": 1e-4},
    "warmup_ratio": {"min": 0.0, "max": 0.2}
}
```

#### ❌ 나쁜 예: 무계획적 전체 탐색
```python
# 너무 많은 조합 (3^7 = 2,187개 실험!)
huge_grid_search = {
    "learning_rate": [1e-6, 1e-5, 1e-4],
    "batch_size": [4, 8, 16],
    "warmup_ratio": [0.0, 0.1, 0.2],
    "weight_decay": [0.0, 0.01, 0.1],
    "num_epochs": [3, 5, 7],
    "dropout": [0.0, 0.1, 0.2],
    "label_smoothing": [0.0, 0.1, 0.2]
}
```

### 2. 실험 우선순위 결정

#### 실험 영향도 매트릭스
| 파라미터 | 영향도 | 탐색 범위 | 우선순위 |
|---------|--------|----------|----------|
| Learning Rate | 높음 | 1e-6 ~ 1e-3 | 1 |
| Batch Size | 높음 | 4 ~ 32 | 2 |
| Model Architecture | 높음 | 4개 모델 | 3 |
| Warmup Ratio | 중간 | 0.0 ~ 0.2 | 4 |
| Weight Decay | 중간 | 0.0 ~ 0.1 | 5 |
| Dropout | 낮음 | 0.0 ~ 0.3 | 6 |

### 3. 실험 명명 규칙

```python
def generate_experiment_name(config, sweep_type):
    """체계적인 실험명 생성"""
    from datetime import datetime
    
    components = [
        sweep_type,                              # hp_tuning, model_comp
        config['model']['architecture'],         # kobart, kogpt2
        f"lr{config['training']['learning_rate']:.0e}",
        f"bs{config['training']['per_device_train_batch_size']}",
        datetime.now().strftime("%m%d_%H%M")
    ]
    
    return "_".join(components)
    # 예: hp_tuning_kobart_lr5e-05_bs16_0726_1430
```

---

## 효율적인 하이퍼파라미터 튜닝

### 1. 점진적 탐색 전략

#### Phase 1: 넓은 범위 탐색 (Random Search)
```yaml
# config/sweep/phase1_wide_search.yaml
method: "random"
metric:
  name: "eval/rouge_combined_f1"
  goal: "maximize"

parameters:
  learning_rate:
    distribution: "log_uniform_values"
    min: 1e-6
    max: 1e-3
  
  batch_size:
    values: [4, 8, 16, 32]

count: 20  # 적은 수의 실험으로 대략적인 범위 파악
```

#### Phase 2: 좁은 범위 탐색 (Bayesian Optimization)
```yaml
# config/sweep/phase2_narrow_search.yaml
method: "bayes"
metric:
  name: "eval/rouge_combined_f1"
  goal: "maximize"

parameters:
  learning_rate:
    distribution: "log_uniform_values"
    min: 3e-5    # Phase 1에서 찾은 최적 범위
    max: 7e-5
  
  warmup_ratio:
    distribution: "uniform"
    min: 0.05
    max: 0.15

count: 50  # 더 많은 실험으로 정밀 튜닝
```

### 2. 효율적인 Early Stopping

```yaml
# Hyperband 설정으로 자원 효율적 사용
early_terminate:
  type: "hyperband"
  s: 2          # 최대 2번 반감
  eta: 3        # 1/3만 다음 단계로
  max_iter: 27  # 최대 27 스텝

# 커스텀 조기 종료 조건
training:
  early_stopping_patience: 3
  early_stopping_threshold: 0.001
  # 3번 연속으로 0.001 미만 개선 시 종료
```

### 3. 파라미터 간 의존성 고려

```python
def get_dependent_params(primary_params):
    """파라미터 간 의존성을 고려한 설정"""
    
    config = {}
    
    # 배치 크기에 따른 학습률 조정 (Linear Scaling Rule)
    base_lr = 5e-5
    base_batch = 16
    actual_batch = primary_params['batch_size']
    config['learning_rate'] = base_lr * (actual_batch / base_batch) ** 0.5
    
    # 시퀀스 길이에 따른 배치 크기 조정
    if primary_params['encoder_max_len'] > 512:
        config['batch_size'] = min(primary_params['batch_size'], 8)
        config['gradient_accumulation_steps'] = max(2, 16 // config['batch_size'])
    
    return config
```

---

## 성능 최적화 전략

### 1. GPU 메모리 최적화

#### 메모리 사용량 프로파일링
```python
import torch
from trainer import DialogueSummarizationTrainer

def profile_memory_usage(config):
    """메모리 사용량 측정"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    trainer = DialogueSummarizationTrainer(config)
    trainer.initialize_components()
    
    # 더미 데이터로 한 스텝 실행
    dummy_batch = trainer.create_dummy_batch()
    trainer.training_step(dummy_batch)
    
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
    print(f"Peak memory usage: {peak_memory:.2f} GB")
    
    return peak_memory
```

#### 메모리 최적화 체크리스트
- [ ] FP16/BF16 training 활성화
- [ ] Gradient checkpointing 사용
- [ ] Gradient accumulation으로 효과적 배치 크기 유지
- [ ] 불필요한 중간 텐서 즉시 삭제
- [ ] DataLoader에서 pin_memory=True 설정

### 2. 학습 속도 최적화

#### 병목 지점 분석
```python
import time
from contextlib import contextmanager

@contextmanager
def timer(name):
    """구간별 시간 측정"""
    start = time.time()
    yield
    print(f"{name}: {time.time() - start:.2f}s")

# 사용 예
with timer("Data Loading"):
    batch = next(iter(train_dataloader))

with timer("Forward Pass"):
    outputs = model(**batch)

with timer("Backward Pass"):
    loss.backward()
```

#### 최적화 기법별 효과
| 기법 | 속도 향상 | 메모리 절약 | 권장 상황 |
|-----|----------|------------|----------|
| Mixed Precision (FP16) | 1.5-2x | 50% | 항상 권장 |
| Gradient Accumulation | - | 선형 감소 | 큰 배치 필요 시 |
| Gradient Checkpointing | 0.7x | 35% | 메모리 부족 시 |
| DataLoader Workers | 1.2-1.5x | - | I/O 병목 시 |
| Compiled Model | 1.1-1.3x | - | PyTorch 2.0+ |

### 3. 모델별 최적화 전략

#### KoBART 최적화
```yaml
# KoBART는 인코더-디코더 구조로 메모리 사용량이 큼
model:
  gradient_checkpointing: true
  
training:
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 2
  fp16: true
  
generation:
  num_beams: 4  # 메모리 고려
```

#### KoGPT2 최적화
```yaml
# GPT2는 left-padding과 캐시 활용이 중요
tokenizer:
  padding_side: "left"
  
model:
  use_cache: true  # 추론 시 KV 캐시
  
training:
  per_device_train_batch_size: 4  # 더 작은 배치
  gradient_accumulation_steps: 4
```

---

## 실험 관리 및 추적

### 1. WandB 활용 전략

#### 효과적인 로깅
```python
# trainer.py의 커스텀 로깅 예시
def log_detailed_metrics(self, metrics, examples=None):
    """상세 메트릭과 예시 로깅"""
    
    # 기본 메트릭
    wandb.log(metrics)
    
    # 예시 텍스트 로깅 (테이블 형태)
    if examples:
        table = wandb.Table(
            columns=["input", "prediction", "reference", "rouge_score"]
        )
        for ex in examples[:10]:  # 상위 10개만
            table.add_data(
                ex['input'][:100] + "...",
                ex['prediction'],
                ex['reference'],
                ex['rouge_score']
            )
        wandb.log({"examples": table})
    
    # 학습률 스케줄 시각화
    if self.lr_scheduler:
        wandb.log({
            "learning_rate": self.lr_scheduler.get_last_lr()[0]
        })
```

#### WandB Reports 활용
```python
# 실험 종료 후 자동 리포트 생성
def create_sweep_report(sweep_id):
    """Sweep 결과 리포트 자동 생성"""
    import wandb.apis
    
    api = wandb.Api()
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    
    # 베스트 런 찾기
    best_run = sweep.best_run()
    
    # 리포트 생성
    report = wandb.Report(
        project=project,
        title=f"Sweep Results: {sweep.name}",
        description=f"Best ROUGE: {best_run.summary['best/rouge_combined_f1']:.4f}"
    )
    
    # 차트 추가
    report.add_panel(
        wandb.ParallelCoordinatesPlot(
            columns=["learning_rate", "batch_size", "rouge_combined_f1"]
        )
    )
    
    report.save()
```

### 2. 로컬 실험 관리

#### 실험 결과 구조화
```
outputs/
├── experiments/
│   ├── hp_tuning_kobart_0726_1430/
│   │   ├── config.yaml           # 사용된 설정
│   │   ├── metrics.json          # 최종 메트릭
│   │   ├── training.log          # 학습 로그
│   │   ├── predictions.jsonl     # 예측 결과
│   │   └── models/
│   │       └── best_model/       # 최고 성능 모델
│   └── model_comp_0726_1630/
└── sweep_results/
    ├── hyperparameter_sweep/
    │   ├── all_results.jsonl     # 모든 실행 결과
    │   ├── best_config.yaml      # 최적 설정
    │   └── analysis_report.html  # 분석 리포트
    └── model_comparison/
```

#### 실험 비교 스크립트
```python
# compare_experiments.py
import json
import pandas as pd
from pathlib import Path

def compare_experiments(exp_dirs):
    """여러 실험 결과 비교"""
    
    results = []
    for exp_dir in exp_dirs:
        metrics_file = Path(exp_dir) / "metrics.json"
        with open(metrics_file) as f:
            metrics = json.load(f)
        
        results.append({
            'experiment': Path(exp_dir).name,
            'rouge1': metrics['best_rouge1_f1'],
            'rouge2': metrics['best_rouge2_f1'],
            'rougeL': metrics['best_rougeL_f1'],
            'combined': metrics['best_rouge_combined_f1']
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values('combined', ascending=False)
    
    print("\n실험 결과 비교:")
    print(df.to_string(index=False))
    
    # 최고 성능 실험
    best_exp = df.iloc[0]
    print(f"\n최고 성능: {best_exp['experiment']}")
    print(f"ROUGE Combined F1: {best_exp['combined']:.4f}")
```

---

## 팀 협업 가이드

### 1. 실험 분담 전략

#### 역할 기반 분담
```yaml
# team_assignments.yaml
assignments:
  - member: "Alice"
    focus: "hyperparameter_tuning"
    models: ["kobart"]
    gpu: "cuda:0"
    
  - member: "Bob"
    focus: "model_comparison"
    models: ["kogpt2", "t5"]
    gpu: "cuda:1"
    
  - member: "Charlie"
    focus: "generation_params"
    models: ["kobart", "mt5"]
    gpu: "cuda:2"

# 병렬 실행 예시
# Alice: python sweep_runner.py --sweep-config hp_tuning_kobart
# Bob: CUDA_VISIBLE_DEVICES=1 python sweep_runner.py --sweep-config model_comp_gpt_t5
# Charlie: CUDA_VISIBLE_DEVICES=2 python sweep_runner.py --sweep-config gen_params
```

#### 시간대별 GPU 활용
```python
# gpu_scheduler.py
import schedule
import subprocess
from datetime import datetime

def run_experiment(gpu_id, config_name):
    """특정 GPU에서 실험 실행"""
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    cmd = [
        'python', 'sweep_runner.py',
        '--base-config', 'config/base_config.yaml',
        '--sweep-config', config_name,
        '--count', '10'
    ]
    
    subprocess.Popen(cmd, env=env)
    print(f"[{datetime.now()}] Started {config_name} on GPU {gpu_id}")

# 스케줄 설정
schedule.every().day.at("22:00").do(run_experiment, 0, "overnight_hp_search")
schedule.every().day.at("02:00").do(run_experiment, 1, "overnight_model_comp")
```

### 2. 결과 공유 및 리뷰

#### 실험 리뷰 템플릿
```markdown
## 실험 리뷰: [실험명]

### 개요
- **실행자**: [이름]
- **날짜**: [YYYY-MM-DD]
- **Sweep ID**: [wandb_sweep_id]
- **총 실행 수**: [N]

### 주요 발견사항
1. **최적 하이퍼파라미터**
   - Learning Rate: 5e-5
   - Batch Size: 16
   - Warmup Ratio: 0.1

2. **성능 향상**
   - Baseline: 0.42 → Best: 0.48 (+14.3%)
   - 주요 요인: 학습률과 warmup의 조합

3. **특이사항**
   - Batch size 32에서 메모리 에러 발생
   - Label smoothing은 성능 저하

### 권장사항
- [ ] 최적 설정으로 전체 데이터 재학습
- [ ] Warmup ratio 0.05-0.15 범위 추가 탐색
- [ ] 다른 optimizer (AdaFactor) 테스트

### 첨부
- [WandB Report 링크]
- [최적 모델 경로]
```

#### 주간 실험 대시보드
```python
# weekly_dashboard.py
import wandb
from datetime import datetime, timedelta

def create_weekly_dashboard(project_name, days=7):
    """주간 실험 대시보드 생성"""
    api = wandb.Api()
    
    # 최근 7일 실행 가져오기
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    runs = api.runs(
        f"{entity}/{project_name}",
        filters={
            "created_at": {"$gte": start_date.isoformat()}
        }
    )
    
    # 통계 수집
    stats = {
        'total_runs': len(runs),
        'successful_runs': sum(1 for r in runs if r.state == 'finished'),
        'failed_runs': sum(1 for r in runs if r.state == 'failed'),
        'total_gpu_hours': sum(r.summary.get('_runtime', 0) / 3600 for r in runs),
        'best_rouge': max((r.summary.get('best/rouge_combined_f1', 0) for r in runs), default=0)
    }
    
    print(f"\n주간 실험 요약 ({start_date.date()} ~ {end_date.date()})")
    print("=" * 50)
    print(f"총 실험 수: {stats['total_runs']}")
    print(f"성공/실패: {stats['successful_runs']}/{stats['failed_runs']}")
    print(f"GPU 사용 시간: {stats['total_gpu_hours']:.1f}시간")
    print(f"최고 ROUGE F1: {stats['best_rouge']:.4f}")
```

### 3. 실험 재현성 보장

#### 실험 스냅샷 생성
```python
# create_snapshot.py
import shutil
import git
import json
from pathlib import Path

def create_experiment_snapshot(exp_name):
    """실험 재현을 위한 스냅샷 생성"""
    snapshot_dir = Path(f"snapshots/{exp_name}")
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 코드 상태 저장
    repo = git.Repo('.')
    with open(snapshot_dir / "git_info.json", 'w') as f:
        json.dump({
            'commit': repo.head.commit.hexsha,
            'branch': repo.active_branch.name,
            'dirty': repo.is_dirty(),
            'untracked': repo.untracked_files
        }, f, indent=2)
    
    # 2. 설정 파일 복사
    shutil.copytree("config", snapshot_dir / "config")
    
    # 3. 환경 정보 저장
    import torch
    import transformers
    
    env_info = {
        'python': sys.version,
        'torch': torch.__version__,
        'transformers': transformers.__version__,
        'cuda': torch.cuda.is_available(),
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }
    
    with open(snapshot_dir / "environment.json", 'w') as f:
        json.dump(env_info, f, indent=2)
    
    # 4. 실행 명령어 저장
    with open(snapshot_dir / "run_command.sh", 'w') as f:
        f.write(f"#!/bin/bash\n")
        f.write(f"# Experiment: {exp_name}\n")
        f.write(f"# Created: {datetime.now()}\n\n")
        f.write(f"python sweep_runner.py --base-config config/base_config.yaml ...\n")
    
    print(f"Snapshot created: {snapshot_dir}")
```

---

## 일반적인 실수와 해결책

### 1. 하이퍼파라미터 튜닝 실수

#### ❌ 실수: 너무 넓은 범위 설정
```yaml
# 나쁜 예
learning_rate:
  min: 1e-7
  max: 1e-1  # 너무 넓음!
```

#### ✅ 해결책: 단계적 범위 좁히기
```yaml
# 좋은 예
# Step 1: 로그 스케일로 대략적 범위 찾기
learning_rate:
  values: [1e-6, 1e-5, 1e-4, 1e-3]

# Step 2: 최적 범위 근처에서 세밀하게
learning_rate:
  min: 3e-5
  max: 7e-5
```

### 2. 메모리 관리 실수

#### ❌ 실수: 고정된 배치 크기
```python
# 나쁜 예
config = {
    "training": {
        "per_device_train_batch_size": 32  # 항상 32?
    }
}
```

#### ✅ 해결책: 동적 배치 크기 조정
```python
# 좋은 예
def get_optimal_batch_size(model_name, seq_length):
    """모델과 시퀀스 길이에 따른 최적 배치 크기"""
    
    # GPU 메모리 체크
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    # 모델별 메모리 사용량 (추정치)
    memory_per_sample = {
        'kobart': 0.5,    # GB per sample
        'kogpt2': 0.7,
        't5': 0.6,
        'mt5': 0.8
    }
    
    # 시퀀스 길이 보정
    length_factor = seq_length / 512
    
    # 안전 마진 (80% 사용)
    safe_batch_size = int(
        (gpu_memory * 0.8) / 
        (memory_per_sample.get(model_name, 0.5) * length_factor)
    )
    
    # 2의 배수로 조정
    return max(1, min(safe_batch_size, 32))
```

### 3. 실험 추적 실수

#### ❌ 실수: 불충분한 로깅
```python
# 나쁜 예
wandb.log({"loss": loss})
```

#### ✅ 해결책: 포괄적인 로깅
```python
# 좋은 예
wandb.log({
    # 기본 메트릭
    "train/loss": loss,
    "train/learning_rate": optimizer.param_groups[0]['lr'],
    "train/epoch": epoch,
    "train/global_step": global_step,
    
    # 상세 메트릭
    "train/grad_norm": grad_norm,
    "train/gpu_memory_used": torch.cuda.memory_allocated() / 1024**3,
    "train/gpu_utilization": get_gpu_utilization(),
    
    # 시간 정보
    "time/data_loading": data_time,
    "time/forward_pass": forward_time,
    "time/backward_pass": backward_time,
    "time/total_step": total_time
})
```

### 4. 평가 전략 실수

#### ❌ 실수: 단일 메트릭 의존
```python
# 나쁜 예
if rouge1_score > best_score:
    save_model()  # ROUGE-1만 고려?
```

#### ✅ 해결책: 복합 메트릭 사용
```python
# 좋은 예
def calculate_composite_score(metrics):
    """복합 평가 점수 계산"""
    # 가중 평균
    weights = {
        'rouge1_f1': 0.3,
        'rouge2_f1': 0.3,
        'rougeL_f1': 0.4
    }
    
    composite = sum(
        metrics[key] * weight 
        for key, weight in weights.items()
    )
    
    # 추가 페널티/보너스
    if metrics.get('avg_length') < 50:  # 너무 짧은 요약 페널티
        composite *= 0.9
    
    return composite
```

### 5. 병렬 실행 실수

#### ❌ 실수: 무분별한 병렬화
```bash
# 나쁜 예 - 메모리 경쟁
for i in {0..9}; do
    python train.py --gpu 0 &  # 모두 같은 GPU!
done
```

#### ✅ 해결책: 스마트한 리소스 분배
```python
# 좋은 예
def distribute_experiments(experiments, available_gpus):
    """실험을 GPU에 효율적으로 분배"""
    
    # GPU별 메모리 체크
    gpu_memory = {}
    for gpu_id in available_gpus:
        device = torch.device(f'cuda:{gpu_id}')
        gpu_memory[gpu_id] = torch.cuda.get_device_properties(device).total_memory
    
    # 실험별 예상 메모리 사용량 계산
    exp_memory = {}
    for exp in experiments:
        model_size = get_model_size(exp['model'])
        batch_size = exp['batch_size']
        exp_memory[exp['name']] = model_size * batch_size
    
    # 빈 패킹 알고리즘으로 최적 분배
    allocation = allocate_experiments(experiments, gpu_memory, exp_memory)
    
    return allocation
```

---

## 고급 팁과 트릭

### 1. 커스텀 콜백 활용

```python
# custom_callbacks.py
from transformers import TrainerCallback

class MemoryMonitorCallback(TrainerCallback):
    """메모리 사용량 모니터링 콜백"""
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 100 == 0:
            memory_used = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            
            if memory_used > 0.9 * memory_reserved:
                logger.warning(
                    f"High memory usage: {memory_used:.2f}GB / {memory_reserved:.2f}GB"
                )
                # 캐시 정리
                torch.cuda.empty_cache()

class LearningRateWarmupCallback(TrainerCallback):
    """커스텀 학습률 웜업"""
    
    def __init__(self, warmup_steps, warmup_strategy='linear'):
        self.warmup_steps = warmup_steps
        self.warmup_strategy = warmup_strategy
    
    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step < self.warmup_steps:
            if self.warmup_strategy == 'linear':
                factor = state.global_step / self.warmup_steps
            elif self.warmup_strategy == 'cosine':
                factor = 0.5 * (1 + math.cos(math.pi * (1 - state.global_step / self.warmup_steps)))
            
            # 학습률 조정
            for param_group in kwargs['optimizer'].param_groups:
                param_group['lr'] = param_group['initial_lr'] * factor
```

### 2. 실험 자동 분석

```python
# auto_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_sweep_results(sweep_dir):
    """Sweep 결과 자동 분석 및 시각화"""
    
    # 결과 로딩
    results = []
    for result_file in Path(sweep_dir).glob("run_*_result.json"):
        with open(result_file) as f:
            results.append(json.load(f))
    
    df = pd.DataFrame(results)
    
    # 1. 파라미터 중요도 분석
    feature_importance = calculate_feature_importance(
        df[['learning_rate', 'batch_size', 'warmup_ratio']], 
        df['best_rouge_combined_f1']
    )
    
    # 2. 최적 파라미터 조합
    best_runs = df.nlargest(5, 'best_rouge_combined_f1')
    
    # 3. 시각화
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 학습률 vs 성능
    axes[0, 0].scatter(df['learning_rate'], df['best_rouge_combined_f1'])
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_xlabel('Learning Rate')
    axes[0, 0].set_ylabel('ROUGE Combined F1')
    
    # 파라미터 히트맵
    pivot = df.pivot_table(
        values='best_rouge_combined_f1',
        index='batch_size',
        columns='warmup_ratio',
        aggfunc='mean'
    )
    sns.heatmap(pivot, annot=True, fmt='.3f', ax=axes[0, 1])
    
    # 수렴 속도
    for idx, run in best_runs.iterrows():
        axes[1, 0].plot(run['training_history'], label=f"Run {run['run_id'][:8]}")
    axes[1, 0].set_xlabel('Steps')
    axes[1, 0].set_ylabel('Validation Loss')
    axes[1, 0].legend()
    
    plt.tight_layout()
    plt.savefig(sweep_dir / 'analysis_report.png')
    
    # 리포트 생성
    with open(sweep_dir / 'analysis_report.md', 'w') as f:
        f.write("# Sweep Analysis Report\n\n")
        f.write(f"## Best Configuration\n")
        f.write(f"- Learning Rate: {best_runs.iloc[0]['learning_rate']:.2e}\n")
        f.write(f"- Batch Size: {best_runs.iloc[0]['batch_size']}\n")
        f.write(f"- Best ROUGE F1: {best_runs.iloc[0]['best_rouge_combined_f1']:.4f}\n")
```

---

## 체크리스트

### 실험 시작 전
- [ ] GPU 가용성 확인
- [ ] 데이터 경로 확인
- [ ] WandB 로그인 상태 확인
- [ ] 빠른 테스트로 파이프라인 검증
- [ ] 실험 계획 문서화

### 실험 중
- [ ] GPU 사용율 모니터링
- [ ] WandB 대시보드 주기적 확인
- [ ] 이상 동작 조기 발견 및 중단
- [ ] 중간 결과 백업

### 실험 후
- [ ] 최고 성능 모델 백업
- [ ] 실험 결과 문서화
- [ ] 팀과 결과 공유
- [ ] 다음 실험 계획 수립
- [ ] 리소스 정리 (GPU 캐시, 임시 파일)

---

*이 문서는 지속적으로 업데이트됩니다. 팀원들의 경험과 인사이트를 공유해주세요!*
