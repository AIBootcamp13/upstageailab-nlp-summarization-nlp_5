# 설정 파일 상세 가이드

## 목차
1. [설정 파일 계층 구조](#설정-파일-계층-구조)
2. [기본 설정 (base_config.yaml)](#기본-설정-base_configyaml)
3. [모델별 설정](#모델별-설정)
4. [Sweep 설정](#sweep-설정)
5. [환경변수 활용](#환경변수-활용)
6. [고급 설정 기법](#고급-설정-기법)

---

## 설정 파일 계층 구조

우리의 설정 시스템은 계층적 구조로 되어 있어, 유연하고 재사용 가능한 설정 관리가 가능합니다.

```
config/
├── base_config.yaml          # 기본 설정 (필수)
├── models/                   # 모델별 특화 설정
│   ├── kobart.yaml
│   ├── kogpt2.yaml
│   ├── t5.yaml
│   └── mt5.yaml
└── sweep/                    # Sweep 실험 설정
    ├── hyperparameter_sweep.yaml
    ├── model_comparison_sweep.yaml
    ├── generation_params_sweep.yaml
    └── ablation_study_sweep.yaml
```

### 설정 병합 우선순위
1. 환경변수 (최우선)
2. Sweep 파라미터
3. 모델별 설정
4. 기본 설정 (base_config.yaml)

---

## 기본 설정 (base_config.yaml)

### 전체 구조 및 상세 설명

```yaml
# ===== 메타 정보 =====
meta:
  experiment_name: "dialogue_summarization_baseline"
  version: "1.0"
  description: "NLP 대화 요약 기본 실험 설정"
  author: "NLP Team 5"
  created_at: "2024-07-26"
  tags: ["baseline", "kobart", "dialogue"]

# ===== 일반 설정 =====
general:
  seed: 42                    # 재현성을 위한 시드값
  model_name: "digit82/kobart-summarization"  # 기본 모델
  output_dir: "./outputs"     # 결과 저장 디렉토리
  device: "auto"              # auto, cuda, cuda:0, cuda:1, cpu
  num_workers: 4              # 데이터 로딩 워커 수
  mixed_precision: true       # AMP 사용 여부

# ===== 데이터 설정 =====
data:
  train_path: "../data/train.csv"
  val_path: "../data/dev.csv"
  test_path: "../data/test.csv"
  max_train_samples: null     # null이면 전체 사용
  max_val_samples: null
  preprocessing:
    remove_special_chars: false
    lowercase: false
    max_source_length: 1024   # 원본 대화 최대 길이
    max_target_length: 128    # 요약문 최대 길이

# ===== 모델 설정 =====
model:
  architecture: "kobart"      # kobart, kogpt2, t5, mt5
  checkpoint: "digit82/kobart-summarization"
  load_pretrained: true       # 사전학습 가중치 사용
  freeze_encoder: false       # 인코더 고정 여부
  freeze_embeddings: false    # 임베딩 레이어 고정
  dropout_rate: 0.1          # 드롭아웃 비율
  gradient_checkpointing: false  # 메모리 절약 (속도 감소)

# ===== 토크나이저 설정 =====
tokenizer:
  encoder_max_len: 512        # 인코더 입력 최대 길이
  decoder_max_len: 128        # 디코더 입력 최대 길이
  padding: "max_length"       # padding 전략: max_length, longest
  truncation: true            # 긴 시퀀스 자르기
  stride: 50                  # 긴 문서 처리 시 오버랩

# ===== 학습 설정 =====
training:
  # 기본 학습 파라미터
  per_device_train_batch_size: 16
  per_device_eval_batch_size: 16
  gradient_accumulation_steps: 1    # 효과적 배치 크기 = 16 * 1 = 16
  learning_rate: 5e-5
  weight_decay: 0.01
  adam_beta1: 0.9
  adam_beta2: 0.999
  adam_epsilon: 1e-8
  max_grad_norm: 1.0
  
  # 학습 스케줄
  num_train_epochs: 3
  max_steps: -1              # -1이면 에폭 기준
  warmup_ratio: 0.1          # 전체 스텝의 10% warmup
  warmup_steps: 0            # 0이면 warmup_ratio 사용
  lr_scheduler_type: "linear"  # linear, cosine, cosine_with_restarts
  
  # 평가 및 저장
  evaluation_strategy: "steps"   # steps, epoch
  eval_steps: 500
  save_strategy: "steps"
  save_steps: 500
  save_total_limit: 3        # 최대 체크포인트 개수
  load_best_model_at_end: true
  metric_for_best_model: "eval_rouge_combined_f1"
  greater_is_better: true
  
  # 조기 종료
  early_stopping_patience: 3
  early_stopping_threshold: 0.001
  
  # 최적화 옵션
  fp16: true                 # Mixed precision training
  fp16_opt_level: "O1"       # O0, O1, O2, O3
  dataloader_num_workers: 4
  remove_unused_columns: false
  label_smoothing_factor: 0.0
  optim: "adamw_torch"       # adamw_torch, adamw_hf, sgd, adafactor
  
  # 로깅
  logging_dir: "./logs"
  logging_steps: 50
  logging_first_step: true
  report_to: ["wandb"]       # wandb, tensorboard, none
  
# ===== 생성 설정 =====
generation:
  max_length: 100            # 생성 최대 길이
  min_length: 10             # 생성 최소 길이
  num_beams: 4               # 빔 서치 크기
  length_penalty: 1.0        # 길이 패널티 (>1: 긴 문장 선호)
  no_repeat_ngram_size: 2    # n-gram 반복 방지
  early_stopping: true       # 모든 빔이 EOS 도달 시 종료
  do_sample: false           # 샘플링 사용 여부
  temperature: 1.0           # 샘플링 온도 (do_sample=true일 때)
  top_k: 50                  # Top-k 샘플링
  top_p: 0.95               # Top-p (nucleus) 샘플링
  repetition_penalty: 1.0    # 반복 패널티
  bad_words_ids: null        # 생성 금지 토큰 ID

# ===== 평가 설정 =====
evaluation:
  metrics: ["rouge1", "rouge2", "rougeL"]  # 사용할 메트릭
  multi_reference: false     # 다중 참조 요약문 지원
  rouge_use_stemmer: true    # 어간 추출 사용
  rouge_tokenize_korean: true  # 한국어 토크나이저 사용
  prediction_loss_only: false
  generation_num_beams: 4    # 평가 시 빔 크기
  generation_max_length: 100  # 평가 시 최대 길이

# ===== WandB 설정 =====
wandb:
  project: "nlp-dialogue-summarization"
  entity: null               # 팀/조직 이름
  name: null                 # 실행 이름 (자동 생성)
  tags: ["baseline"]         # 태그
  notes: null                # 실행 설명
  mode: "online"             # online, offline, disabled
  log_model: "end"           # false, end, checkpoint
  save_code: true            # 코드 저장 여부
  group: null                # 실행 그룹

# ===== 실험 추적 설정 =====
experiment_tracking:
  enabled: true
  track_gradients: false     # 그래디언트 히스토그램 (느림)
  track_parameters: true     # 파라미터 통계
  log_every_n_steps: 100     # 상세 로깅 주기

# ===== 추론 설정 =====
inference:
  batch_size: 32             # 추론 배치 크기
  use_cache: true            # KV 캐시 사용
  num_workers: 4             # 데이터 로딩 워커
```

---

## 모델별 설정

각 모델의 특성에 맞는 최적화된 설정을 제공합니다.

### KoBART 설정 (models/kobart.yaml)
```yaml
model:
  architecture: "kobart"
  checkpoint: "digit82/kobart-summarization"
  
tokenizer:
  encoder_max_len: 512       # BART는 512가 적당
  decoder_max_len: 128

training:
  learning_rate: 5e-5        # BART 최적 학습률
  warmup_ratio: 0.1

generation:
  num_beams: 4
  length_penalty: 1.0
  no_repeat_ngram_size: 2
```

### KoGPT2 설정 (models/kogpt2.yaml)
```yaml
model:
  architecture: "kogpt2"
  checkpoint: "skt/kogpt2-base-v2"
  
tokenizer:
  encoder_max_len: 1024      # GPT2는 더 긴 컨텍스트 가능
  decoder_max_len: 128
  padding_side: "left"       # GPT는 left padding

training:
  learning_rate: 3e-5        # GPT는 약간 낮은 학습률
  per_device_train_batch_size: 8  # 메모리 사용량이 높음

generation:
  do_sample: true            # GPT는 샘플링이 효과적
  temperature: 0.8
  top_p: 0.9
```

### T5 설정 (models/t5.yaml)
```yaml
model:
  architecture: "t5"
  checkpoint: "KETI-AIR/ke-t5-base"
  
tokenizer:
  encoder_max_len: 512
  decoder_max_len: 128
  add_prefix_space: false

training:
  learning_rate: 1e-4        # T5는 높은 학습률 선호
  warmup_steps: 1000         # 고정 warmup

generation:
  num_beams: 4
  early_stopping: true
```

### mT5 설정 (models/mt5.yaml)
```yaml
model:
  architecture: "mt5"
  checkpoint: "google/mt5-base"
  
tokenizer:
  encoder_max_len: 512
  decoder_max_len: 128

training:
  learning_rate: 1e-4
  gradient_accumulation_steps: 2  # mT5는 큰 배치 선호
  fp16: true                 # 메모리 절약 필수

generation:
  num_beams: 4
  length_penalty: 0.8        # 짧은 요약 선호
```

---

## Sweep 설정

### 하이퍼파라미터 튜닝 (sweep/hyperparameter_sweep.yaml)
```yaml
name: "Hyperparameter Optimization"
method: "bayes"              # grid, random, bayes
metric:
  name: "best/rouge_combined_f1"
  goal: "maximize"

parameters:
  # 학습률 탐색 (로그 스케일)
  learning_rate:
    distribution: "log_uniform_values"
    min: 1e-6
    max: 1e-3
  
  # 배치 크기 (메모리 고려)
  per_device_train_batch_size:
    values: [4, 8, 16, 32]
  
  # Warmup 비율
  warmup_ratio:
    distribution: "uniform"
    min: 0.0
    max: 0.2
  
  # Weight decay
  weight_decay:
    values: [0.0, 0.01, 0.1]
  
  # 학습 에폭
  num_train_epochs:
    values: [3, 5, 7]
  
  # Label smoothing
  label_smoothing_factor:
    values: [0.0, 0.1, 0.2]

# 조기 종료 설정
early_terminate:
  type: "hyperband"
  s: 2                       # 최대 반감기
  eta: 3                     # 반감 비율
  max_iter: 27               # 최대 반복
```

### 모델 비교 (sweep/model_comparison_sweep.yaml)
```yaml
name: "Model Architecture Comparison"
method: "grid"               # 모든 조합 테스트

parameters:
  # 모델 선택
  model_architecture:
    values: ["kobart", "kogpt2", "t5", "mt5"]
  
  # 모델별 체크포인트 (자동 매핑)
  model_checkpoint:
    values: [
      "digit82/kobart-summarization",
      "skt/kogpt2-base-v2",
      "KETI-AIR/ke-t5-base",
      "google/mt5-base"
    ]
  
  # 공통 파라미터
  learning_rate:
    values: [3e-5, 5e-5]
  
  per_device_train_batch_size:
    values: [8, 16]
```

### 생성 파라미터 최적화 (sweep/generation_params_sweep.yaml)
```yaml
name: "Generation Parameters Optimization"
method: "random"

parameters:
  # 빔 서치 크기
  num_beams:
    values: [1, 2, 4, 8]
  
  # 길이 패널티
  length_penalty:
    distribution: "uniform"
    min: 0.5
    max: 2.0
  
  # n-gram 반복 방지
  no_repeat_ngram_size:
    values: [0, 2, 3, 4]
  
  # 샘플링 파라미터
  do_sample:
    values: [true, false]
  
  temperature:
    distribution: "uniform"
    min: 0.5
    max: 1.5
  
  top_p:
    values: [0.9, 0.92, 0.95]
```

### Ablation Study (sweep/ablation_study_sweep.yaml)
```yaml
name: "Component Ablation Study"
method: "grid"

parameters:
  # 데이터 증강
  use_data_augmentation:
    values: [true, false]
  
  # 토크나이저 설정
  encoder_max_len:
    values: [256, 512, 1024]
  
  # 모델 구성 요소
  freeze_encoder:
    values: [true, false]
  
  freeze_embeddings:
    values: [true, false]
  
  # 정규화
  dropout_rate:
    values: [0.0, 0.1, 0.3]
  
  label_smoothing_factor:
    values: [0.0, 0.1]
```

---

## 환경변수 활용

### 지원되는 환경변수

```bash
# WandB 설정
export WANDB_PROJECT="my-project"
export WANDB_ENTITY="my-team"
export WANDB_MODE="offline"  # offline 실행

# 모델 설정
export MODEL_NAME="kogpt2"
export OUTPUT_DIR="/path/to/outputs"

# 학습 파라미터
export BATCH_SIZE=8
export LEARNING_RATE=1e-4
export NUM_EPOCHS=5

# 디바이스 설정
export CUDA_VISIBLE_DEVICES="0,1"  # GPU 0, 1번 사용
```

### 사용 예시

```bash
# 환경변수로 배치 크기 오버라이드
BATCH_SIZE=4 python trainer.py --config config/base_config.yaml

# 여러 환경변수 동시 설정
export LEARNING_RATE=1e-4
export BATCH_SIZE=8
export NUM_EPOCHS=5
python sweep_runner.py --base-config config/base_config.yaml
```

---

## 고급 설정 기법

### 1. 조건부 설정

```python
# config_modifier.py
from utils.config_manager import ConfigManager

def modify_config_for_long_sequences(config):
    """긴 시퀀스를 위한 설정 자동 조정"""
    if config['tokenizer']['encoder_max_len'] > 512:
        # 배치 크기 감소
        config['training']['per_device_train_batch_size'] = min(
            config['training']['per_device_train_batch_size'], 8
        )
        # Gradient accumulation 증가
        config['training']['gradient_accumulation_steps'] = max(
            config['training']['gradient_accumulation_steps'], 2
        )
        # Gradient checkpointing 활성화
        config['model']['gradient_checkpointing'] = True
    
    return config
```

### 2. 동적 설정 생성

```python
# dynamic_config.py
import yaml
from datetime import datetime

def create_experiment_config(base_config_path, experiment_type):
    """실험 타입에 따른 동적 설정 생성"""
    
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 실험 타입별 설정
    if experiment_type == "quick_test":
        config['training']['num_train_epochs'] = 1
        config['training']['eval_steps'] = 100
        config['data']['max_train_samples'] = 1000
        
    elif experiment_type == "full_training":
        config['training']['num_train_epochs'] = 10
        config['training']['early_stopping_patience'] = 5
        
    elif experiment_type == "memory_efficient":
        config['training']['per_device_train_batch_size'] = 4
        config['training']['gradient_accumulation_steps'] = 4
        config['training']['fp16'] = True
        config['model']['gradient_checkpointing'] = True
    
    # 실험명 자동 생성
    config['meta']['experiment_name'] = f"{experiment_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    return config
```

### 3. 설정 검증 및 자동 수정

```python
# config_validator.py
def validate_and_fix_config(config):
    """설정 검증 및 자동 수정"""
    
    # GPU 메모리에 따른 자동 조정
    import torch
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        
        if gpu_memory < 16 * 1024**3:  # 16GB 미만
            print("Limited GPU memory detected, adjusting settings...")
            config['training']['per_device_train_batch_size'] = min(
                config['training']['per_device_train_batch_size'], 8
            )
            config['training']['fp16'] = True
    
    # 모델과 토크나이저 호환성 검증
    if config['model']['architecture'] == 'kogpt2':
        # GPT2는 decoder-only이므로 조정
        config['tokenizer']['padding_side'] = 'left'
        config['generation']['do_sample'] = True
    
    return config
```

### 4. 다중 설정 파일 병합

```python
# merge_configs.py
from utils.config_manager import ConfigManager

# 여러 설정 파일 병합
cm = ConfigManager()

# 기본 설정 로드
config = cm.load_config("config/base_config.yaml")

# 실험별 설정 병합
experiment_config = cm.load_config("config/experiments/long_document.yaml")
config = cm._deep_merge(config, experiment_config)

# 팀 설정 병합
team_config = cm.load_config("config/team_settings.yaml")
config = cm._deep_merge(config, team_config)

# 최종 설정 저장
cm.save_config(config, "config/final_config.yaml")
```

---

## 설정 최적화 팁

### 메모리 최적화
1. **배치 크기와 Gradient Accumulation 균형**
   ```yaml
   # 효과적인 배치 크기 = 4 * 4 = 16
   training:
     per_device_train_batch_size: 4
     gradient_accumulation_steps: 4
   ```

2. **Mixed Precision Training**
   ```yaml
   training:
     fp16: true
     fp16_opt_level: "O2"  # 더 공격적인 최적화
   ```

3. **Gradient Checkpointing**
   ```yaml
   model:
     gradient_checkpointing: true  # 메모리 35% 절약, 속도 15% 감소
   ```

### 학습 속도 최적화
1. **데이터 로더 최적화**
   ```yaml
   training:
     dataloader_num_workers: 8
     dataloader_pin_memory: true
   ```

2. **효율적인 평가 주기**
   ```yaml
   training:
     evaluation_strategy: "steps"
     eval_steps: 1000  # 너무 자주 평가하지 않기
   ```

### 성능 최적화
1. **학습률 스케줄링**
   ```yaml
   training:
     lr_scheduler_type: "cosine"
     warmup_ratio: 0.1
   ```

2. **정규화 기법**
   ```yaml
   training:
     weight_decay: 0.01
     label_smoothing_factor: 0.1
   model:
     dropout_rate: 0.1
   ```

---

## 문제 해결 가이드

### 설정 파일 에러
```python
# 설정 파일 검증
from utils.config_manager import ConfigManager

cm = ConfigManager()
is_valid = cm.validate_config_file("config/my_config.yaml")
if not is_valid:
    print("Configuration file has errors!")
```

### 설정 디버깅
```python
# 최종 병합된 설정 확인
import json

config = cm.load_config("config/base_config.yaml", model_config="kobart")
print(json.dumps(config, indent=2))
```

---

*더 자세한 내용은 [README.md](README.md)와 [best_practices.md](best_practices.md)를 참고하세요.*
