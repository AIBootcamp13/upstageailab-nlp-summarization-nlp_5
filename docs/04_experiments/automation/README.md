# NLP 대화 요약 프로젝트 - 실험 자동화 시스템 가이드

## 목차
1. [시스템 개요](#시스템-개요)
2. [주요 장점](#주요-장점)
3. [시작하기](#시작하기)
4. [빠른 시작 가이드](#빠른-시작-가이드)
5. [상세 사용법](#상세-사용법)
6. [실험 결과 분석](#실험-결과-분석)
7. [문제 해결](#문제-해결)
8. [자주 묻는 질문](#자주-묻는-질문)

---

## 시스템 개요

우리 팀의 실험 자동화 시스템은 체계적인 하이퍼파라미터 튜닝과 실험 관리를 통해 NLP 대화 요약 태스크의 성능을 최적화하는 시스템입니다. 이 시스템을 통해 다음과 같은 작업을 자동화할 수 있습니다:

- 🚀 **하이퍼파라미터 최적화**: 학습률, 배치 크기, 에폭 수 등을 자동으로 튜닝
- 🔬 **모델 비교 실험**: KoBART, KoGPT2, T5, mT5 등 다양한 모델 성능 비교
- 📊 **체계적인 실험 관리**: WandB를 통한 실시간 모니터링과 결과 추적
- ⚡ **병렬 실험 실행**: 여러 실험을 동시에 실행하여 시간 단축

### 시스템 아키텍처

```
project_root/
├── code/
│   ├── config/                 # 설정 파일들
│   │   ├── base_config.yaml   # 기본 설정
│   │   ├── models/            # 모델별 설정
│   │   └── sweep/             # Sweep 설정
│   ├── utils/                  # 유틸리티 모듈
│   ├── trainer.py             # 학습 모듈
│   ├── sweep_runner.py        # Sweep 실행기
│   └── scripts/               # 실행 스크립트
├── data/                      # 데이터 파일
└── outputs/                   # 실험 결과
```

---

## 주요 장점

### 1. 🎯 **기존 워크플로우와의 완벽한 호환성**
- baseline.ipynb의 모든 기능을 그대로 사용 가능
- 기존 config.yaml 형식 자동 마이그레이션
- 점진적 도입 가능 (일부 실험만 자동화)

### 2. 🔧 **유연한 설정 관리**
- YAML 기반 계층적 설정 시스템
- 환경변수를 통한 오버라이드 지원
- 모델별, 실험별 설정 분리

### 3. 📈 **강력한 실험 추적**
- WandB 대시보드에서 실시간 모니터링
- 자동 메트릭 로깅 (ROUGE-1, ROUGE-2, ROUGE-L)
- 최적 모델 자동 저장

### 4. ⏱️ **시간 효율성**
- 병렬 실험 실행으로 시간 단축
- 자동 재시작 기능 (중단된 실험 이어서 실행)
- 스마트 메모리 관리

---

## 시작하기

### 환경 설정

1. **필요한 패키지 설치**
```bash
pip install -r requirements.txt
```

주요 패키지:
- `transformers>=4.30.0`
- `wandb>=0.15.0`
- `torch>=2.0.0`
- `datasets>=2.10.0`
- `evaluate>=0.4.0`
- `rouge-score>=0.1.2`

2. **WandB 설정**
```bash
# WandB 계정 로그인
wandb login

# 프로젝트 설정 (선택사항)
export WANDB_PROJECT="nlp-dialogue-summarization"
export WANDB_ENTITY="your-team-name"
```

3. **GPU 확인**
```python
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")
```

---

## 빠른 시작 가이드

### 🎮 케이스 1: 기본 학습 실행
기존 baseline.ipynb처럼 단일 실험을 실행하고 싶을 때:

```bash
cd code
python trainer.py \
    --config config/base_config.yaml \
    --train-data ../data/train.csv \
    --val-data ../data/dev.csv
```

### 🔬 케이스 2: 하이퍼파라미터 튜닝
최적의 학습률과 배치 크기를 찾고 싶을 때:

```bash
cd code
python sweep_runner.py \
    --base-config config/base_config.yaml \
    --sweep-config hyperparameter_sweep \
    --count 20
```

### 🏆 케이스 3: 모델 성능 비교
여러 모델의 성능을 비교하고 싶을 때:

```bash
cd code
python sweep_runner.py \
    --base-config config/base_config.yaml \
    --sweep-config model_comparison_sweep \
    --count 16  # 4개 모델 × 4번 실행
```

### ⚡ 케이스 4: 병렬 실험 실행
시간을 절약하기 위해 4개의 GPU로 동시 실행:

```bash
cd code
python parallel_sweep_runner.py \
    --base-config config/base_config.yaml \
    --single-parallel hyperparameter_sweep \
    --num-workers 4 \
    --runs-per-worker 5
```

---

## 상세 사용법

### 1. 설정 파일 구조 이해하기

#### base_config.yaml
```yaml
meta:
  experiment_name: "dialogue_summarization"
  version: "1.0"
  description: "대화 요약 기본 실험"

general:
  seed: 42
  model_name: "digit82/kobart-summarization"
  output_dir: "./outputs"
  device: "auto"  # auto, cuda, cpu

model:
  architecture: "kobart"  # kobart, kogpt2, t5, mt5
  checkpoint: "digit82/kobart-summarization"
  load_pretrained: true

tokenizer:
  encoder_max_len: 512
  decoder_max_len: 128

training:
  per_device_train_batch_size: 16
  learning_rate: 5e-5
  num_train_epochs: 3
  warmup_ratio: 0.1
  fp16: true  # GPU 메모리 절약

generation:
  num_beams: 4
  length_penalty: 1.0
  no_repeat_ngram_size: 2
  max_length: 100

wandb:
  project: "nlp-dialogue-summarization"
  entity: null  # 팀 이름
  mode: "online"  # online, offline, disabled
```

### 2. Sweep 설정 커스터마이징

#### 예시: 나만의 Sweep 설정 만들기
```yaml
# config/sweep/my_custom_sweep.yaml
name: "Custom Learning Rate Search"
method: "bayes"  # grid, random, bayes
metric:
  name: "best/rouge_combined_f1"
  goal: "maximize"

parameters:
  learning_rate:
    distribution: "log_uniform_values"
    min: 1e-6
    max: 1e-3
  
  warmup_ratio:
    values: [0.0, 0.1, 0.2]
  
  label_smoothing:
    values: [0.0, 0.1, 0.2]

early_terminate:
  type: "hyperband"
  min_iter: 3
```

### 3. 실행 스크립트 활용

Windows 사용자를 위한 배치 파일:
```batch
# scripts/run_hyperparameter_sweep.bat
cd ..
python sweep_runner.py ^
    --base-config config\base_config.yaml ^
    --sweep-config hyperparameter_sweep ^
    --count 50
```

Linux/Mac 사용자를 위한 쉘 스크립트:
```bash
# scripts/run_hyperparameter_sweep.sh
#!/bin/bash
python sweep_runner.py \
    --base-config ../config/base_config.yaml \
    --sweep-config hyperparameter_sweep \
    --count 50
```

### 4. 고급 기능 활용

#### 기존 Sweep 재개하기
```bash
python sweep_runner.py \
    --base-config config/base_config.yaml \
    --sweep-config hyperparameter_sweep \
    --sweep-id "your-sweep-id" \
    --resume \
    --count 20
```

#### 환경변수로 설정 오버라이드
```bash
export LEARNING_RATE=1e-4
export BATCH_SIZE=8
python trainer.py --config config/base_config.yaml
```

---

## 실험 결과 분석

### 1. WandB 대시보드 활용

1. [wandb.ai](https://wandb.ai)에 접속
2. 프로젝트 선택
3. Sweep 탭에서 결과 확인

주요 확인 사항:
- **Parallel Coordinates Plot**: 파라미터 조합과 성능의 관계
- **Importance Plot**: 각 파라미터의 중요도
- **Best Runs**: 최고 성능 실행 목록

### 2. 로컬 결과 파일 분석

#### 결과 파일 구조
```
outputs/
├── sweep_hyperparameter_sweep/
│   ├── sweep_info.json           # Sweep 정보
│   ├── all_sweep_results.jsonl   # 모든 실행 결과
│   └── sweep_summary.json        # 요약 통계
└── experiments/
    └── 20240726_143052_a1b2c3d4/
        ├── training_results.json  # 학습 결과
        └── models/best_model/     # 저장된 모델
```

#### 최고 성능 모델 찾기
```python
import json

# Sweep 요약 파일 읽기
with open('outputs/sweep_hyperparameter_sweep/sweep_summary.json', 'r') as f:
    summary = json.load(f)

print(f"Best ROUGE Combined F1: {summary['best_rouge_combined_f1']:.4f}")
print(f"Best Model Path: {summary['best_model_path']}")
print("\nBest Hyperparameters:")
for param, value in summary['best_params'].items():
    print(f"  {param}: {value}")
```

### 3. 모델 로딩 및 사용

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# 최고 성능 모델 로딩
model_path = "outputs/experiments/.../models/best_model"
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 추론 실행
text = "대화 내용..."
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
outputs = model.generate(**inputs, max_length=100, num_beams=4)
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

---

## 문제 해결

### 🔧 일반적인 문제와 해결 방법

#### 1. GPU 메모리 부족 (CUDA Out of Memory)

**증상**: `RuntimeError: CUDA out of memory`

**해결 방법**:
```yaml
# 배치 크기 감소
training:
  per_device_train_batch_size: 8  # 16 → 8
  gradient_accumulation_steps: 2   # 추가하여 효과적인 배치 크기 유지

# FP16 활성화
training:
  fp16: true

# 시퀀스 길이 감소
tokenizer:
  encoder_max_len: 256  # 512 → 256
```

#### 2. WandB 연결 문제

**증상**: `wandb: ERROR Failed to connect to W&B servers`

**해결 방법**:
```bash
# 오프라인 모드로 실행
export WANDB_MODE=offline

# 또는 config에서 설정
wandb:
  mode: "offline"
```

#### 3. 느린 학습 속도

**증상**: 1 에폭에 2시간 이상 소요

**해결 방법**:
```yaml
# 데이터 로더 워커 증가
training:
  dataloader_num_workers: 8

# Mixed precision training
training:
  fp16: true
  fp16_opt_level: "O2"
```

#### 4. 실험 중단 후 재시작

**증상**: 학습이 중간에 중단됨

**해결 방법**:
```bash
# 체크포인트에서 재시작
python trainer.py \
    --config config/base_config.yaml \
    --resume-from-checkpoint outputs/experiments/.../checkpoints/checkpoint-500
```

### 🐛 디버깅 팁

1. **로그 파일 확인**
```bash
# 실험 로그
cat outputs/experiments/*/training.log

# Sweep 워커 로그
cat sweep_results/logs/worker_*.log
```

2. **설정 검증**
```python
from utils.config_manager import ConfigManager

cm = ConfigManager()
config = cm.load_config("config/base_config.yaml")
print(cm.validate_config_file("config/base_config.yaml"))
```

3. **GPU 사용률 모니터링**
```bash
# 실시간 GPU 모니터링
watch -n 1 nvidia-smi
```

---

## 자주 묻는 질문

### Q1: 기존 baseline.ipynb를 계속 사용할 수 있나요?
**A**: 네, 완전히 호환됩니다. 실험 자동화 시스템은 선택적으로 사용할 수 있으며, 기존 노트북과 병행 사용이 가능합니다.

### Q2: 어떤 상황에서 Sweep을 사용해야 하나요?
**A**: 
- 최적의 하이퍼파라미터를 찾고 싶을 때
- 여러 모델의 성능을 공정하게 비교하고 싶을 때
- 같은 실험을 여러 시드로 반복하고 싶을 때

### Q3: WandB 없이도 사용할 수 있나요?
**A**: 네, 가능합니다. config에서 `wandb.mode: "disabled"`로 설정하면 로컬에만 결과가 저장됩니다.

### Q4: 커스텀 메트릭을 추가하고 싶어요.
**A**: `utils/metrics.py`에 메트릭을 추가하고, `trainer.py`의 `compute_metrics` 함수를 수정하면 됩니다.

### Q5: 특정 GPU를 지정하고 싶어요.
**A**: 환경변수로 설정할 수 있습니다:
```bash
export CUDA_VISIBLE_DEVICES=0,1  # GPU 0, 1번만 사용
```

### Q6: 실험 결과를 팀원과 공유하고 싶어요.
**A**: 
1. WandB 팀 계정 사용 (추천)
2. `outputs/` 폴더를 압축하여 공유
3. 최고 성능 모델만 공유: `outputs/experiments/.../models/best_model/`

### Q7: 메모리가 부족한데 큰 모델을 써야 해요.
**A**: 
- Gradient checkpointing 활성화
- DeepSpeed 통합 (향후 지원 예정)
- 모델 양자화 사용

---

## 다음 단계

1. **기본 실험 실행**: 시스템에 익숙해지기 위해 간단한 실험부터 시작
2. **설정 가이드 읽기**: [configuration_guide.md](configuration_guide.md)에서 상세 설정 방법 확인
3. **베스트 프랙티스 학습**: [best_practices.md](best_practices.md)에서 효율적인 실험 방법 확인
4. **팀 표준 수립**: 팀 내 실험 명명 규칙, 파라미터 범위 등 표준화

---

## 기여하기

실험 자동화 시스템 개선에 기여하고 싶다면:

1. 버그 발견 시 이슈 생성
2. 새로운 기능 제안
3. 문서 개선
4. 코드 기여 (PR 환영!)

---

## 참고 자료

- [WandB 공식 문서](https://docs.wandb.ai)
- [Transformers 라이브러리 문서](https://huggingface.co/docs/transformers)

- [프로젝트 구조 확장 설계](../project_extension_design.md)

---

*이 문서는 지속적으로 업데이트됩니다. 최신 버전은 프로젝트 저장소에서 확인하세요.*
