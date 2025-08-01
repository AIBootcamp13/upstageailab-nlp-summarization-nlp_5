# Baseline.py vs Run_main_5_experiments.sh 상세 비교 분석

## 📋 목차
1. [개요](#1-개요)
2. [전체 아키텍처 비교](#2-전체-아키텍처-비교)
3. [Baseline.py 상세 분석](#3-baselinepy-상세-분석)
4. [Run_main_5_experiments.sh 파이프라인 분석](#4-run_main_5_experimentssh-파이프라인-분석)
5. [데이터 처리 흐름 비교](#5-데이터-처리-흐름-비교)
6. [모델 학습 과정 비교](#6-모델-학습-과정-비교)
7. [추론 및 결과 생성 비교](#7-추론-및-결과-생성-비교)
8. [주요 차이점 및 통합 방안](#8-주요-차이점-및-통합-방안)

---

## 1. 개요

### 1.1 Baseline.py
- **목적**: 대화 요약 모델의 기본 구현체
- **형태**: Jupyter Notebook에서 변환된 단일 Python 스크립트
- **특징**: 
  - 간단하고 직관적인 구조
  - KoBART 모델 기반
  - 단일 실험용 설계
  - 수동 설정 방식

### 1.2 Run_main_5_experiments.sh
- **목적**: 여러 모델을 자동으로 실험하는 고급 파이프라인
- **형태**: Bash 스크립트 + Python 모듈 시스템
- **특징**:
  - 자동화된 다중 실험 시스템
  - GPU 메모리 최적화
  - 다양한 모델 지원 (mT5, T5, KoBART 등)
  - 모듈화된 구조

---

## 2. 전체 아키텍처 비교

### 2.1 Baseline.py 아키텍처
```
baseline.py
├── 설정 생성 (YAML)
├── 데이터 로드
├── 전처리 클래스 (Preprocess)
├── Dataset 클래스들
├── 모델 로드
├── Trainer 설정
├── 학습 실행
└── 추론 및 결과 저장
```

### 2.2 Run_main_5_experiments.sh 아키텍처
```
run_main_5_experiments.sh
├── GPU 모니터링 함수들
├── 실험 목록 정의
├── 각 실험별 루프
│   ├── auto_experiment_runner.py 호출
│   │   ├── 설정 로드 (YAML)
│   │   ├── trainer.py 호출
│   │   │   ├── DataProcessor (utils/data_utils.py)
│   │   │   ├── 모델 로드 (다양한 모델 지원)
│   │   │   ├── 학습 실행
│   │   │   └── 메트릭 계산
│   │   └── post_training_inference.py
│   └── GPU 메모리 정리
└── 결과 요약
```

---

## 3. Baseline.py 상세 분석

### 3.1 데이터 처리 과정

#### 3.1.1 데이터 로드
```python
# baseline.py 라인 160-165
train_df = pd.read_csv(os.path.join(data_path,'train.csv'))
val_df = pd.read_csv(os.path.join(data_path,'dev.csv'))
```

#### 3.1.2 전처리 클래스
```python
# baseline.py 라인 174-203
class Preprocess:
    def __init__(self, bos_token: str, eos_token: str):
        self.bos_token = bos_token
        self.eos_token = eos_token
    
    def make_input(self, dataset, is_test=False):
        if is_test:
            # 테스트용: 대화만 인코더 입력으로
            encoder_input = dataset['dialogue']
            decoder_input = [self.bos_token] * len(dataset['dialogue'])
        else:
            # 학습용: 대화는 인코더, 요약은 디코더 입력으로
            encoder_input = dataset['dialogue']
            decoder_input = dataset['summary'].apply(lambda x: self.bos_token + str(x))
            decoder_output = dataset['summary'].apply(lambda x: str(x) + self.eos_token)
```

**개념 설명**:
- **BOS (Beginning of Sequence)**: 문장의 시작을 나타내는 특수 토큰
- **EOS (End of Sequence)**: 문장의 끝을 나타내는 특수 토큰
- **인코더-디코더 구조**: 
  - 인코더: 입력(대화)을 이해하는 부분
  - 디코더: 출력(요약)을 생성하는 부분

### 3.2 모델 구성

#### 3.2.1 모델 로드
```python
# baseline.py 라인 384-395
def load_tokenizer_and_model_for_train(config, device):
    model_name = config['general']['model_name']  # "digit82/kobart-summarization"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    generate_model = BartForConditionalGeneration.from_pretrained(model_name)
    
    # 특수 토큰 추가
    special_tokens_dict = {
        'additional_special_tokens': ['#Person1#', '#Person2#', '#Person3#', 
                                     '#PhoneNumber#', '#Address#', '#PassportNumber#']
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    generate_model.resize_token_embeddings(len(tokenizer))
```

**개념 설명**:
- **BART**: Bidirectional and Auto-Regressive Transformers
- **특수 토큰**: 개인정보를 마스킹하는 토큰들
- **토큰 임베딩 리사이즈**: 새 토큰 추가 시 모델 크기 조정

### 3.3 학습 설정

```python
# baseline.py 라인 321-350
training_args = Seq2SeqTrainingArguments(
    output_dir=config['general']['output_dir'],
    num_train_epochs=20,
    learning_rate=1e-5,
    per_device_train_batch_size=50,
    per_device_eval_batch_size=32,
    warmup_ratio=0.1,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    fp16=True,  # 16비트 부동소수점 사용 (메모리 절약)
    predict_with_generate=True,  # 생성 작업용
    generation_max_length=100
)
```

---

## 4. Run_main_5_experiments.sh 파이프라인 분석

### 4.1 실험 자동화 시스템

#### 4.1.1 GPU 메모리 모니터링
```bash
# run_main_5_experiments.sh 라인 42-80
enhanced_gpu_monitor() {
    local gpu_data=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu \
                     --format=csv,noheader,nounits)
    # GPU 상태 분석 및 경고
    if [ "$memory_used" -gt 22000 ]; then
        echo "⚠️ 경고: GPU 메모리 임계 상태 (22GB 초과)"
    fi
}
```

#### 4.1.2 스마트 대기 시스템
```bash
# run_main_5_experiments.sh 라인 83-112
smart_wait() {
    local target_memory=${1:-5000}  # 목표: 5GB 이하
    # GPU 메모리가 목표치 이하가 될 때까지 대기
    while [ "$current_memory" -gt "$target_memory" ]; do
        sleep 10
    done
}
```

### 4.2 실험 실행 흐름

#### 4.2.1 실험 목록 정의
```bash
# run_main_5_experiments.sh 라인 290-299
declare -a experiments=(
    01_mt5_xlsum_ultimate_korean_qlora.yaml:🚀_mT5_한국어_QLoRA_극한최적화:60분
    02_eenzeenee_t5_rtx3090.yaml:💪_eenzeenee_T5_RTX3090_극한최적화:40분
    01_baseline_kobart_rtx3090.yaml:💪_KoBART_RTX3090_극한최적화:45분
    # ...
)
```

#### 4.2.2 각 실험 실행
```bash
# run_main_5_experiments.sh 라인 382-440
for i in "${!experiments[@]}"; do
    # 1. GPU 상태 확인
    enhanced_gpu_monitor "실험 $EXPERIMENT_NUM 시작 전"
    
    # 2. auto_experiment_runner.py 실행
    EXPERIMENT_CMD="python code/auto_experiment_runner.py --config config/experiments/${config_file}"
    
    # 3. 실험 성공/실패 처리
    if eval "$EXPERIMENT_CMD > ${LOG_FILE} 2>&1"; then
        echo "✅ 실험 ${EXPERIMENT_NUM} 완료!"
    else
        handle_experiment_error "$exp_name" "$LOG_FILE" "$EXPERIMENT_NUM"
    fi
    
    # 4. GPU 정리 및 대기
    cleanup_gpu
    smart_wait 5000 240
done
```

### 4.3 Auto Experiment Runner 분석

#### 4.3.1 설정 병합 시스템
```python
# auto_experiment_runner.py 라인 186-208
def _load_and_merge_config(self, config_path: str) -> Dict[str, Any]:
    # 1. 기본 설정 로드
    base_config = load_config(self.base_config_path)
    
    # 2. 실험별 설정 로드
    exp_config = load_config(exp_config_path)
    
    # 3. 딥 머지 (실험 설정이 우선)
    merged = self._deep_merge(base_config, exp_config)
    return merged
```

#### 4.3.2 디바이스 최적화
```python
# auto_experiment_runner.py 라인 251-275
def _apply_device_config(self, config: Dict[str, Any]) -> None:
    # 모델 크기 추정
    if 'large' in model_name or 'xl' in model_name:
        model_size = 'large'
    
    # 최적화 설정 생성
    opt_config = setup_device_config(self.device_info, model_size)
    
    # 디바이스별 최적 배치 크기, gradient accumulation 등 설정
```

---

## 5. 데이터 처리 흐름 비교

### 5.1 Baseline.py 데이터 처리
```
train.csv → Pandas DataFrame → Preprocess 클래스 → Tokenizer → Dataset 클래스 → DataLoader
```

### 5.2 Pipeline 데이터 처리
```
train.csv → DataProcessor (utils/data_utils.py) → 
├── 다양한 전처리 옵션 (노이즈 제거, 정규화 등)
├── 데이터 증강 옵션
├── 동적 토큰화 (모델별 최적화)
└── HuggingFace Dataset 형식으로 변환
```

### 5.3 주요 차이점

#### Baseline.py:
```python
# 간단한 전처리
encoder_input = dataset['dialogue']
decoder_input = dataset['summary'].apply(lambda x: self.bos_token + str(x))
```

#### Pipeline (DataProcessor):
```python
# utils/data_utils.py의 고급 전처리
def preprocess_dialogue(self, text: str) -> str:
    # 1. HTML 태그 제거
    text = re.sub(r'<[^>]+>', '', text)
    
    # 2. 특수 문자 정규화
    text = self._normalize_whitespace(text)
    
    # 3. 개인정보 마스킹
    text = self._mask_personal_info(text)
    
    # 4. 대화 구조 파싱
    text = self._parse_dialogue_structure(text)
    
    return text
```

---

## 6. 모델 학습 과정 비교

### 6.1 Baseline.py 학습

#### 6.1.1 단일 모델 학습
```python
# baseline.py 라인 361-373
trainer = Seq2SeqTrainer(
    model=generate_model,
    args=training_args,
    train_dataset=train_inputs_dataset,
    eval_dataset=val_inputs_dataset,
    compute_metrics=lambda pred: compute_metrics(config, tokenizer, pred),
    callbacks=[EarlyStoppingCallback()]
)
trainer.train()
```

### 6.2 Pipeline 학습

#### 6.2.1 다중 모델 지원 (trainer.py)
```python
# trainer.py의 모델 로드 로직
def load_model(self, config: Dict[str, Any]):
    model_name = config['general']['model_name']
    
    if 'mt5' in model_name.lower():
        # mT5 모델 로드
        model = self._load_mt5_model(config)
    elif 't5' in model_name.lower():
        # T5 모델 로드
        model = self._load_t5_model(config)
    elif 'bart' in model_name.lower():
        # BART 모델 로드
        model = self._load_bart_model(config)
    elif 'solar' in model_name.lower():
        # Solar 모델 로드 (Causal LM)
        model = self._load_solar_model(config)
    
    # QLoRA 적용 (선택적)
    if config.get('use_qlora', False):
        model = self._apply_qlora(model, config)
    
    return model
```

#### 6.2.2 고급 학습 기능
```python
# trainer.py의 학습 최적화
class DialogueSummarizationTrainer:
    def train(self):
        # 1. Gradient Checkpointing (메모리 절약)
        if self.config.get('gradient_checkpointing', False):
            self.model.gradient_checkpointing_enable()
        
        # 2. Mixed Precision Training (속도 향상)
        if self.config.get('fp16', False) or self.config.get('bf16', False):
            self._setup_mixed_precision()
        
        # 3. DeepSpeed 통합 (대규모 모델)
        if self.config.get('deepspeed'):
            self._setup_deepspeed()
        
        # 4. 동적 배치 크기 조정
        if self.config.get('auto_find_batch_size', False):
            self._find_optimal_batch_size()
```

---

## 7. 추론 및 결과 생성 비교

### 7.1 Baseline.py 추론

```python
# baseline.py 라인 499-523
def inference(config):
    # 1. 모델 로드
    generate_model, tokenizer = load_tokenizer_and_model_for_test(config, device)
    
    # 2. 테스트 데이터 준비
    test_data, test_encoder_inputs_dataset = prepare_test_dataset(config, preprocessor, tokenizer)
    dataloader = DataLoader(test_encoder_inputs_dataset, batch_size=32)
    
    # 3. 추론 실행
    summary = []
    with torch.no_grad():
        for item in tqdm(dataloader):
            generated_ids = generate_model.generate(
                input_ids=item['input_ids'].to('cuda:0'),
                no_repeat_ngram_size=2,
                early_stopping=True,
                max_length=100,
                num_beams=4
            )
            for ids in generated_ids:
                result = tokenizer.decode(ids)
                summary.append(result)
    
    # 4. 결과 저장
    output = pd.DataFrame({
        "fname": test_data['fname'],
        "summary": preprocessed_summary
    })
    output.to_csv(os.path.join(result_path, "output.csv"), index=False)
```

### 7.2 Pipeline 추론

#### 7.2.1 자동 추론 시스템 (auto_experiment_runner.py)
```python
# auto_experiment_runner.py 라인 420-470
# 학습 완료 후 자동으로 test.csv 추론 수행
try:
    from post_training_inference import generate_submission_after_training
    
    submission_path = generate_submission_after_training(
        experiment_name=experiment_id,
        model_path=str(best_checkpoint),
        config_dict=config
    )
except:
    # 대안: run_inference.py 직접 사용
    inference_cmd = [
        sys.executable,
        "code/run_inference.py",
        "--model_path", str(best_checkpoint),
        "--input_file", "data/test.csv",
        "--output_file", f"outputs/{experiment_id}_submission.csv"
    ]
```

#### 7.2.2 고급 생성 전략 (run_inference.py)
```python
# run_inference.py의 생성 전략
def generate_summary(self, batch_texts: List[str]) -> List[str]:
    # 1. 다양한 생성 전략 지원
    if self.generation_strategy == 'beam_search':
        outputs = self._beam_search_generate(batch_texts)
    elif self.generation_strategy == 'sampling':
        outputs = self._sampling_generate(batch_texts)
    elif self.generation_strategy == 'contrastive':
        outputs = self._contrastive_generate(batch_texts)
    
    # 2. 후처리
    summaries = self._postprocess_summaries(outputs)
    
    return summaries
```

---

## 8. 주요 차이점 및 통합 방안

### 8.1 주요 차이점 요약

| 항목 | Baseline.py | Pipeline |
|------|------------|----------|
| **구조** | 단일 스크립트 | 모듈화된 시스템 |
| **모델** | KoBART만 지원 | 다양한 모델 지원 |
| **실험** | 수동 실행 | 자동화된 다중 실험 |
| **GPU 관리** | 기본적 | 고급 메모리 최적화 |
| **데이터 처리** | 간단한 전처리 | 고급 전처리 및 증강 |
| **설정 관리** | 하드코딩 | YAML 기반 유연한 설정 |
| **추론** | 수동 실행 필요 | 학습 후 자동 실행 |
| **결과 추적** | 기본적 | WandB, CSV, JSON 등 다양한 형식 |

### 8.2 통합 방안

#### 8.2.1 Baseline 로직을 Pipeline에 통합
```python
# config/experiments/00_baseline_exact.yaml
general:
  model_name: "digit82/kobart-summarization"
  experiment_name: "baseline_exact_reproduction"

tokenizer:
  encoder_max_len: 512
  decoder_max_len: 100
  special_tokens: ['#Person1#', '#Person2#', '#Person3#', 
                   '#PhoneNumber#', '#Address#', '#PassportNumber#']

training:
  num_train_epochs: 20
  learning_rate: 1e-5
  per_device_train_batch_size: 50
  per_device_eval_batch_size: 32
  warmup_ratio: 0.1
  weight_decay: 0.01
  lr_scheduler_type: 'cosine'
  fp16: true
  evaluation_strategy: 'epoch'
  save_strategy: 'epoch'
  early_stopping_patience: 3

inference:
  no_repeat_ngram_size: 2
  early_stopping: true
  max_length: 100
  num_beams: 4
  batch_size: 32
```

#### 8.2.2 데이터 처리 통합
```python
# utils/data_utils.py에 baseline 모드 추가
class DataProcessor:
    def __init__(self, config: Dict[str, Any], baseline_mode: bool = False):
        self.baseline_mode = baseline_mode
        
    def preprocess_dialogue(self, text: str) -> str:
        if self.baseline_mode:
            # Baseline과 동일한 간단한 처리
            return text
        else:
            # 고급 전처리
            return self._advanced_preprocess(text)
```

### 8.3 실행 명령어 비교

#### Baseline.py 실행:
```bash
python code/baseline.py
```

#### Pipeline 실행:
```bash
# 전체 실험 실행
bash run_main_5_experiments.sh

# 특정 실험만 실행
python code/auto_experiment_runner.py --config config/experiments/01_baseline.yaml

# 1에포크 빠른 테스트
bash run_main_5_experiments.sh -1
```

### 8.4 결과 파일 구조

#### Baseline.py:
```
outputs/
└── prediction/
    └── output.csv
```

#### Pipeline:
```
outputs/
├── auto_experiments/
│   ├── experiments/          # 실험별 상세 결과
│   ├── models/              # 저장된 모델들
│   ├── csv_results/         # CSV 형식 결과
│   └── experiment_summary.json
├── checkpoints/             # 학습 체크포인트
└── submissions/            # 제출용 CSV 파일
```

---

## 결론

1. **Baseline.py**는 단순하고 이해하기 쉬운 구조로, 빠른 프로토타이핑에 적합합니다.

2. **Pipeline**은 대규모 실험과 프로덕션 환경에 적합한 고급 기능을 제공합니다.

3. 두 시스템은 **동일한 데이터 흐름**을 따르므로, Pipeline에서 Baseline과 완전히 동일한 결과를 재현할 수 있습니다.

4. **통합 권장사항**:
   - 초기 개발: Baseline.py로 빠른 검증
   - 본격 실험: Pipeline으로 다양한 모델/설정 테스트
   - 최종 제출: Pipeline의 best 모델 사용

5. **test.csv 처리**는 두 시스템 모두 동일한 형식으로 수행되며, Pipeline은 학습 완료 후 자동으로 추론까지 실행합니다.
