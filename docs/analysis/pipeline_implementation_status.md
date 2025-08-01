# Run_main_5_experiments.sh의 Baseline.py 기능 구현 완전 분석

## 📌 핵심 요약

**run_main_5_experiments.sh는 이미 baseline.py의 모든 기능을 포함하고 있으며, 훨씬 더 강력한 기능을 제공합니다.**

---

## 1. 데이터 로딩 구현 상태 ✅

### Baseline.py 방식
```python
# baseline.py 라인 160-165
train_df = pd.read_csv(os.path.join(data_path,'train.csv'))
val_df = pd.read_csv(os.path.join(data_path,'dev.csv'))
```

### Pipeline 구현 (이미 구현됨)
```python
# utils/data_utils.py의 DataProcessor 클래스
class DataProcessor:
    def load_dataset(self, data_path: str, split: str = 'train'):
        """데이터셋 로드 - baseline과 동일한 경로 사용"""
        if split == 'train':
            df = pd.read_csv(os.path.join(data_path, 'train.csv'))
        elif split == 'dev' or split == 'validation':
            df = pd.read_csv(os.path.join(data_path, 'dev.csv'))
        elif split == 'test':
            df = pd.read_csv(os.path.join(data_path, 'test.csv'))
        return df
```

**위치**: `code/utils/data_utils.py`의 `DataProcessor.load_dataset()` 메서드

---

## 2. 모델 학습 구현 상태 ✅

### Baseline.py 방식
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

### Pipeline 구현 (이미 구현됨)
```python
# trainer.py의 DialogueSummarizationTrainer 클래스
def train(self):
    """모델 학습 - baseline과 동일한 Seq2SeqTrainer 사용"""
    trainer = Seq2SeqTrainer(
        model=self.model,
        args=self.training_args,
        train_dataset=self.train_dataset,
        eval_dataset=self.eval_dataset,
        tokenizer=self.tokenizer,
        data_collator=self.data_collator,
        compute_metrics=self.compute_metrics,
        callbacks=self.callbacks
    )
    
    # 학습 실행
    trainer.train()
    
    # 베스트 모델 저장
    trainer.save_model()
```

**위치**: `code/trainer.py`의 `DialogueSummarizationTrainer.train()` 메서드

---

## 3. Test.csv 추론 구현 상태 ✅

### Baseline.py 방식
```python
# baseline.py 라인 499-542
def inference(config):
    # test.csv 로드
    test_data = preprocessor.make_set_as_df(test_file_path, is_train=False)
    
    # 배치 추론
    for item in tqdm(dataloader):
        generated_ids = generate_model.generate(
            input_ids=item['input_ids'],
            num_beams=4,
            max_length=100
        )
```

### Pipeline 구현 (이미 구현됨)

#### 3.1 자동 추론 트리거 (auto_experiment_runner.py)
```python
# auto_experiment_runner.py 라인 420-470
if process.returncode == 0:  # 학습 성공 시
    print(f"\n📊 Test 추론 시작: {experiment_id}")
    
    # post_training_inference 활용
    from post_training_inference import generate_submission_after_training
    
    submission_path = generate_submission_after_training(
        experiment_name=experiment_id,
        model_path=str(best_checkpoint),
        config_dict=config
    )
```

#### 3.2 실제 추론 구현 (post_training_inference.py)
```python
def generate_submission_after_training(experiment_name, model_path, config_dict):
    # 1. test.csv 로드
    test_df = pd.read_csv('data/test.csv')
    
    # 2. 모델 로드
    model, tokenizer = load_trained_model(model_path, config_dict)
    
    # 3. 배치 추론 (baseline과 동일한 방식)
    for i in range(0, len(test_df), batch_size):
        batch = test_df.iloc[i:i+batch_size]
        outputs = model.generate(
            inputs,
            max_length=config_dict.get('generation', {}).get('max_length', 100),
            num_beams=config_dict.get('generation', {}).get('num_beams', 4),
            no_repeat_ngram_size=2,
            early_stopping=True
        )
```

**위치**: 
- `code/auto_experiment_runner.py`의 `_run_single_experiment()` 메서드
- `code/post_training_inference.py`의 `generate_submission_after_training()` 함수

---

## 4. 결과 CSV 생성 구현 상태 ✅

### Baseline.py 방식
```python
# baseline.py 라인 544-560
output = pd.DataFrame({
    "fname": test_data['fname'],
    "summary": preprocessed_summary
})
output.to_csv(os.path.join(result_path, "output.csv"), index=False)
```

### Pipeline 구현 (이미 구현됨)
```python
# post_training_inference.py와 csv_results_saver.py
def save_submission(self, experiment_name, test_df, summaries):
    """제출용 CSV 파일 생성 - baseline과 동일한 형식"""
    submission_df = pd.DataFrame({
        'fname': test_df['fname'],
        'summary': summaries
    })
    
    # 실험별 고유 파일명
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f'outputs/submissions/{experiment_name}_{timestamp}.csv'
    
    submission_df.to_csv(output_path, index=False)
    return output_path
```

**위치**: `code/utils/csv_results_saver.py`의 `CSVResultsSaver` 클래스

---

## 5. 실제 실행 흐름 (전체 통합)

### 5.1 run_main_5_experiments.sh 실행 시
```bash
#!/bin/bash
# 1. GPU 상태 확인
enhanced_gpu_monitor "실험 전"

# 2. 각 실험 실행 (예: KoBART baseline)
for experiment in "${experiments[@]}"; do
    # 3. auto_experiment_runner.py 호출
    python code/auto_experiment_runner.py \
        --config config/experiments/01_baseline_kobart_rtx3090.yaml
    
    # 이 명령은 다음을 수행:
    # a) trainer.py를 통해 모델 학습 (train.csv 사용)
    # b) 학습 완료 후 자동으로 test.csv 추론
    # c) 결과 CSV 생성 및 저장
done
```

### 5.2 데이터 흐름
```
1. data/train.csv → DataProcessor → 모델 학습
2. data/dev.csv → 평가 및 조기 종료
3. 학습 완료 → 베스트 체크포인트 저장
4. data/test.csv → 자동 추론 실행
5. outputs/submissions/실험명_timestamp.csv 생성
```

---

## 6. Baseline과 완전히 동일한 결과를 얻는 방법

### 6.1 설정 파일 준비
```yaml
# config/experiments/baseline_exact_reproduction.yaml
experiment_name: baseline_exact_reproduction

general:
  model_name: digit82/kobart-summarization
  data_path: data/

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
  fp16: true
  evaluation_strategy: epoch
  save_strategy: epoch
  early_stopping_patience: 3

generation:
  max_length: 100
  num_beams: 4
  no_repeat_ngram_size: 2
  early_stopping: true

inference:
  batch_size: 32
  remove_tokens: ['<usr>', '<s>', '</s>', '<pad>']
```

### 6.2 실행 명령
```bash
# 단일 실험 실행
python code/auto_experiment_runner.py \
    --config config/experiments/baseline_exact_reproduction.yaml

# 또는 run_main_5_experiments.sh에 추가하여 실행
```

---

## 7. Pipeline의 추가 기능들

### 7.1 이미 구현된 고급 기능들
1. **자동 GPU 메모리 관리**: GPU 상태 모니터링 및 최적화
2. **다중 모델 지원**: mT5, T5, KoBART, Solar 등
3. **QLoRA/Unsloth 최적화**: 메모리 효율적 학습
4. **WandB 통합**: 실험 추적 및 시각화
5. **자동 하이퍼파라미터 최적화**: Sweep 기능
6. **병렬 실험 실행**: 여러 GPU에서 동시 실행
7. **실험 결과 자동 분석**: 메트릭 비교 및 베스트 모델 선택

### 7.2 Baseline에는 없는 Pipeline 기능
```python
# 1. 데이터 증강
augmentation:
  use_augmentation: true
  augmentation_types: ['backtranslation', 'paraphrase', 'noise']

# 2. 앙상블
ensemble:
  models: ['model1', 'model2', 'model3']
  strategy: 'weighted_average'

# 3. 고급 생성 전략
generation:
  strategy: 'contrastive_search'
  top_k: 50
  penalty_alpha: 0.6
```

---

## 결론

**run_main_5_experiments.sh는 이미 baseline.py의 모든 기능을 완벽하게 구현하고 있습니다:**

1. ✅ **데이터 로딩**: `data/train.csv`, `data/dev.csv`, `data/test.csv` 모두 사용
2. ✅ **모델 학습**: 동일한 Seq2SeqTrainer 기반
3. ✅ **Test 추론**: 학습 완료 후 자동으로 실행
4. ✅ **결과 CSV 생성**: 동일한 형식의 submission 파일 생성

**추가로 제공하는 기능:**
- 🚀 자동화된 실험 관리
- 💪 GPU 메모리 최적화
- 📊 실험 결과 추적 및 분석
- 🔥 다양한 모델 및 최적화 기법 지원

**사용 권장사항:**
- 빠른 프로토타입: `python code/baseline.py`
- 본격적인 실험: `bash run_main_5_experiments.sh`
- 최종 제출: Pipeline의 베스트 모델 사용
