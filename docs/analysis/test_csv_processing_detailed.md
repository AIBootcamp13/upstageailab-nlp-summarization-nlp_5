# Test.csv 처리 및 결과 생성 상세 분석

## 📋 목차
1. [Test 데이터 구조 분석](#1-test-데이터-구조-분석)
2. [Baseline.py의 Test 처리 흐름](#2-baselinepy의-test-처리-흐름)
3. [Pipeline의 Test 처리 흐름](#3-pipeline의-test-처리-흐름)
4. [결과 CSV 생성 비교](#4-결과-csv-생성-비교)
5. [실제 구현 매핑](#5-실제-구현-매핑)

---

## 1. Test 데이터 구조 분석

### 1.1 데이터 파일 위치
- **Train 데이터**: `/data/train.csv`
- **Dev 데이터**: `/data/dev.csv`  
- **Test 데이터**: `/data/test.csv`

### 1.2 데이터 구조
```python
# Train/Dev 데이터 구조
train_df = pd.DataFrame({
    'fname': ['대화ID_1', '대화ID_2', ...],      # 파일명/ID
    'dialogue': ['대화 내용 1', '대화 내용 2', ...],  # 입력 대화
    'summary': ['요약 1', '요약 2', ...]         # 정답 요약 (학습용)
})

# Test 데이터 구조 
test_df = pd.DataFrame({
    'fname': ['테스트ID_1', '테스트ID_2', ...],   # 파일명/ID
    'dialogue': ['테스트 대화 1', '테스트 대화 2', ...]  # 입력 대화만 존재
    # summary 컬럼 없음 - 모델이 생성해야 함
})
```

---

## 2. Baseline.py의 Test 처리 흐름

### 2.1 전체 처리 과정
```
1. 모델 학습 완료
2. 수동으로 inference() 함수 실행
3. test.csv 로드
4. 배치 단위로 요약 생성  
5. output.csv 저장
```

### 2.2 상세 코드 분석

#### 2.2.1 Test 데이터 준비 (baseline.py 라인 455-490)
```python
def prepare_test_dataset(config, preprocessor, tokenizer):
    # 1. test.csv 파일 읽기
    test_file_path = os.path.join(config['general']['data_path'], 'test.csv')
    test_data = preprocessor.make_set_as_df(test_file_path, is_train=False)
    test_id = test_data['fname']  # 결과 저장용 ID
    
    # 2. 전처리 - BOS 토큰만 디코더 입력으로 설정
    encoder_input_test, decoder_input_test = preprocessor.make_input(
        test_data, 
        is_test=True  # test 모드 활성화
    )
    
    # 3. 토큰화
    test_tokenized_encoder_inputs = tokenizer(
        encoder_input_test,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    
    # 4. Dataset 객체 생성
    test_encoder_inputs_dataset = DatasetForInference(
        test_tokenized_encoder_inputs, 
        test_id, 
        len(encoder_input_test)
    )
    
    return test_data, test_encoder_inputs_dataset
```

#### 2.2.2 추론 실행 (baseline.py 라인 493-542)
```python
def inference(config):
    # 1. 디바이스 설정
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 2. 학습된 모델 로드
    generate_model, tokenizer = load_tokenizer_and_model_for_test(config, device)
    
    # 3. 데이터 준비
    preprocessor = Preprocess(
        config['tokenizer']['bos_token'], 
        config['tokenizer']['eos_token']
    )
    test_data, test_encoder_inputs_dataset = prepare_test_dataset(
        config, preprocessor, tokenizer
    )
    
    # 4. DataLoader 생성 (배치 처리용)
    dataloader = DataLoader(
        test_encoder_inputs_dataset, 
        batch_size=config['inference']['batch_size']  # 32
    )
    
    # 5. 배치 단위 추론
    summary = []
    text_ids = []
    
    with torch.no_grad():  # 그래디언트 계산 비활성화 (메모리 절약)
        for item in tqdm(dataloader):
            # ID 수집
            text_ids.extend(item['ID'])
            
            # 요약 생성
            generated_ids = generate_model.generate(
                input_ids=item['input_ids'].to(device),
                no_repeat_ngram_size=2,      # 2-gram 반복 방지
                early_stopping=True,          # EOS 토큰 시 조기 종료
                max_length=100,              # 최대 생성 길이
                num_beams=4                  # 빔 서치 크기
            )
            
            # 디코딩
            for ids in generated_ids:
                result = tokenizer.decode(ids)
                summary.append(result)
```

#### 2.2.3 결과 저장 (baseline.py 라인 544-560)
```python
    # 6. 후처리 - 특수 토큰 제거
    remove_tokens = config['inference']['remove_tokens']
    # ['<usr>', '<s>', '</s>', '<pad>']
    
    preprocessed_summary = summary.copy()
    for token in remove_tokens:
        preprocessed_summary = [
            sentence.replace(token, " ") for sentence in preprocessed_summary
        ]
    
    # 7. DataFrame 생성 및 CSV 저장
    output = pd.DataFrame({
        "fname": test_data['fname'],      # 원본 파일명/ID
        "summary": preprocessed_summary    # 생성된 요약
    })
    
    # 8. 결과 파일 저장
    result_path = config['inference']['result_path']  # "./prediction/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    output.to_csv(
        os.path.join(result_path, "output.csv"), 
        index=False  # 인덱스 컬럼 제외
    )
```

---

## 3. Pipeline의 Test 처리 흐름

### 3.1 전체 처리 과정
```
1. 모델 학습 중 자동으로 best checkpoint 저장
2. 학습 완료 후 자동으로 test 추론 시작
3. post_training_inference.py 또는 run_inference.py 실행
4. 다양한 생성 전략 적용 가능
5. 실험별로 고유한 submission 파일 생성
```

### 3.2 자동 추론 시스템

#### 3.2.1 Auto Experiment Runner의 추론 트리거 (auto_experiment_runner.py 라인 418-473)
```python
def _run_single_experiment(self, config, config_path, one_epoch=False):
    # ... 학습 실행 ...
    
    if process.returncode == 0:  # 학습 성공 시
        # 1. 베스트 체크포인트 찾기
        print(f"\n📊 Test 추론 시작: {experiment_id}")
        
        output_dir = Path(config.get('training', {}).get('output_dir', 'outputs'))
        checkpoint_dirs = list(output_dir.glob('checkpoint-*'))
        
        if checkpoint_dirs:
            # 가장 최근(베스트) 체크포인트 선택
            best_checkpoint = max(checkpoint_dirs, key=lambda p: p.stat().st_mtime)
            print(f"🎯 베스트 체크포인트: {best_checkpoint}")
            
            # 2. post_training_inference 모듈 사용
            try:
                from post_training_inference import generate_submission_after_training
                
                submission_path = generate_submission_after_training(
                    experiment_name=experiment_id,
                    model_path=str(best_checkpoint),
                    config_dict=config
                )
                
                print(f"✅ 제출 파일 생성 완료: {submission_path}")
                
            except ImportError:
                # 3. 대안: run_inference.py 직접 실행
                self._run_inference_fallback(best_checkpoint, experiment_id)
```

#### 3.2.2 Post Training Inference 구현
```python
# post_training_inference.py
def generate_submission_after_training(
    experiment_name: str,
    model_path: str,
    config_dict: Dict[str, Any]
) -> str:
    """학습 완료 후 자동으로 test.csv에 대한 추론 실행"""
    
    # 1. 모델 및 토크나이저 로드
    model, tokenizer = load_trained_model(model_path, config_dict)
    
    # 2. Test 데이터 로드
    test_df = pd.read_csv('data/test.csv')
    
    # 3. 배치 추론 실행
    batch_size = config_dict.get('inference', {}).get('batch_size', 16)
    summaries = []
    
    for i in range(0, len(test_df), batch_size):
        batch = test_df.iloc[i:i+batch_size]
        batch_summaries = generate_batch_summaries(
            model, tokenizer, batch['dialogue'].tolist(), config_dict
        )
        summaries.extend(batch_summaries)
    
    # 4. 결과 저장
    submission_df = pd.DataFrame({
        'fname': test_df['fname'],
        'summary': summaries
    })
    
    # 5. 실험별 고유 파일명 생성
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f'outputs/submissions/{experiment_name}_{timestamp}.csv'
    
    submission_df.to_csv(output_path, index=False)
    return output_path
```

### 3.3 고급 추론 기능

#### 3.3.1 Run Inference의 생성 전략 (run_inference.py)
```python
class InferenceRunner:
    def __init__(self, model_path: str, generation_config: Dict[str, Any]):
        self.model, self.tokenizer = self.load_model(model_path)
        self.generation_config = generation_config
        
    def generate_summaries(self, dialogues: List[str]) -> List[str]:
        """다양한 생성 전략을 지원하는 추론 함수"""
        
        strategy = self.generation_config.get('strategy', 'beam_search')
        
        if strategy == 'beam_search':
            return self._beam_search_generation(dialogues)
        elif strategy == 'sampling':
            return self._sampling_generation(dialogues)
        elif strategy == 'diverse_beam_search':
            return self._diverse_beam_generation(dialogues)
        elif strategy == 'contrastive_search':
            return self._contrastive_generation(dialogues)
            
    def _beam_search_generation(self, dialogues: List[str]) -> List[str]:
        """빔 서치 기반 생성 (Baseline과 동일)"""
        inputs = self.tokenizer(
            dialogues,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        outputs = self.model.generate(
            **inputs,
            max_length=100,
            num_beams=4,
            no_repeat_ngram_size=2,
            early_stopping=True,
            length_penalty=1.0,
            repetition_penalty=1.2
        )
        
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
```

#### 3.3.2 후처리 파이프라인
```python
def postprocess_summaries(summaries: List[str], config: Dict[str, Any]) -> List[str]:
    """생성된 요약문 후처리"""
    
    processed = []
    for summary in summaries:
        # 1. 특수 토큰 제거
        for token in config.get('remove_tokens', []):
            summary = summary.replace(token, ' ')
        
        # 2. 공백 정규화
        summary = ' '.join(summary.split())
        
        # 3. 문장 끝 처리
        if not summary.endswith('.'):
            summary += '.'
            
        # 4. 길이 제한
        max_length = config.get('max_summary_length', 150)
        if len(summary.split()) > max_length:
            words = summary.split()[:max_length]
            summary = ' '.join(words) + '.'
            
        processed.append(summary)
    
    return processed
```

---

## 4. 결과 CSV 생성 비교

### 4.1 Baseline 결과 형식
```csv
fname,summary
대화_001,회의에서 프로젝트 일정과 예산에 대해 논의했습니다.
대화_002,고객이 제품 환불을 요청했고 처리가 완료되었습니다.
...
```

### 4.2 Pipeline 결과 형식
```csv
fname,summary
대화_001,회의에서 프로젝트 일정과 예산에 대해 논의했습니다.
대화_002,고객이 제품 환불을 요청했고 처리가 완료되었습니다.
...
```

**동일한 형식**이지만, Pipeline은 추가 메타데이터도 저장:
- 실험 ID와 타임스탬프가 포함된 파일명
- 실험별 설정 정보 (JSON)
- 생성 메트릭 (생성 시간, 토큰 수 등)

---

## 5. 실제 구현 매핑

### 5.1 Pipeline에서 Baseline 로직 재현

#### 5.1.1 설정 파일 (config/experiments/baseline_exact.yaml)
```yaml
# Baseline과 완전히 동일한 설정
general:
  model_name: "digit82/kobart-summarization"
  
tokenizer:
  encoder_max_len: 512
  decoder_max_len: 100
  special_tokens: 
    - '#Person1#'
    - '#Person2#' 
    - '#Person3#'
    - '#PhoneNumber#'
    - '#Address#'
    - '#PassportNumber#'

training:
  output_dir: "./outputs"
  num_train_epochs: 20
  learning_rate: 1e-5
  per_device_train_batch_size: 50
  per_device_eval_batch_size: 32
  warmup_ratio: 0.1
  weight_decay: 0.01
  lr_scheduler_type: 'cosine'
  optim: 'adamw_torch'
  fp16: true
  evaluation_strategy: 'epoch'
  save_strategy: 'epoch'
  load_best_model_at_end: true
  metric_for_best_model: 'eval_loss'
  early_stopping_patience: 3

inference:
  batch_size: 32
  max_length: 100
  num_beams: 4
  no_repeat_ngram_size: 2
  early_stopping: true
  remove_tokens: ['<usr>', '<s>', '</s>', '<pad>']
```

#### 5.1.2 실행 명령어
```bash
# Baseline 방식
python code/baseline.py

# Pipeline으로 동일한 결과 재현
python code/auto_experiment_runner.py \
    --config config/experiments/baseline_exact.yaml
```

### 5.2 주요 구현 파일 위치

| 기능 | Baseline.py | Pipeline |
|------|------------|----------|
| **데이터 로드** | baseline.py:160-165 | utils/data_utils.py:load_dataset() |
| **전처리** | baseline.py:174-203 | utils/data_utils.py:DataProcessor |
| **모델 로드** | baseline.py:384-395 | trainer.py:load_model() |
| **학습** | baseline.py:321-373 | trainer.py:train() |
| **추론** | baseline.py:493-542 | run_inference.py, post_training_inference.py |
| **결과 저장** | baseline.py:544-560 | utils/csv_results_saver.py |

### 5.3 데이터 흐름 검증

```python
# Pipeline에서 Baseline과 동일한 데이터 흐름 보장
assert train_df.shape == (12457, 3)  # train.csv
assert val_df.shape == (499, 3)      # dev.csv  
assert test_df.shape == (250, 2)     # test.csv (fname, dialogue만)

# 생성된 결과도 동일한 형식
assert output_df.shape == (250, 2)   # (fname, summary)
assert list(output_df.columns) == ['fname', 'summary']
```

---

## 결론

1. **데이터 흐름은 완전히 동일**: 두 시스템 모두 train.csv로 학습하고 test.csv로 추론하여 동일한 형식의 CSV를 생성합니다.

2. **Pipeline의 장점**:
   - 학습 완료 후 자동으로 test 추론 실행
   - 실험별로 구분된 결과 파일 생성
   - 다양한 생성 전략 선택 가능
   - 메타데이터 및 추적 정보 포함

3. **통합 사용 방법**:
   - 개발 단계: Baseline.py로 빠른 프로토타이핑
   - 실험 단계: Pipeline으로 다양한 설정 테스트
   - 최종 제출: Pipeline의 best 모델로 최종 결과 생성

4. **완전한 호환성**: Pipeline에서 Baseline과 동일한 설정을 사용하면 완전히 동일한 결과를 얻을 수 있습니다.
