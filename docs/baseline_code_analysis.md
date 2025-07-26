# Dialogue Summarization 베이스라인 코드 상세 분석

## 목차
1. [코드 개요](#1-코드-개요)
2. [환경 설정 및 라이브러리](#2-환경-설정-및-라이브러리)
3. [Configuration 설정](#3-configuration-설정)
4. [데이터 전처리](#4-데이터-전처리)
5. [데이터셋 클래스](#5-데이터셋-클래스)
6. [모델 학습 설정](#6-모델-학습-설정)
7. [모델 추론](#7-모델-추론)
8. [주요 개념 설명](#8-주요-개념-설명)

---

## 1. 코드 개요

이 베이스라인 코드는 **BART(Bidirectional and Auto-Regressive Transformers)** 모델을 사용하여 일상 대화를 요약하는 시스템을 구현합니다. 주요 특징은:

- **모델**: KoBART (한국어 특화 BART 모델)
- **프레임워크**: Hugging Face Transformers + PyTorch
- **학습 관리**: Weights & Biases (wandb)
- **평가 지표**: ROUGE (ROUGE-1, ROUGE-2, ROUGE-L)

## 2. 환경 설정 및 라이브러리

### 필수 라이브러리 설치 (코드 위치: 셀 3)

> 💡 **팁**: 패키지 설치가 느리다면 `uv`를 사용해보세요! 
> ```bash
> pip install uv
> uv pip install -r requirements.txt  # 10배 이상 빠름!
> ```
> 자세한 내용은 [uv 패키지 관리자 가이드](uv_package_manager_guide.md) 참고

```python
import pandas as pd              # 데이터 처리
import os                       # 파일 경로 관리
import re                       # 정규표현식
import json                     # JSON 데이터 처리
import yaml                     # YAML 설정 파일 처리
from glob import glob           # 파일 패턴 매칭
from tqdm import tqdm           # 진행 상황 표시
from pprint import pprint       # 예쁜 출력
import torch                    # PyTorch 딥러닝 프레임워크
import pytorch_lightning as pl  # PyTorch Lightning
from rouge import Rouge         # ROUGE 평가 지표

# Hugging Face Transformers 라이브러리
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BartForConditionalGeneration, BartConfig
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback

import wandb  # 실험 관리 및 시각화
```

### 주요 라이브러리 설명

1. **pandas**: CSV 파일 형태의 데이터를 DataFrame으로 관리
2. **torch**: PyTorch 딥러닝 프레임워크의 핵심
3. **transformers**: Hugging Face의 사전학습 모델 라이브러리
4. **rouge**: 텍스트 요약 평가를 위한 ROUGE 점수 계산
5. **wandb**: 모델 학습 과정 추적 및 시각화

## 3. Configuration 설정

### 3.1 Config 파일 구조 (코드 위치: 셀 5, 줄 1-47)

```python
config_data = {
    "general": {
        "data_path": "../data/",  # 데이터 경로
        "model_name": "digit82/kobart-summarization",  # 사용할 모델
        "output_dir": "./"  # 출력 디렉토리
    },
    "tokenizer": {
        "encoder_max_len": 512,  # 인코더 최대 토큰 길이
        "decoder_max_len": 100,  # 디코더 최대 토큰 길이
        "bos_token": f"{tokenizer.bos_token}",  # 시작 토큰
        "eos_token": f"{tokenizer.eos_token}",  # 종료 토큰
        "special_tokens": [...]  # 특수 토큰 리스트
    },
    "training": {
        "num_train_epochs": 20,  # 학습 에폭 수
        "learning_rate": 1e-5,   # 학습률
        "per_device_train_batch_size": 50,  # 배치 크기
        # ... 기타 학습 설정
    },
    "wandb": {
        "entity": "wandb_repo",
        "project": "project_name",
        "name": "run_name"
    },
    "inference": {
        "ckt_path": "model ckt path",  # 체크포인트 경로
        "batch_size": 32,
        # ... 기타 추론 설정
    }
}
```

### 3.2 주요 설정 항목 설명

#### General 설정
- **data_path**: 학습/검증/테스트 데이터가 저장된 경로
- **model_name**: Hugging Face 모델 허브에서 불러올 모델 이름
- **output_dir**: 학습된 모델과 로그가 저장될 경로

#### Tokenizer 설정
- **encoder_max_len**: 입력 대화문의 최대 토큰 길이 (512)
- **decoder_max_len**: 생성할 요약문의 최대 토큰 길이 (100)
- **special_tokens**: 사람 구분자(#Person1#, #Person2# 등) 포함

#### Training 설정
- **num_train_epochs**: 전체 데이터를 몇 번 반복 학습할지 (20)
- **learning_rate**: 모델 가중치 업데이트 속도 (1e-5)
- **per_device_train_batch_size**: GPU당 한 번에 처리할 샘플 수 (50)
- **fp16**: 16비트 부동소수점 사용 여부 (메모리 절약)

## 4. 데이터 전처리

### 4.1 Preprocess 클래스 (코드 위치: 셀 13, 줄 1-37)

```python
class Preprocess:
    def __init__(self, bos_token: str, eos_token: str) -> None:
        self.bos_token = bos_token  # 시작 토큰
        self.eos_token = eos_token  # 종료 토큰

    @staticmethod
    def make_set_as_df(file_path, is_train=True):
        """CSV 파일을 DataFrame으로 변환"""
        if is_train:
            df = pd.read_csv(file_path)
            train_df = df[['fname','dialogue','summary']]
            return train_df
        else:
            df = pd.read_csv(file_path)
            test_df = df[['fname','dialogue']]
            return test_df

    def make_input(self, dataset, is_test=False):
        """BART 모델 입력 형태로 데이터 가공"""
        if is_test:
            encoder_input = dataset['dialogue']
            decoder_input = [self.bos_token] * len(dataset['dialogue'])
            return encoder_input.tolist(), list(decoder_input)
        else:
            encoder_input = dataset['dialogue']
            # 디코더 입력: <s> + 정답 요약문
            decoder_input = dataset['summary'].apply(lambda x: self.bos_token + str(x))
            # 디코더 출력: 정답 요약문 + </s>
            decoder_output = dataset['summary'].apply(lambda x: str(x) + self.eos_token)
            return encoder_input.tolist(), decoder_input.tolist(), decoder_output.tolist()
```

### 4.2 데이터 전처리 과정 설명

1. **학습 데이터 구조**:
   - **Encoder 입력**: 원본 대화문 (dialogue)
   - **Decoder 입력**: `<s>` + 정답 요약문
   - **Decoder 출력**: 정답 요약문 + `</s>`

2. **테스트 데이터 구조**:
   - **Encoder 입력**: 원본 대화문
   - **Decoder 입력**: `<s>` 토큰만 (요약문 생성 시작)

### 4.3 prepare_train_dataset 함수 (코드 위치: 셀 15, 줄 1-45)

```python
def prepare_train_dataset(config, preprocessor, data_path, tokenizer):
    # 1. CSV 파일 경로 설정
    train_file_path = os.path.join(data_path,'train.csv')
    val_file_path = os.path.join(data_path,'dev.csv')

    # 2. DataFrame 생성
    train_data = preprocessor.make_set_as_df(train_file_path)
    val_data = preprocessor.make_set_as_df(val_file_path)

    # 3. 입력 데이터 생성
    encoder_input_train, decoder_input_train, decoder_output_train = preprocessor.make_input(train_data)
    encoder_input_val, decoder_input_val, decoder_output_val = preprocessor.make_input(val_data)

    # 4. 토큰화 (Tokenization)
    tokenized_encoder_inputs = tokenizer(
        encoder_input_train, 
        return_tensors="pt",      # PyTorch 텐서로 반환
        padding=True,             # 패딩 추가
        add_special_tokens=True,  # 특수 토큰 추가
        truncation=True,          # 최대 길이 초과시 자르기
        max_length=config['tokenizer']['encoder_max_len'],
        return_token_type_ids=False
    )

    # 5. Dataset 객체 생성
    train_inputs_dataset = DatasetForTrain(
        tokenized_encoder_inputs, 
        tokenized_decoder_inputs, 
        tokenized_decoder_outputs,
        len(encoder_input_train)
    )
```

## 5. 데이터셋 클래스

### 5.1 DatasetForTrain 클래스 (코드 위치: 셀 14, 줄 1-21)

```python
class DatasetForTrain(Dataset):
    def __init__(self, encoder_input, decoder_input, labels, len):
        self.encoder_input = encoder_input
        self.decoder_input = decoder_input
        self.labels = labels
        self.len = len

    def __getitem__(self, idx):
        # 1. 인코더 입력 준비
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()}
        # item에는 'input_ids'와 'attention_mask' 포함

        # 2. 디코더 입력 준비
        item2 = {key: val[idx].clone().detach() for key, val in self.decoder_input.items()}
        
        # 3. 키 이름 변경 (디코더용)
        item2['decoder_input_ids'] = item2['input_ids']
        item2['decoder_attention_mask'] = item2['attention_mask']
        item2.pop('input_ids')
        item2.pop('attention_mask')
        
        # 4. 인코더와 디코더 입력 병합
        item.update(item2)
        
        # 5. 레이블(정답) 추가
        item['labels'] = self.labels['input_ids'][idx]
        
        return item

    def __len__(self):
        return self.len
```

### 5.2 데이터셋 클래스 구조 설명

**최종 반환 데이터 구조**:
```python
{
    'input_ids': tensor([...]),              # 인코더 입력 토큰
    'attention_mask': tensor([...]),          # 인코더 어텐션 마스크
    'decoder_input_ids': tensor([...]),       # 디코더 입력 토큰
    'decoder_attention_mask': tensor([...]),  # 디코더 어텐션 마스크
    'labels': tensor([...])                   # 정답 레이블
}
```

## 6. 모델 학습 설정

### 6.1 compute_metrics 함수 (코드 위치: 셀 16, 줄 1-39)

```python
def compute_metrics(config, tokenizer, pred):
    rouge = Rouge()
    predictions = pred.predictions
    labels = pred.label_ids

    # 1. 패딩 토큰(-100) 처리
    predictions[predictions == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id

    # 2. 토큰을 텍스트로 디코딩
    decoded_preds = tokenizer.batch_decode(predictions, clean_up_tokenization_spaces=True)
    labels = tokenizer.batch_decode(labels, clean_up_tokenization_spaces=True)

    # 3. 불필요한 특수 토큰 제거
    remove_tokens = config['inference']['remove_tokens']
    for token in remove_tokens:
        replaced_predictions = [sentence.replace(token," ") for sentence in replaced_predictions]
        replaced_labels = [sentence.replace(token," ") for sentence in replaced_labels]

    # 4. ROUGE 점수 계산
    results = rouge.get_scores(replaced_predictions, replaced_labels, avg=True)
    
    # 5. F1 점수만 추출
    result = {key: value["f"] for key, value in results.items()}
    return result
```

### 6.2 Trainer 설정 (코드 위치: 셀 17, 줄 1-72)

```python
def load_trainer_for_train(config, generate_model, tokenizer, train_inputs_dataset, val_inputs_dataset):
    # 1. 학습 인자 설정
    training_args = Seq2SeqTrainingArguments(
        output_dir=config['general']['output_dir'],
        num_train_epochs=config['training']['num_train_epochs'],
        learning_rate=config['training']['learning_rate'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        evaluation_strategy="epoch",  # 매 에폭마다 평가
        save_strategy="epoch",        # 매 에폭마다 저장
        fp16=True,                    # 16비트 부동소수점 사용
        predict_with_generate=True,   # 생성 모드로 평가
        # ... 기타 설정
    )

    # 2. wandb 초기화 (선택사항)
    wandb.init(
        entity=config['wandb']['entity'],
        project=config['wandb']['project'],
        name=config['wandb']['name'],
    )

    # 3. Early Stopping 콜백
    MyCallback = EarlyStoppingCallback(
        early_stopping_patience=3,      # 3번 개선 없으면 중단
        early_stopping_threshold=0.001  # 최소 개선 폭
    )

    # 4. Trainer 생성
    trainer = Seq2SeqTrainer(
        model=generate_model,
        args=training_args,
        train_dataset=train_inputs_dataset,
        eval_dataset=val_inputs_dataset,
        compute_metrics=lambda pred: compute_metrics(config, tokenizer, pred),
        callbacks=[MyCallback]
    )
```

### 6.3 모델 로드 함수 (코드 위치: 셀 18, 줄 1-19)

```python
def load_tokenizer_and_model_for_train(config, device):
    # 1. 모델 이름 및 설정
    model_name = config['general']['model_name']
    bart_config = BartConfig().from_pretrained(model_name)
    
    # 2. 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 3. 모델 로드
    generate_model = BartForConditionalGeneration.from_pretrained(
        config['general']['model_name'],
        config=bart_config
    )

    # 4. 특수 토큰 추가
    special_tokens_dict = {'additional_special_tokens': config['tokenizer']['special_tokens']}
    tokenizer.add_special_tokens(special_tokens_dict)

    # 5. 임베딩 크기 조정 (특수 토큰 추가로 인해)
    generate_model.resize_token_embeddings(len(tokenizer))
    
    # 6. GPU로 이동
    generate_model.to(device)
```

## 7. 모델 추론

### 7.1 추론용 데이터 준비 (코드 위치: 셀 23, 줄 1-24)

```python
def prepare_test_dataset(config, preprocessor, tokenizer):
    # 1. 테스트 파일 로드
    test_file_path = os.path.join(config['general']['data_path'], 'test.csv')
    test_data = preprocessor.make_set_as_df(test_file_path, is_train=False)
    test_id = test_data['fname']

    # 2. 테스트 입력 생성 (디코더 입력은 <s> 토큰만)
    encoder_input_test, decoder_input_test = preprocessor.make_input(test_data, is_test=True)

    # 3. 토큰화
    test_tokenized_encoder_inputs = tokenizer(
        encoder_input_test,
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
        truncation=True,
        max_length=config['tokenizer']['encoder_max_len'],
        return_token_type_ids=False
    )

    # 4. 추론용 Dataset 생성
    test_encoder_inputs_dataset = DatasetForInference(
        test_tokenized_encoder_inputs,
        test_id,
        len(encoder_input_test)
    )
```

### 7.2 추론 실행 (코드 위치: 셀 25, 줄 1-51)

```python
def inference(config):
    # 1. 모델과 토크나이저 로드
    generate_model, tokenizer = load_tokenizer_and_model_for_test(config, device)

    # 2. 데이터 준비
    test_data, test_encoder_inputs_dataset = prepare_test_dataset(config, preprocessor, tokenizer)
    dataloader = DataLoader(test_encoder_inputs_dataset, batch_size=config['inference']['batch_size'])

    # 3. 추론 실행
    summary = []
    text_ids = []
    
    with torch.no_grad():  # 그래디언트 계산 비활성화
        for item in tqdm(dataloader):
            text_ids.extend(item['ID'])
            
            # 4. 텍스트 생성
            generated_ids = generate_model.generate(
                input_ids=item['input_ids'].to('cuda:0'),
                no_repeat_ngram_size=2,      # 2-gram 반복 방지
                early_stopping=True,         # 조기 종료
                max_length=100,              # 최대 생성 길이
                num_beams=4,                 # 빔 서치 크기
            )
            
            # 5. 디코딩
            for ids in generated_ids:
                result = tokenizer.decode(ids)
                summary.append(result)

    # 6. 특수 토큰 제거
    remove_tokens = config['inference']['remove_tokens']
    for token in remove_tokens:
        preprocessed_summary = [sentence.replace(token, " ") for sentence in preprocessed_summary]

    # 7. 결과 저장
    output = pd.DataFrame({
        "fname": test_data['fname'],
        "summary": preprocessed_summary,
    })
    output.to_csv(os.path.join(result_path, "output.csv"), index=False)
```

## 8. 주요 개념 설명

### 8.1 BART 모델

**BART (Bidirectional and Auto-Regressive Transformers)**는 Facebook AI Research에서 개발한 시퀀스-투-시퀀스(Seq2Seq) 모델입니다.

**특징**:
- **인코더**: 양방향(Bidirectional) - 문맥을 완전히 이해
- **디코더**: 자기회귀(Auto-Regressive) - 순차적으로 텍스트 생성
- **사전학습**: 텍스트 손상 복원 태스크로 학습

**KoBART**:
- 한국어 데이터로 사전학습된 BART 모델
- 40GB 이상의 한국어 텍스트로 학습
- 한국어 요약, 번역, 생성에 특화

### 8.2 토크나이저 (Tokenizer)

**역할**: 텍스트를 모델이 이해할 수 있는 숫자(토큰 ID)로 변환

**주요 기능**:
1. **토큰화**: 텍스트 → 토큰 리스트
2. **인코딩**: 토큰 → 토큰 ID
3. **디코딩**: 토큰 ID → 텍스트

**특수 토큰**:
- `<s>` (BOS): 문장 시작
- `</s>` (EOS): 문장 종료
- `<pad>`: 패딩
- `#Person1#`, `#Person2#`: 화자 구분

### 8.3 Attention Mask

**목적**: 모델이 주목해야 할 토큰과 무시해야 할 토큰 구분

```python
attention_mask = [1, 1, 1, 1, 0, 0]  # 1: 실제 토큰, 0: 패딩
```

### 8.4 Teacher Forcing

학습 시 디코더에 정답을 입력하는 기법:

```
시간 t=0: 입력 <s>       → 출력 "오늘"
시간 t=1: 입력 "오늘"    → 출력 "날씨가"
시간 t=2: 입력 "날씨가"  → 출력 "좋다"
시간 t=3: 입력 "좋다"    → 출력 </s>
```

### 8.5 Beam Search

추론 시 여러 후보를 동시에 고려하는 디코딩 전략:

```python
num_beams=4  # 상위 4개 후보 유지
```

각 단계에서 확률이 가장 높은 4개의 시퀀스를 유지하며 진행

### 8.6 Early Stopping

**학습 조기 종료**:
- `patience=3`: 3 에폭 동안 개선 없으면 중단
- `threshold=0.001`: 최소 개선 폭

**생성 조기 종료**:
- EOS 토큰 생성 시 종료

### 8.7 Mixed Precision Training (fp16)

**장점**:
- 메모리 사용량 50% 감소
- 학습 속도 향상
- 정확도 손실 최소화

```python
fp16=True  # 16비트 부동소수점 사용
```

### 8.8 Gradient Accumulation

작은 배치를 여러 번 누적하여 큰 배치 효과:

```python
gradient_accumulation_steps=1  # 1스텝마다 업데이트
```

실제 배치 크기 = `per_device_batch_size` × `gradient_accumulation_steps`

### 8.9 Learning Rate Scheduler

학습률을 동적으로 조정:

```python
lr_scheduler_type='cosine'  # 코사인 스케줄러
warmup_ratio=0.1           # 10% 워밍업
```

### 8.10 Weight Decay

과적합 방지를 위한 가중치 정규화:

```python
weight_decay=0.01  # L2 정규화 강도
```

## 마무리

이 베이스라인 코드는 대화 요약을 위한 완전한 파이프라인을 제공합니다. 주요 구성요소는:

1. **데이터 전처리**: 대화문을 모델 입력 형태로 변환
2. **모델 학습**: BART 모델을 한국어 대화 요약에 맞게 미세조정
3. **평가**: ROUGE 지표로 성능 측정
4. **추론**: 학습된 모델로 새로운 대화 요약 생성

성능 향상을 위한 개선 방향:
- 하이퍼파라미터 튜닝 (학습률, 배치 크기 등)
- 데이터 증강 기법 적용
- 다른 사전학습 모델 실험 (T5, GPT 등)
- 앙상블 기법 적용
