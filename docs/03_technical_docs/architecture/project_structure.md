# 프로젝트 구조 및 추가 파일 분석

## 1. 프로젝트 전체 구조

```
nlp-sum-lyj/
├── code/                   # 소스 코드 디렉토리
│   ├── baseline.ipynb     # 메인 베이스라인 코드
│   ├── solar_api.ipynb    # Solar API 활용 코드
│   ├── config.yaml        # 설정 파일
│   └── requirements.txt   # 필요 패키지 목록
├── data/                  # 데이터 디렉토리
│   ├── train.csv         # 학습 데이터 (12,457개)
│   ├── dev.csv           # 검증 데이터 (499개)
│   ├── test.csv          # 테스트 데이터 (250개)
│   └── sample_submission.csv  # 제출 양식
└── docs/                  # 문서 디렉토리
    ├── competition_overview.md     # 대회 개요
    ├── baseline_code_analysis.md   # 베이스라인 분석
    └── rouge_metrics_detail.md     # ROUGE 지표 설명
```

## 2. config.yaml 상세 분석

### 2.1 전체 구조

```yaml
general:          # 일반 설정
tokenizer:        # 토크나이저 설정
training:         # 학습 설정
wandb:           # 실험 관리 설정
inference:       # 추론 설정
```

### 2.2 General 설정

```yaml
general:
  data_path: ../data/                    # 데이터 경로
  model_name: digit82/kobart-summarization  # 사용 모델
  output_dir: ./                         # 출력 디렉토리
```

**설명**:
- `data_path`: train.csv, dev.csv, test.csv가 있는 디렉토리
- `model_name`: Hugging Face Model Hub의 모델 이름
- `output_dir`: 체크포인트와 로그가 저장될 위치

### 2.3 Tokenizer 설정

```yaml
tokenizer:
  bos_token: <s>           # Beginning of Sentence
  eos_token: </s>          # End of Sentence
  encoder_max_len: 512     # 입력 최대 길이
  decoder_max_len: 100     # 출력 최대 길이
  special_tokens:          # 특수 토큰 목록
    - '#Person1#'          # 화자 1
    - '#Person2#'          # 화자 2
    - '#Person3#'          # 화자 3
    - '#PhoneNumber#'      # 전화번호 마스킹
    - '#Address#'          # 주소 마스킹
    - '#PassportNumber#'   # 여권번호 마스킹
```

**특수 토큰의 중요성**:
- 화자 구분자(`#Person1#` 등)가 서브워드로 분해되지 않도록 보호
- 개인정보 마스킹 토큰도 하나의 단위로 처리

### 2.4 Training 설정

```yaml
training:
  # 기본 학습 설정
  num_train_epochs: 20              # 에폭 수
  learning_rate: 1.0e-05           # 학습률 (0.00001)
  per_device_train_batch_size: 50  # GPU당 배치 크기
  per_device_eval_batch_size: 32   # 평가시 배치 크기
  
  # 최적화 설정
  optim: adamw_torch               # AdamW 옵티마이저
  lr_scheduler_type: cosine        # 코사인 학습률 스케줄러
  warmup_ratio: 0.1                # 10% 워밍업
  weight_decay: 0.01               # 가중치 감쇠
  
  # 학습 전략
  gradient_accumulation_steps: 1    # 그래디언트 누적
  fp16: true                       # 혼합 정밀도 학습
  seed: 42                         # 랜덤 시드
  
  # 평가 및 저장
  evaluation_strategy: epoch        # 에폭마다 평가
  save_strategy: epoch             # 에폭마다 저장
  save_total_limit: 5              # 최대 5개 체크포인트
  load_best_model_at_end: true     # 최고 성능 모델 로드
  
  # 조기 종료
  early_stopping_patience: 3        # 3에폭 개선 없으면 종료
  early_stopping_threshold: 0.001   # 최소 개선 폭
  
  # 생성 설정
  predict_with_generate: true       # 생성 모드로 평가
  generation_max_length: 100        # 생성 최대 길이
  
  # 로깅
  logging_dir: ./logs              # 로그 디렉토리
  logging_strategy: epoch          # 에폭마다 로깅
  report_to: wandb                 # wandb에 리포트
```

### 2.5 Inference 설정

```yaml
inference:
  batch_size: 32                    # 추론 배치 크기
  ckt_path: model ckt path         # 체크포인트 경로
  result_path: ./prediction/        # 결과 저장 경로
  
  # 생성 설정
  no_repeat_ngram_size: 2          # 2-gram 반복 방지
  early_stopping: true             # 조기 종료 사용
  generate_max_length: 100         # 최대 생성 길이
  num_beams: 4                     # 빔 서치 크기
  
  # 후처리
  remove_tokens:                   # 제거할 토큰
    - <usr>
    - <s>
    - </s>
    - <pad>
```

### 2.6 WandB 설정

```yaml
wandb:
  entity: wandb_repo    # WandB 계정/조직명
  project: project_name # 프로젝트명
  name: run_name       # 실행 이름
```

## 3. requirements.txt 분석

### 3.1 핵심 라이브러리

```txt
# 데이터 처리
pandas==2.1.4          # 데이터프레임 처리
numpy==1.23.5          # 수치 연산

# 딥러닝 프레임워크
pytorch_lightning==2.1.2    # PyTorch 래퍼
transformers[torch]==4.35.2  # Hugging Face Transformers

# 평가 및 모니터링
rouge==1.0.1          # ROUGE 평가 지표
wandb==0.16.1         # 실험 추적 및 시각화

# 유틸리티
tqdm==4.66.1          # 진행 바 표시

# 개발 환경
jupyter==1.0.0        # Jupyter 노트북
jupyterlab==4.0.9     # JupyterLab 환경
```

### 3.2 라이브러리 버전 호환성

- **PyTorch**: transformers[torch] 설치 시 자동으로 호환 버전 설치
- **CUDA**: GPU 사용 시 CUDA 11.x 이상 권장
- **Python**: 3.8 이상 권장

## 4. 데이터 형식 분석

### 4.1 train.csv / dev.csv 구조

```csv
fname,dialogue,summary,topic
train_0,"#Person1#: 안녕하세요, Mr. Smith. 저는 Dr. Hawkins입니다...",
"Mr. Smith가 건강검진을 받으러 왔습니다...",health
```

**컬럼 설명**:
- `fname`: 대화 고유 ID (train_0, train_1, ...)
- `dialogue`: 대화 내용 (화자는 #PersonN#으로 구분)
- `summary`: 요약문 (정답)
- `topic`: 대화 주제 (참고용, 학습에 미사용)

### 4.2 test.csv 구조

```csv
fname,dialogue
test_0,"#Person1#: 대화 내용..."
```

- `summary` 컬럼 없음 (예측 대상)

### 4.3 sample_submission.csv 구조

```csv
fname,summary
test_0,"예측된 요약문"
test_1,"예측된 요약문"
...
```

## 5. 실행 가이드

### 5.1 환경 설정

#### 방법 1: 기존 방식 (pip)
```bash
# 1. 가상환경 생성
python -m venv dialogue_sum_env
source dialogue_sum_env/bin/activate  # Windows: .\dialogue_sum_env\Scripts\activate

# 2. 패키지 설치
pip install -r requirements.txt

# 3. CUDA 확인 (GPU 사용 시)
python -c "import torch; print(torch.cuda.is_available())"
```

#### 방법 2: uv 사용 (권장 - 10배 이상 빠름)
```bash
# 1. uv 설치 (처음 한 번만)
pip install uv

# 2. 가상환경 생성 (0.1초!)
uv venv dialogue_sum_env
source dialogue_sum_env/bin/activate  # Windows: .\dialogue_sum_env\Scripts\activate

# 3. 패키지 설치 (매우 빠름!)
uv pip install -r requirements.txt

# 4. Lock 파일 생성 (팀 협업용)
uv pip compile requirements.txt -o requirements.lock

# 5. CUDA 확인
python -c "import torch; print(torch.cuda.is_available())"
```

> 💡 **실제 측정 결과**: 
> - pip: 약 90초
> - uv: 약 7초 (12배 빠름!)
> 
> 자세한 내용은 [uv 패키지 관리자 가이드](uv_package_manager_guide.md) 참고

### 5.2 학습 실행

```python
# baseline.ipynb에서

# 1. config 수정
loaded_config['general']['data_path'] = "your_data_path"
loaded_config['wandb']['entity'] = "your_wandb_account"
loaded_config['wandb']['project'] = "your_project_name"

# 2. 학습 실행
main(loaded_config)
```

### 5.3 추론 실행

```python
# 1. 체크포인트 경로 설정
loaded_config['inference']['ckt_path'] = "./checkpoint-best"

# 2. 추론 실행
output = inference(loaded_config)

# 3. 결과 확인
print(output.head())
```

## 6. 성능 최적화 팁

### 6.1 메모리 최적화

```python
# 배치 크기 조정
config['training']['per_device_train_batch_size'] = 32  # 메모리 부족 시 감소

# Gradient Accumulation 사용
config['training']['gradient_accumulation_steps'] = 2  # 실효 배치 크기 = 32 * 2 = 64

# fp16 사용 (이미 설정됨)
config['training']['fp16'] = True
```

### 6.2 학습 속도 개선

```python
# DataLoader 워커 수 증가 (코드 수정 필요)
dataloader = DataLoader(..., num_workers=4)

# 체크포인트 저장 빈도 조정
config['training']['save_strategy'] = 'steps'
config['training']['save_steps'] = 500
```

#### 환경 설정 속도 개선 (uv 사용)
```bash
# 기존 pip (약 90초)
time pip install -r requirements.txt

# uv 사용 (약 7초 - 12배 빠름!)
time uv pip install -r requirements.txt

# CI/CD 파이프라인에서 특히 유용
# 빌드 시간 90% 단축 가능
```

### 6.3 성능 향상

```python
# 학습률 조정
config['training']['learning_rate'] = 5e-5  # 또는 3e-5

# 에폭 수 증가
config['training']['num_train_epochs'] = 30

# 빔 서치 크기 증가
config['inference']['num_beams'] = 8

# 최대 길이 조정
config['tokenizer']['decoder_max_len'] = 150
config['inference']['generate_max_length'] = 150
```

## 7. 일반적인 문제 해결

### 7.1 CUDA Out of Memory

```python
# 해결 방법
1. 배치 크기 감소
2. gradient_accumulation_steps 증가
3. 모델을 CPU로 이동 후 추론
4. torch.cuda.empty_cache() 사용
```

### 7.2 학습이 수렴하지 않음

```python
# 해결 방법
1. 학습률 조정 (더 작게)
2. 워밍업 비율 증가
3. 데이터 품질 확인
4. 더 긴 에폭 학습
```

### 7.3 생성된 요약문 품질 문제

```python
# 해결 방법
1. no_repeat_ngram_size 조정
2. temperature 파라미터 추가 (0.8~1.0)
3. top_k, top_p 샘플링 사용
4. 더 큰 모델 사용 (예: kobart-base → kobart-large)
```
