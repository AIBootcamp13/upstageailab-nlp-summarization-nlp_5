# Run_main_5_experiments.sh 5개 모델 상세 분석

## 📌 개요

run_main_5_experiments.sh에는 RTX 3090 24GB GPU에 최적화된 5개의 실험이 포함되어 있습니다. 각 모델은 특정 목적과 최적화 전략을 가지고 설계되었습니다.

---

## 1. 🚀 mT5 XLSum 한국어 QLoRA 극한최적화

### 기본 정보
- **파일**: `01_mt5_xlsum_ultimate_korean_qlora.yaml`
- **모델**: `csebuetnlp/mT5_multilingual_XLSum`
- **아키텍처**: mT5 (multilingual T5)
- **예상 시간**: 60분

### 모델 특징
- **다국어 지원**: 101개 언어로 사전학습된 모델
- **XLSum 데이터셋**: 뉴스 요약에 특화된 사전학습
- **한국어 적응**: prefix로 "dialogue summarization in korean:" 사용
- **대용량 모델**: mT5-base는 약 580M 파라미터

### 하이퍼파라미터

#### 시퀀스 길이
```yaml
encoder_max_len: 1024    # 입력 대화 최대 길이
decoder_max_len: 200     # 출력 요약 최대 길이
```

#### 배치 설정
```yaml
per_device_train_batch_size: 32     # GPU당 학습 배치
per_device_eval_batch_size: 48      # GPU당 평가 배치  
gradient_accumulation_steps: 2      # 유효 배치 = 32 × 2 = 64
```

#### 학습 파라미터
```yaml
num_train_epochs: 6
learning_rate: 8.0e-05              # 높은 학습률
lr_scheduler_type: cosine_with_restarts
warmup_ratio: 0.02                  # 전체 스텝의 2% 웜업
weight_decay: 0.0001
```

#### QLoRA 설정
```yaml
use_qlora: true
lora_rank: 128                      # LoRA 차원
lora_alpha: 256                     # LoRA 스케일링 (alpha/rank = 2.0)
lora_dropout: 0.01                  # 낮은 드롭아웃
target_modules: ["q", "k", "v", "o", "wi", "wo", "lm_head", "embed_tokens"]
load_in_4bit: true                  # 4비트 양자화
```

#### 생성 설정
```yaml
generation_num_beams: 12            # 빔 서치 크기
generation_max_length: 200
generation_min_length: 15
generation_length_penalty: 0.7      # 짧은 요약 선호
generation_no_repeat_ngram_size: 3  # 3-gram 반복 방지
```

### 최적화 전략
- **메모리 절약**: QLoRA로 4비트 양자화 + LoRA 어댑터
- **대용량 배치**: 유효 배치 64로 안정적 학습
- **긴 시퀀스**: 1024 토큰까지 처리 가능

---

## 2. 💪 eenzeenee T5 한국어 극한최적화

### 기본 정보
- **파일**: `02_eenzeenee_t5_rtx3090.yaml`
- **모델**: `eenzeenee/t5-base-korean-summarization`
- **아키텍처**: T5-base
- **예상 시간**: 40분

### 모델 특징
- **한국어 전용**: 한국어 요약에 특화된 T5 모델
- **중간 크기**: T5-base는 약 250M 파라미터
- **요약 특화**: 한국어 요약 데이터로 추가 학습됨
- **효율적**: mT5보다 작지만 한국어에 최적화

### 하이퍼파라미터

#### 배치 설정
```yaml
per_device_train_batch_size: 20     # 더 큰 배치 가능
per_device_eval_batch_size: 32
gradient_accumulation_steps: 2      # 유효 배치 = 20 × 2 = 40
```

#### 학습 파라미터
```yaml
num_train_epochs: 5
learning_rate: 8.0e-05
lr_scheduler_type: cosine_with_restarts
warmup_ratio: 0.03
weight_decay: 0.001
```

#### 최적화 설정
```yaml
bf16: true                          # Brain Float 16 사용
tf32: true                          # Tensor Float 32 (Ampere 최적화)
dataloader_num_workers: 36          # 병렬 데이터 로딩
gradient_checkpointing: false       # 메모리 여유로 비활성화
```

### 최적화 전략
- **큰 배치**: 작은 모델 크기를 활용해 배치 20 사용
- **빠른 학습**: 40분 내 완료 목표
- **한국어 특화**: prefix와 특수 토큰으로 한국어 대화 최적화

---

## 3. 💪 KoBART Baseline 극한최적화

### 기본 정보
- **파일**: `01_baseline_kobart_rtx3090.yaml`
- **모델**: `digit82/kobart-summarization`
- **아키텍처**: BART
- **예상 시간**: 45분

### 모델 특징
- **한국어 BART**: 한국어에 특화된 BART 모델
- **안정적**: 가장 널리 사용되는 한국어 요약 모델
- **중간 크기**: 약 124M 파라미터
- **검증됨**: 많은 벤치마크에서 우수한 성능

### 하이퍼파라미터

#### 시퀀스 길이
```yaml
encoder_max_len: 1280              # 매우 긴 입력 지원
decoder_max_len: 200
```

#### 배치 설정
```yaml
per_device_train_batch_size: 16
per_device_eval_batch_size: 24
gradient_accumulation_steps: 3      # 유효 배치 = 16 × 3 = 48
```

#### QLoRA 설정
```yaml
lora_rank: 192                      # 높은 LoRA 차원
lora_alpha: 384                     # alpha/rank = 2.0
lora_dropout: 0.05
target_modules: ["q", "k", "v", "o", "fc1", "fc2", "lm_head", "embed_tokens"]
```

### 최적화 전략
- **균형잡힌 설정**: 안정성과 성능의 균형
- **긴 시퀀스**: 1280 토큰으로 긴 대화 처리
- **높은 LoRA rank**: 192로 표현력 증가

---

## 4. 💪 고학습률 실험 극한최적화

### 기본 정보
- **파일**: `03_high_learning_rate_rtx3090.yaml`
- **모델**: `digit82/kobart-summarization` (KoBART 기반)
- **아키텍처**: BART
- **예상 시간**: 35분

### 모델 특징
- **실험적**: 매우 높은 학습률 테스트
- **빠른 수렴**: 35분 내 학습 완료 목표
- **공격적 최적화**: 일반적인 범위를 벗어난 학습률

### 하이퍼파라미터

#### 핵심: 극한 학습률
```yaml
learning_rate: 1.2e-04              # 일반적인 1e-05의 12배!
warmup_ratio: 0.01                  # 매우 짧은 웜업 (1%)
```

#### 배치 설정
```yaml
per_device_train_batch_size: 14
gradient_accumulation_steps: 4      # 유효 배치 = 14 × 4 = 56
```

#### 안정화 전략
```yaml
max_grad_norm: 0.3                  # 낮은 gradient clipping
early_stopping_patience: 10         # 긴 patience
save_steps: 100                     # 자주 저장
```

### 최적화 전략
- **고위험 고수익**: 빠른 학습 vs 불안정성
- **세밀한 모니터링**: 25 스텝마다 로깅
- **많은 체크포인트**: 최대 10개 저장

---

## 5. 💪 배치 극한최적화

### 기본 정보
- **파일**: `04_batch_optimization_rtx3090.yaml`
- **모델**: `digit82/kobart-summarization` (KoBART 기반)
- **아키텍처**: BART
- **예상 시간**: 40분

### 모델 특징
- **대용량 배치**: 유효 배치 64 달성
- **안정적 학습**: 큰 배치로 gradient 안정화
- **효율적**: GPU 활용도 극대화

### 하이퍼파라미터

#### 배치 최적화
```yaml
per_device_train_batch_size: 16     # 큰 배치
per_device_eval_batch_size: 24
gradient_accumulation_steps: 4      # 유효 배치 = 16 × 4 = 64
group_by_length: true               # 길이별 그룹화로 메모리 효율
```

#### 학습 파라미터
```yaml
learning_rate: 7.0e-05              # 큰 배치에 맞는 학습률
num_train_epochs: 5
dataloader_num_workers: 36          # 많은 워커로 데이터 로딩 최적화
```

### 최적화 전략
- **메모리 효율**: group_by_length로 패딩 최소화
- **안정성**: 큰 배치로 노이즈 감소
- **처리량**: 높은 GPU 활용률

---

## 🔥 공통 최적화 기술

### 1. RTX 3090 최적화
```yaml
bf16: true                          # BFloat16 (Ampere 지원)
tf32: true                          # TensorFloat32
dataloader_pin_memory: true         # CPU-GPU 전송 최적화
dataloader_persistent_workers: true # 워커 재사용
```

### 2. Unsloth 통합
- 모든 모델이 Unsloth 라이브러리 활용
- 메모리 사용량 최대 50% 감소
- 학습 속도 2-3배 향상

### 3. 동적 메모리 관리
- GPU 메모리 모니터링
- 실험 간 자동 메모리 정리
- 스마트 대기 시스템

---

## 📊 모델 비교 요약

| 모델 | 크기 | 배치 | 유효배치 | 학습률 | 특징 |
|------|------|------|----------|--------|------|
| mT5 XLSum | 580M | 32 | 64 | 8e-5 | 다국어, 큰 모델 |
| eenzeenee T5 | 250M | 20 | 40 | 8e-5 | 한국어 특화 |
| KoBART | 124M | 16 | 48 | 6e-5 | 안정적, 검증됨 |
| 고학습률 | 124M | 14 | 56 | 1.2e-4 | 실험적, 빠른 수렴 |
| 배치최적화 | 124M | 16 | 64 | 7e-5 | 큰 배치, 안정성 |

---

## 🎯 사용 권장사항

1. **안정적인 결과**: KoBART Baseline 사용
2. **최고 성능 도전**: mT5 XLSum 사용
3. **빠른 실험**: eenzeenee T5 또는 고학습률 실험
4. **대용량 데이터**: 배치 최적화 실험 사용
5. **한국어 특화**: eenzeenee T5 또는 KoBART 사용

모든 모델은 RTX 3090 24GB에 최적화되어 있으며, 메모리 오버플로우 없이 안정적으로 실행됩니다.
