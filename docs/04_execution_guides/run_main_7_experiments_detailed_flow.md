# 🚀 Main 7 Experiments 실행 흐름 완전 가이드

## 📋 개요

이 문서는 AIStages 서버에서 `run_main_7_experiments.sh` 스크립트를 실행할 때의 전체 흐름을 **초보자도 이해할 수 있도록** 상세히 설명합니다.

## 🎯 실험 목표

- **3개 mT5 XLSum 모델**: 다국어 요약 모델을 한국어 대화 요약에 최적화
- **4개 고성능 모델**: 기존 모델들을 RTX 3090에서 극한 성능 달성
- **총 7개 실험**: 각기 다른 접근법으로 ROUGE-1 25% 이상 목표

## 🔧 실행 명령어

```bash
# 전체 실험 (6-7시간)
bash run_main_7_experiments.sh

# 빠른 테스트 (30-45분)
bash run_main_7_experiments.sh -1
```

## 📊 **단계별 실행 흐름**

### **1단계: 스크립트 초기화** (1-2분)

#### 1.1 명령어 파싱
```bash
# 스크립트가 -1 옵션을 확인
if [[ "$1" == "-1" ]]; then
    ONE_EPOCH_MODE=true  # 빠른 테스트 모드
else
    ONE_EPOCH_MODE=false # 전체 실험 모드
fi
```

**💡 개념 설명:**
- **에포크(Epoch)**: 전체 훈련 데이터를 한 번 다 학습하는 단위
- **1에포크 모드**: 빠른 테스트를 위해 데이터를 한 번만 학습
- **전체 모드**: 각 실험마다 4-8에포크 학습 (더 좋은 성능)

#### 1.2 환경 준비
```bash
# 실험 로그 디렉토리 생성
LOG_DIR="logs/main_experiments_20250730_183025"  # 타임스탬프 포함
mkdir -p "$LOG_DIR"

# GPU 상태 확인
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free
```

**📊 AIStages 서버 GPU 정보:**
- GPU: RTX 3090 24GB
- CUDA 버전: 11.8+
- 메모리: 24GB VRAM

### **2단계: 실험 목록 정의** 

#### 2.1 7개 실험 배열
```bash
experiments=(
    # mT5 XLSum 3단계
    "01_mt5_xlsum_optimized_v2.yaml:🔥_mT5_XLSum_1단계:45분"
    "01_mt5_xlsum_korean_adapted.yaml:🔧_mT5_XLSum_2단계:1시간"
    "01_mt5_xlsum_ultimate.yaml:🚀_mT5_XLSum_3단계:1.5시간"
    
    # 고성능 4개
    "02_eenzeenee_t5_rtx3090.yaml:💪_eenzeenee_T5:1시간"
    "01_baseline_kobart_rtx3090.yaml:💪_KoBART:1시간"  
    "03_high_learning_rate_rtx3090.yaml:💪_극한_학습률:1시간"
    "04_batch_optimization_rtx3090.yaml:💪_배치_최적화:1.5시간"
)
```

### **3단계: 각 실험 순차 실행** (실험별 30분-1.5시간)

#### 3.1 실험 1: mT5 XLSum 1단계 (45분)

**🔥 모델 정보:**
- **베이스 모델**: `csebuetnlp/mT5_multilingual_XLSum`
- **모델 타입**: mT5 (다국어 T5)
- **원래 용도**: 44개 언어 뉴스 요약
- **우리 목적**: 한국어 대화 요약으로 전이학습

**📝 설정 파일**: `config/experiments/01_mt5_xlsum_optimized_v2.yaml`

```yaml
# 핵심 설정
model:
  checkpoint: csebuetnlp/mT5_multilingual_XLSum
training:
  per_device_train_batch_size: 6      # RTX 3090 최적화
  learning_rate: 6.0e-05              # 안정적 학습률
  num_train_epochs: 4                 # 충분한 학습
qlora:
  use_unsloth: true                   # Unsloth 가속
  lora_rank: 48                       # QLoRA 효율성
```

**🎯 핵심 개념:**
- **QLoRA**: 모델의 일부만 학습해서 메모리 절약 (24GB → 8GB 사용)
- **Unsloth**: QLoRA를 2-5배 빠르게 만드는 최적화 라이브러리
- **LoRA Rank 48**: 학습할 파라미터의 복잡도 (높을수록 성능↑, 메모리↑)

**🔄 실행 과정:**
```bash
# 실제 실행 명령어
python code/auto_experiment_runner.py \
    --config config/experiments/01_mt5_xlsum_optimized_v2.yaml \
    --one-epoch  # (1에포크 모드일 때만)
```

**📊 내부 동작:**
1. **모델 로딩**: HuggingFace에서 mT5 모델 다운로드 (1-2GB)
2. **데이터 로딩**: `data/train.csv` 읽기 (12MB, 70,000건)
3. **토크나이저 설정**: 
   - 입력: 최대 512토큰 (대화 텍스트)
   - 출력: 최대 128토큰 (요약 텍스트)
   - 특수 토큰: `#Person1#`, `#Person2#` 등
4. **QLoRA 설정**: 
   - 4-bit 양자화로 메모리 절약
   - LoRA로 어댑터만 학습
5. **학습 루프**: 4에포크 × 200스텝씩 = 800스텝 학습

#### 3.2 실험 2: mT5 XLSum 2단계 - 한국어 특화 (1시간)

**🔧 특화 전략:**
- **도메인 적응**: 뉴스 요약 → 대화 요약
- **한국어 최적화**: 더 긴 입력(768토큰), 한국어 특수토큰
- **높은 학습률**: 9.0e-05 (1단계보다 50% 증가)

```yaml
# 핵심 차이점
input_prefix: "한국어 대화를 요약하세요: "    # 한국어 명시
encoder_max_len: 768                        # 더 긴 대화 처리
learning_rate: 9.0e-05                      # 공격적 학습률
lora_rank: 64                               # 더 높은 표현력
```

**💡 도메인 적응이란?**
- **기존**: 뉴스 기사 → 뉴스 헤드라인
- **적응 후**: 일상 대화 → 대화 요약
- **방법**: 한국어 대화 데이터로 추가 학습

#### 3.3 실험 3: mT5 XLSum 3단계 - 극한 최적화 (1.5시간)

**🚀 세계 대회 수준 도전:**
- **최대 컨텍스트**: 1024토큰 (가장 긴 대화도 처리)
- **극한 학습률**: 1.2e-04 (표준의 2-3배)
- **최대 빔 서치**: 10개 후보 중 최고 선택

```yaml
# 극한 설정
encoder_max_len: 1024                       # 최대 컨텍스트
learning_rate: 1.2e-04                      # 극한 학습률
generation_num_beams: 10                    # 최고 품질 생성
lora_rank: 96                               # 최대 표현력
```

**🎯 빔 서치란?**
- **Beam=1**: 가장 확률 높은 단어만 선택 (빠름, 품질 보통)
- **Beam=10**: 10개 후보 중 최고 선택 (느림, 품질 최고)

#### 3.4 실험 4: eenzeenee T5 최적화 (1시간)

**💪 모델 정보:**
- **베이스 모델**: `eenzeenee/t5-base-korean-summarization`
- **특징**: 한국어 전용으로 이미 학습된 T5
- **장점**: 한국어 성능 특화, 빠른 수렴

```yaml
# T5 최적화 설정
per_device_train_batch_size: 12             # 더 큰 배치 (T5-base는 작음)
learning_rate: 5.0e-05                      # T5 최적 학습률
use_qlora: true                             # QLoRA 적용
```

**💡 mT5 vs T5 차이:**
- **mT5**: 다국어 지원, 큰 모델, 범용성
- **T5**: 단일 언어, 작은 모델, 특화 성능

#### 3.5 실험 5: KoBART 최적화 (1시간)

**💪 모델 정보:**
- **베이스 모델**: `digit82/kobart-summarization`
- **아키텍처**: BART (Bidirectional and Auto-Regressive Transformers)
- **특징**: 한국어 요약에 특화된 모델

```yaml
# KoBART 설정
architecture: bart
per_device_train_batch_size: 8              # BART는 메모리 더 사용
gradient_checkpointing: true                # 메모리 절약
generation_num_beams: 8                     # 고품질 생성
```

**🔄 BART vs T5 차이:**
- **BART**: 인코더-디코더, 노이즈 제거 학습
- **T5**: Text-to-Text, 모든 것을 텍스트 생성으로

#### 3.6 실험 6: 극한 고성능 학습률 (1시간)

**⚡ 극한 최적화 전략:**
- **학습률 8.0e-05**: 일반적 학습률의 2-4배
- **cosine_with_restarts**: 학습률을 주기적으로 리셋
- **bfloat16**: fp16보다 안정적인 혼합 정밀도

```yaml
# 극한 설정
learning_rate: 8.0e-05                      # 극한 학습률
lr_scheduler_type: cosine_with_restarts     # 고성능 스케줄러
bf16: true                                  # 안정적 정밀도
lora_rank: 96                               # 극한 표현력
```

**📈 학습률 스케줄러란?**
- **Linear**: 점진적 감소
- **Cosine**: 코사인 곡선으로 감소
- **Cosine with Restarts**: 주기적으로 높은 학습률로 리셋

#### 3.7 실험 7: 배치 극한 최적화 (1.5시간)

**🔥 RTX 3090 24GB 극한 활용:**
- **유효 배치 크기 32**: 가장 큰 배치로 안정적 학습
- **극세밀 모니터링**: 10스텝마다 로깅
- **최대 빔 서치 12**: 최고 품질 생성

```yaml
# 배치 최적화
per_device_train_batch_size: 4              # 긴 시퀀스 고려
gradient_accumulation_steps: 8              # 4 × 8 = 32 유효 배치
generation_num_beams: 12                    # 최고 품질
lora_rank: 128                              # 최대 표현력
```

**💡 배치 크기의 중요성:**
- **작은 배치**: 빠른 업데이트, 불안정
- **큰 배치**: 안정적 학습, 더 좋은 성능

### **4단계: 실험 간 전환** (각 1분)

```bash
# GPU 메모리 정리
python3 -c "
import torch
import gc
if torch.cuda.is_available():
    torch.cuda.empty_cache()  # GPU 메모리 해제
    torch.cuda.synchronize()  # GPU 동기화
gc.collect()                  # CPU 메모리 정리
"

# 60초 대기 (GPU 쿨다운)
sleep 60
```

**🔧 메모리 정리 이유:**
- 이전 실험의 모델이 GPU 메모리 점유
- 다음 실험을 위한 공간 확보
- GPU 온도 관리

### **5단계: 결과 수집 및 요약** (1-2분)

#### 5.1 실험 결과 파일 생성
```bash
# 각 실험별 로그 파일
logs/main_experiments_20250730_183025/
├── experiment_1_mT5_XLSum_1단계.log
├── experiment_2_mT5_XLSum_2단계.log
├── ...
└── experiment_summary.txt
```

#### 5.2 WandB 실험 추적
- **프로젝트**: https://wandb.ai/lyjune37-juneictlab/nlp-5
- **실시간 모니터링**: 손실, 메트릭, GPU 사용률
- **모델 비교**: 7개 실험 성능 비교

## 📁 **핵심 파일 구조**

### 설정 파일 (`config/experiments/`)
```
01_mt5_xlsum_optimized_v2.yaml     # mT5 1단계
01_mt5_xlsum_korean_adapted.yaml   # mT5 2단계  
01_mt5_xlsum_ultimate.yaml         # mT5 3단계
02_eenzeenee_t5_rtx3090.yaml       # eenzeenee T5
01_baseline_kobart_rtx3090.yaml    # KoBART
03_high_learning_rate_rtx3090.yaml # 극한 학습률
04_batch_optimization_rtx3090.yaml # 배치 최적화
```

### 실행 코드 (`code/`)
```
auto_experiment_runner.py          # 실험 관리자
trainer.py                         # 핵심 학습 로직
utils/                             # 유틸리티 모듈
├── path_utils.py                  # 경로 관리
├── device_utils.py                # GPU 최적화
├── experiment_utils.py            # 실험 추적
└── data_utils.py                  # 데이터 처리
```

### 데이터 (`data/`)
```
train.csv                          # 훈련 데이터 (70,000건)
dev.csv                           # 검증 데이터 (8,000건)
test.csv                          # 테스트 데이터 (9,000건)
```

## 🎯 **핵심 기술 개념**

### QLoRA (Quantized Low-Rank Adaptation)
- **목적**: 24GB GPU에서 대형 모델 학습
- **방법**: 4-bit 양자화 + LoRA 어댑터
- **효과**: 메모리 75% 절약, 성능 유지

### Unsloth 가속
- **라이브러리**: Facebook 연구진 개발
- **효과**: QLoRA 2-5배 속도 향상
- **원리**: 커널 최적화, 메모리 레이아웃 개선

### 혼합 정밀도 (Mixed Precision)
- **FP16**: 절반 정밀도, 빠름, 때로 불안정
- **BF16**: Brain Float 16, 안정적, RTX 3090 최적
- **효과**: 속도 2배, 메모리 절약

### 빔 서치 (Beam Search)
- **Greedy**: 가장 확률 높은 토큰만 선택
- **Beam**: 여러 후보 동시 탐색
- **Trade-off**: 품질 vs 속도

## 📊 **예상 성능 목표**

| 실험 | 모델 | 목표 ROUGE-1 | 특징 |
|------|------|--------------|------|
| 1 | mT5 1단계 | 15-18% | 기본 전이학습 |
| 2 | mT5 2단계 | 18-22% | 한국어 특화 |
| 3 | mT5 3단계 | 22-25% | 극한 최적화 |
| 4 | eenzeenee T5 | 16-20% | 한국어 전용 |
| 5 | KoBART | 18-22% | 요약 특화 |
| 6 | 극한 학습률 | 20-24% | 고성능 학습 |
| 7 | 배치 최적화 | 21-25% | 안정적 수렴 |

## 🚨 **문제 해결**

### 자주 발생하는 에러

#### CUDA Out of Memory
```bash
# 해결: 배치 크기 줄이기
per_device_train_batch_size: 4 → 2
gradient_accumulation_steps: 2 → 4  # 유효 배치 유지
```

#### 모델 다운로드 실패
```bash
# 해결: HuggingFace 토큰 설정
export HF_TOKEN="your_token_here"
```

#### WandB 로그인 실패
```bash
# 해결: .env 파일에 API 키 설정
WANDB_API_KEY="your_wandb_key"
```

## 📈 **모니터링 및 결과 확인**

### 실시간 모니터링
- **콘솔**: 실험 진행 상황, GPU 사용률
- **WandB**: 실시간 차트, 메트릭 비교
- **로그 파일**: 상세한 실행 기록

### 최종 결과 위치
```
outputs/auto_experiments/
├── experiments/               # 실험별 결과
├── models/                   # 모델 체크포인트
└── experiment_summary.json   # 전체 요약
```

## 🎉 **성공 기준**

- **7개 실험 모두 완료**: 에러 없이 실행
- **ROUGE-1 향상**: 베이스라인 10.23% → 20%+ 달성
- **안정적 수렴**: 조기 종료 없이 학습 완료
- **메모리 효율성**: RTX 3090 24GB 내에서 실행

이 가이드를 통해 7개 실험의 전체 흐름과 각 실험의 목적, 사용되는 기술, 예상 결과를 완전히 이해할 수 있습니다.
