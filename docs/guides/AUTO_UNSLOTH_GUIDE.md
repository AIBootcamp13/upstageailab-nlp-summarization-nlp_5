# 🚀 새로운 모델 자동 Unsloth 활성화 가이드

## 📊 시스템 개요

이제 **모든 새로운 모델**이 추가될 때 **환경을 자동으로 감지**하고, **Ubuntu + CUDA 환경**에서는 **자동으로 Unsloth를 활성화**합니다.

## ⚡ 자동 활성화 조건

### ✅ **Unsloth 자동 활성화 조건**
1. **OS**: Ubuntu (Linux)
2. **CUDA**: 사용 가능 (torch.cuda.is_available() == True)
3. **GPU 메모리**: 6GB 이상
4. **CUDA 버전**: 11.8 이상
5. **Unsloth 패키지**: 설치됨

### 🎯 **자동 최적화 적용**
- **RTX 3090 (24GB)**: batch_size=12, bf16=True
- **RTX 4080 (16GB)**: batch_size=8, fp16=True  
- **RTX 4070 (12GB)**: batch_size=6, fp16=True
- **RTX 4060 (8GB)**: batch_size=4, fp16=True

## 🔧 새로운 모델 설정 방법

### 1. **설정 파일에서 명시적 활성화**
```yaml
# config/new_model.yaml
model:
  architecture: t5
  checkpoint: my-new-t5-model

qlora:
  use_unsloth: true    # 명시적 활성화
  use_qlora: true
```

### 2. **환경 자동 감지 의존 (권장)**
```yaml
# config/new_model.yaml
model:
  architecture: bart
  checkpoint: my-new-bart-model

qlora:
  # use_unsloth를 명시하지 않음
  # → Ubuntu + CUDA 환경에서 자동 활성화
  use_qlora: true
```

### 3. **자동 감지 무시 (비활성화)**
```yaml
# config/new_model.yaml
model:
  architecture: gpt2
  checkpoint: my-new-gpt2-model

qlora:
  use_unsloth: false   # 명시적 비활성화
  use_qlora: true
```

## 🎯 우선순위 규칙

```python
# Unsloth 활성화 결정 로직
config_use_unsloth = qlora_config.get('use_unsloth', False)
auto_use_unsloth = auto_config.get('use_unsloth', False)

# 최종 결정: 설정파일 OR 자동감지
use_unsloth = (config_use_unsloth or auto_use_unsloth) and UNSLOTH_AVAILABLE
```

**우선순위:**
1. **설정 파일 명시** (`use_unsloth: true/false`)
2. **환경 자동 감지** (Ubuntu + CUDA + GPU 메모리 충분)
3. **기본값** (`false`)

## 📱 실행 시 로그 예시

### ✅ **AIStages 서버 (자동 활성화)**
```
🔍 자동 환경 감지 결과
============================================================
OS: Linux (Ubuntu 20.04)
Python: 3.11.13
CPU Cores: 48
🎮 CUDA: Available (v12.6)
GPU Count: 1
  - NVIDIA GeForce RTX 3090: 24.0GB

⚡ Unsloth 지원
추천 여부: ✅ 추천
설치 상태: ✅ 설치됨

🚀 자동 최적화 설정
use_unsloth: True
recommended_batch_size: 12
fp16: False, bf16: True
dataloader_num_workers: 8
============================================================

🚀 환경 자동 감지: Ubuntu + CUDA 환경에서 Unsloth 자동 활성화
📊 기본 배치 크기 권장: 12
Loading model: my-new-model (t5)
QLoRA enabled: True, unsloth enabled: True
🚀 unsloth로 고효율 모델 로딩 중...
```

### ❌ **macOS (자동 비활성화)**
```
🔍 자동 환경 감지 결과
============================================================
OS: Darwin (macOS 14.0)
Python: 3.11.13
🎮 CUDA: Not Available

⚡ Unsloth 지원
추천 여부: ❌ 비추천
설치 상태: ❌ 미설치

🚀 자동 최적화 설정
use_unsloth: False
recommended_batch_size: 2
fp16: False, bf16: False
dataloader_num_workers: 0
============================================================

Loading model: my-new-model (t5)
QLoRA enabled: True, unsloth enabled: False
Loading model with standard QLoRA...
```

## 🔄 기존 모델에 미치는 영향

### ✅ **이미 활성화된 모델들**
- 기존 설정 파일의 `use_unsloth: true`는 그대로 유지
- 추가적인 자동 최적화 효과 적용

### 🔧 **설정 없던 모델들**
- AIStages 서버에서는 자동으로 Unsloth 활성화
- macOS/Windows에서는 기존과 동일하게 비활성화

## 🧪 테스트 방법

### 1. **환경 감지 테스트**
```bash
python test_auto_environment.py
```

### 2. **새로운 모델 설정 테스트**
```yaml
# config/test_new_model.yaml
general:
  model_name: facebook/bart-base
  
# qlora 섹션 없음 → 자동 감지 적용
```

```bash
python code/trainer.py --config config/test_new_model.yaml
```

### 3. **로그 확인**
```bash
# 환경 감지 로그 확인
tail -f logs/training.log | grep "환경 자동 감지"

# Unsloth 활성화 로그 확인  
tail -f logs/training.log | grep "unsloth"
```

## 🎯 실제 적용 예시

### 새로운 KoGPT 모델 추가
```yaml
# config/experiments/new_kogpt_experiment.yaml
experiment_name: kogpt_new_test
description: "새로운 KoGPT 모델 테스트"

general:
  model_name: skt/kogpt2-base-v2
  model_type: causal_lm

# qlora 설정 없음 → AIStages에서 자동 Unsloth 활성화!

training:
  num_train_epochs: 3
  # batch_size는 자동 권장값 (RTX 3090: 12) 참고
```

**결과**: AIStages 서버에서 실행 시 자동으로 Unsloth 활성화, 메모리 절약 및 속도 향상!

## 💡 권장사항

### ✅ **새로운 모델 설정 시**
1. **qlora 섹션을 비워두기** (자동 감지 활용)
2. **환경별 테스트** 후 명시적 설정 고려
3. **로그를 확인**하여 최적화 적용 여부 검증

### 🎯 **팀 협업 시**
1. **AIStages 서버**: 자동 활성화 기대
2. **개발 환경 (macOS)**: 자동 비활성화 예상
3. **설정 통일 필요시**: 명시적으로 `use_unsloth: true/false` 설정

---

**🚀 결론**: 이제 새로운 모델을 추가할 때마다 별도 설정 없이도 **환경에 맞는 최적화가 자동 적용**됩니다!
