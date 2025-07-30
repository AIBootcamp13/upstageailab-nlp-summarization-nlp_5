# 🚀 GPU 메모리별 권장 배치 크기 가이드

## 📋 모델별 메모리 요구사항

### mT5 모델 (1.2B 파라미터)
- **모델 가중치**: ~2.4GB
- **학습시 메모리**: ~10-15GB 
- **추론시 메모리**: ~4-6GB

### eenzeenee 모델 (220M 파라미터)  
- **모델 가중치**: ~440MB
- **학습시 메모리**: ~3-5GB
- **추론시 메모리**: ~1-2GB

## ⚙️ GPU 메모리별 권장 배치 크기

### 🔥 V100 (16GB) - AIStages 일반적 사양
```yaml
# mT5 모델 설정
xlsum_mt5:
  training:
    per_device_train_batch_size: 1    # 안전
    per_device_eval_batch_size: 2     # 적절
    gradient_accumulation_steps: 4    # 효과적 배치=4
  inference:
    batch_size: 2                     # 안전

# eenzeenee 모델 설정  
eenzeenee:
  training:
    per_device_train_batch_size: 4    # 현재 설정 (적절)
    per_device_eval_batch_size: 4     # 현재 설정 (적절)
    gradient_accumulation_steps: 2    # 효과적 배치=8
  inference:
    batch_size: 8                     # 현재 설정 (적절)
```

### 🚀 A100 (40GB) - 고성능 서버
```yaml
# mT5 모델 설정
xlsum_mt5:
  training:
    per_device_train_batch_size: 4    # 권장
    per_device_eval_batch_size: 8     # 권장
    gradient_accumulation_steps: 2    # 효과적 배치=8
  inference:
    batch_size: 8                     # 권장

# eenzeenee 모델 설정
eenzeenee:
  training:
    per_device_train_batch_size: 8    # 향상 가능
    per_device_eval_batch_size: 8     # 향상 가능
    gradient_accumulation_steps: 2    # 효과적 배치=16
  inference:
    batch_size: 16                    # 향상 가능
```

### 💾 RTX 3080/4090 (10-24GB) - 로컬 개발
```yaml
# mT5 모델 설정 (주의 필요)
xlsum_mt5:
  training:
    per_device_train_batch_size: 1    # 최소
    per_device_eval_batch_size: 1     # 최소  
    gradient_accumulation_steps: 8    # 효과적 배치=8
  inference:
    batch_size: 1                     # 최소

# eenzeenee 모델 설정
eenzeenee:
  training:
    per_device_train_batch_size: 2    # 안전
    per_device_eval_batch_size: 4     # 추론은 더 여유
  inference:
    batch_size: 4                     # 적절
```

## 🛠️ 메모리 최적화 기법

### 1. Gradient Accumulation 활용
```yaml
# 실제 배치 크기 = per_device_batch_size × gradient_accumulation_steps
per_device_train_batch_size: 2
gradient_accumulation_steps: 4  # 효과적 배치 크기 = 8
```

### 2. 혼합 정밀도 학습
```yaml
training:
  fp16: true          # 메모리 50% 절약
  # 또는
  bf16: true          # A100에서 권장
```

### 3. DeepSpeed 활용 (필요시)
```yaml
training:
  deepspeed: "ds_config.json"
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
```

## 🧪 배치 크기 테스트 방법

### 1. GPU 메모리 확인
```bash
# 현재 GPU 상태 확인
nvidia-smi

# 메모리 사용량 모니터링 
watch -n 1 nvidia-smi
```

### 2. 단계별 테스트
```bash
# 작은 배치부터 시작
uv run python code/trainer.py \
    --config config.yaml \
    --config-section eenzeenee \
    --max_steps 10 \
    --save_steps 1000  # 저장 방지

# 메모리 사용량 확인 후 점진적 증가
```

### 3. 자동 배치 크기 찾기
```python
# Trainer에서 자동 탐지 (실험적)
training_args = TrainingArguments(
    auto_find_batch_size=True,  # 자동 배치 크기 탐지
    # ... 기타 설정
)
```

## ⚠️ 주의사항

### 메모리 부족 징후
```bash
❌ CUDA out of memory
❌ RuntimeError: unable to create new native thread  
❌ Killed (프로세스 종료)
```

### 대응 방법
1. **배치 크기 절반으로 감소**
2. **gradient_accumulation_steps 2배 증가**  
3. **sequence length 단축** (512 → 256)
4. **fp16/bf16 활성화**

## 📊 성능 vs 메모리 트레이드오프

| 배치 크기 | 학습 속도 | 메모리 사용량 | 모델 성능 |
|-----------|-----------|---------------|-----------|
| 1 | 느림 | 최소 | 불안정 |
| 2-4 | 보통 | 적정 | 안정 |
| 8-16 | 빠름 | 많음 | 최적 |
| 32+ | 매우 빠름 | 과다 | 수렴 어려움 |

## 🎯 결론

**조장님의 조언대로 모델 크기에 따른 배치 크기 조정이 필수입니다:**

1. **mT5 (1.2B)**: 배치 크기 1-4 권장
2. **eenzeenee (220M)**: 현재 설정(4-8) 적절  
3. **gradient_accumulation_steps**로 효과적 배치 크기 확보
4. **GPU 메모리 모니터링** 필수

**AIStages 서버에서 안전한 실험을 위해 보수적인 배치 크기부터 시작하세요!** 🚀
