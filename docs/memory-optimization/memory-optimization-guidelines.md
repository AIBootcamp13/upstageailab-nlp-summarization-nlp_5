# 메모리 최적화 가이드라인

## 개요
대용량 모델 학습 시 메모리 제약 환경에서의 최적화 전략과 실용적 가이드라인을 제시합니다.

## 메모리 사용량 예측 공식

### GPU 메모리 요구량 계산

```
총 GPU 메모리 = 모델 메모리 + 옵티마이저 메모리 + 그래디언트 메모리 + 배치 메모리

모델 메모리 = 파라미터 수 × 4 bytes (FP32) 또는 2 bytes (FP16)
옵티마이저 메모리 = 모델 메모리 × 1.5 (AdamW 기준)
그래디언트 메모리 = 모델 메모리 × 1.0
배치 메모리 = 배치 크기 × 시퀀스 길이 × 숨겨진 차원 × 4 bytes
```

### 시스템 메모리 요구량 계산

```
총 시스템 메모리 = 기본 프로세스 + DataLoader 워커 + 모델 로딩 + 오버헤드

기본 프로세스 = ~15-20GB
DataLoader 워커 = 워커 수 × 1.5-2GB
모델 로딩 = 모델 크기 × 1.2 (임시 메모리 포함)
오버헤드 = 총합 × 0.1-0.2
```

## 최적화 우선순위

### 1단계: DataLoader 최적화 (가장 효과적)

```yaml
# 권장 워커 수 = 코어 수 / 3
dataloader_num_workers: 16        # 48코어 환경 기준
dataloader_pin_memory: true       # 충분한 RAM이 있는 경우만
dataloader_persistent_workers: true
```

**메모리 절약 효과**: 워커 수 50% 감소 시 ~32GB 절약

### 2단계: 평가 배치 크기 조정

```yaml
# 평가 배치 = 트레이닝 배치 × 1.3 이하
per_device_eval_batch_size: 16    # 트레이닝 배치가 12인 경우
```

**메모리 절약 효과**: 평가 배치 50% 감소 시 ~3-5GB 절약

### 3단계: 트레이닝 배치 크기 조정

```yaml
# gradient_accumulation_steps로 유효 배치 크기 유지
per_device_train_batch_size: 12
gradient_accumulation_steps: 2    # 유효 배치 크기 = 12 × 2 = 24
```

**메모리 절약 효과**: 트레이닝 배치 25% 감소 시 ~5-7GB 절약

## 하드웨어별 권장 설정

### RTX 3090 (24GB VRAM) 환경

| 모델 크기 | 트레이닝 배치 | 평가 배치 | 워커 수 | 예상 사용량 |
|-----------|---------------|-----------|---------|-------------|
| T5-Small (60M) | 32 | 64 | 24 | ~16GB |
| T5-Base (220M) | 20 | 40 | 20 | ~20GB |
| T5-Large (770M) | 12 | 16 | 16 | ~18GB |
| mT5-XL (3.7B) | 8 | 12 | 12 | ~22GB |

### RTX 4090 (24GB VRAM) 환경

| 모델 크기 | 트레이닝 배치 | 평가 배치 | 워커 수 | 예상 사용량 |
|-----------|---------------|-----------|---------|-------------|
| T5-Base (220M) | 24 | 48 | 24 | ~20GB |
| T5-Large (770M) | 16 | 24 | 20 | ~19GB |
| mT5-XL (3.7B) | 12 | 16 | 16 | ~20GB |

## 메모리 부족 문제 해결 체크리스트

### 즉시 확인 사항
- [ ] GPU 메모리 사용량 (`nvidia-smi`)
- [ ] 시스템 메모리 사용량 (`free -h`)  
- [ ] DataLoader 워커 프로세스 수 (`ps aux | grep python`)
- [ ] 콘테이너 메모리 제한 확인

### 단계별 해결 방법

#### 1단계: 환경 설정 최적화
```bash
# CUDA 메모리 단편화 방지
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 메모리 디버깅 활성화
export CUDA_LAUNCH_BLOCKING=1
```

#### 2단계: 설정 조정 (우선순위 순)
1. `dataloader_num_workers` 50% 감소
2. `per_device_eval_batch_size` 30-50% 감소
3. `per_device_train_batch_size` 20-30% 감소
4. `gradient_accumulation_steps` 증가로 유효 배치 보상

#### 3단계: 고급 최적화
```yaml
# 메모리 효율적 옵션들
gradient_checkpointing: true       # 메모리 vs 속도 트레이드오프
fp16: true                         # 메모리 50% 절약 (정확도 약간 손실)
dataloader_drop_last: true         # 마지막 배치 제거로 일관성 확보
```

## 성능 모니터링 지표

### 학습 중 모니터링
```bash
# GPU 사용률 모니터링
watch -n 1 nvidia-smi

# 메모리 사용량 추적  
watch -n 5 'free -h && echo "=== GPU ===" && nvidia-smi --query-gpu=memory.used,memory.total --format=csv'

# 학습 속도 확인
tail -f training.log | grep "it/s"
```

### 성능 지표 목표값
- **GPU 활용률**: 85-100% (학습 중)
- **GPU 메모리 사용률**: 70-85% (여유분 확보)
- **스텝당 처리 시간**: 모델별 기준값 대비 1.5배 이내
- **시스템 메모리**: 제한의 80% 이내

## 트러블슈팅 FAQ

### Q: "CUDA out of memory" 오류
**A**: 배치 크기를 25% 씩 단계적으로 감소
```yaml
per_device_train_batch_size: 16 → 12 → 8 → 6
```

### Q: "DataLoader worker (pid X) is killed by signal 9"
**A**: DataLoader 워커 수 감소
```yaml
dataloader_num_workers: 32 → 16 → 8
```

### Q: 학습 속도가 너무 느림
**A**: 배치 크기와 워커 수의 균형 조정
```yaml
# 속도 우선
dataloader_num_workers: 20
per_device_train_batch_size: 8
gradient_accumulation_steps: 4

# 메모리 우선  
dataloader_num_workers: 12
per_device_train_batch_size: 16
gradient_accumulation_steps: 2
```

### Q: "BrokenPipe" 오류 (one-epoch 모드)
**A**: 전체 학습 모드로 전환
```bash
# 문제 있음
python trainer.py --config config.yaml --one-epoch

# 해결
python trainer.py --config config.yaml
```

## 권장 워크플로우

### 새로운 모델 실험 시

1. **기준선 설정**: 작은 배치로 시작 (배치=4, 워커=4)
2. **점진적 증가**: 메모리 여유 확인하여 25%씩 증가  
3. **최적점 탐색**: GPU 활용률 90% 이상까지 배치 크기 증가
4. **안정성 확인**: 여러 에포크 학습하여 메모리 누수 확인
5. **설정 고정**: 안정적인 설정으로 본 실험 진행

### 메모리 제약 환경 대응

1. **제약 조건 파악**: GPU/시스템 메모리 한계 확인
2. **우선순위 설정**: 학습 안정성 > 속도 > 배치 크기
3. **단계적 최적화**: DataLoader → 배치 크기 → 고급 옵션 순서
4. **성능 측정**: 각 단계별 메모리/속도 지표 기록
5. **문서화**: 최적화 과정과 결과 기록

---

*가이드라인 버전: v1.0*  
*기준 환경: RTX 3090, Ubuntu 20.04, PyTorch 2.0+*  
*최종 업데이트: 2025-08-01*
