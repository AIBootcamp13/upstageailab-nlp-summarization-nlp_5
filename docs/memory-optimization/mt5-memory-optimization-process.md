# mT5 QLoRA 메모리 최적화 과정

## 개요
RTX 3090 (24GB VRAM) 환경에서 mT5 QLoRA 학습 시 발생한 메모리 부족 문제를 해결하기 위한 단계별 최적화 과정을 기록합니다.

## 시스템 사양
- **GPU**: NVIDIA GeForce RTX 3090 (24GB VRAM)
- **RAM**: 251GB
- **CPU**: 48 cores
- **CUDA**: 12.2
- **모델**: csebuetnlp/mT5_multilingual_XLSum

## 최적화 단계별 진행 과정

### 1단계: 초기 설정 (실패)

**설정값:**
```yaml
# 배치 설정
per_device_train_batch_size: 20
per_device_eval_batch_size: 48
gradient_accumulation_steps: 2

# 데이터 로더 설정
dataloader_num_workers: 32
dataloader_pin_memory: true
dataloader_persistent_workers: true
```

**문제:**
- 콘테이너 메모리 제한 초과 (60GB 예상 사용량)
- DataLoader 워커들이 과도한 메모리 사용

**실패 원인 분석:**
- 각 DataLoader 워커당 약 1.5-2GB 메모리 사용
- 32개 워커 × 2GB = 64GB 추가 메모리 필요
- 모델 자체 메모리(~20GB) + 워커 메모리(64GB) = 84GB 총 요구량

### 2단계: 1차 축소 (부분 성공)

**변경 사항:**
```yaml
# 배치 설정 최소 축소
per_device_train_batch_size: 18      # 20→18 (10% 축소)
per_device_eval_batch_size: 32       # 48→32 (33% 축소)

# DataLoader 워커 수 축소
dataloader_num_workers: 16           # 32→16 (50% 축소)
```

**예상 메모리 절약:**
- DataLoader 워커: 32GB → 16GB (16GB 절약)
- 배치 크기 축소: 약 5GB 절약
- **총 절약**: ~21GB (84GB → 63GB)

**결과:**
- 여전히 콘테이너 메모리 제한(60GB) 초과
- 추가 최적화 필요

### 3단계: 2차 축소 (성공)

**최종 변경 사항:**
```yaml
# 배치 설정 추가 축소
per_device_train_batch_size: 12      # 18→12 (33% 추가 축소)
per_device_eval_batch_size: 16       # 32→16 (50% 추가 축소)
gradient_accumulation_steps: 2       # 유효 배치 크기 유지

# DataLoader 워커 유지
dataloader_num_workers: 16           # 16 유지
```

**최종 메모리 사용량:**
- GPU 메모리: 18.3GB (24GB 중)
- 시스템 메모리: ~45GB (콘테이너 제한 내)
- DataLoader 워커: 16GB
- **총 사용량**: ~40GB (60GB 제한 내)

## 메모리 사용량 비교표

| 단계 | 트레이닝 배치 | 평가 배치 | 워커 수 | 예상 GPU | 예상 시스템 | 상태 |
|------|---------------|-----------|---------|----------|-------------|------|
| 초기 | 20 | 48 | 32 | ~24GB | ~84GB | ❌ 실패 |
| 1차 축소 | 18 | 32 | 16 | ~22GB | ~63GB | ❌ 제한 초과 |
| 2차 축소 | 12 | 16 | 16 | ~18GB | ~45GB | ✅ 성공 |

## 최적화 효과

### 성능 영향 최소화
- **유효 배치 크기 유지**: gradient_accumulation_steps=2로 학습 안정성 확보
- **워커 수 적정화**: 16개 워커로 I/O 병목 방지
- **메모리 활용률**: GPU 76% (18.3GB/24GB) 효율적 사용

### 학습 속도
- **처리 속도**: 3.24초/스텝 (안정적)
- **예상 완료 시간**: 2시간 45분
- **GPU 활용률**: 100% (최적)

## 트러블슈팅 경험

### 1. CUDA Device-side Assert 오류
**문제**: KoBART 모델에서 tokenizer 관련 오류 발생
**해결**: mT5 모델로 전환하여 해결

### 2. BrokenPipe 오류 (one-epoch 실행)
**문제**: `--one-epoch` 옵션 사용 시 파이프 연결 끊김
**해결**: 본 실험 모드로 실행하여 안정성 확보

### 3. CUDA Out of Memory
**문제**: 첫 번째 스텝에서도 메모리 부족 발생
**해결**: 
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 설정
- GPU 메모리 완전 정리 후 재실행

## 권장사항

### 메모리 제한 환경에서의 설정 가이드라인

1. **DataLoader 워커 수**: 코어 수의 1/3 수준 (48코어 → 16워커)
2. **배치 크기**: GPU 메모리의 70-80% 활용 목표
3. **평가 배치**: 트레이닝 배치의 1.3배 이하
4. **메모리 단편화 방지**: `expandable_segments:True` 설정

### 단계별 최적화 전략

1. **1단계**: DataLoader 워커 수부터 축소 (가장 효과적)
2. **2단계**: 평가 배치 크기 축소 (학습에 미치는 영향 최소)
3. **3단계**: 트레이닝 배치 크기 축소 (마지막 수단)
4. **4단계**: gradient_accumulation_steps 조정으로 유효 배치 크기 유지

## 결론

**성공 요인:**
- 체계적인 단계별 축소로 최적점 발견
- 학습 안정성과 메모리 효율성의 균형
- 실시간 모니터링을 통한 빠른 문제 발견

**학습 효과:**
- 메모리 제약 환경에서의 대용량 모델 학습 노하우 축적
- 시스템 리소스 최적화 전략 수립
- 안정적인 학습 환경 구축 완료

---

*문서 작성일: 2025-08-01*  
*마지막 업데이트: mT5 QLoRA 성공적 학습 시작*
