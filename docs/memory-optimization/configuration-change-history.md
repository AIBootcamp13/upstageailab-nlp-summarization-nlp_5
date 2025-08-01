# mT5 QLoRA 설정 변경 이력

## 설정 파일 경로
`config/experiments/01_mt5_xlsum_ultimate_korean_qlora.yaml`

## 변경 이력

### 초기 설정 (2025-08-01 06:00)

```yaml
# 🔥 1단계 극한 최적화 배치 설정
per_device_train_batch_size: 20      # 초기 설정
per_device_eval_batch_size: 48       # 48GB 평가 배치
gradient_accumulation_steps: 2       # 유효 배치 80 유지

# 🚀 고성능 DataLoader 설정
dataloader_num_workers: 32           # 48코어의 67% 활용
dataloader_pin_memory: true          # 251GB RAM 활용
dataloader_persistent_workers: true  # 워커 재사용
```

**문제점:**
- 예상 메모리 사용량: ~84GB
- 콘테이너 메모리 제한: 60GB
- 초과량: 24GB

---

### 1차 축소 (2025-08-01 06:25) - Commit: 743d281

```yaml
# 🔥 1단계 극한 최적화 배치 설정
per_device_train_batch_size: 16      # 20→16 최소 축소
per_device_eval_batch_size: 32       # 48→32 평가 배치 축소
gradient_accumulation_steps: 2       # 유효 배치 48 유지

# 🚀 고성능 DataLoader 설정 
dataloader_num_workers: 16           # 32→16 메모리 절약
dataloader_pin_memory: true          # 251GB RAM 활용
dataloader_persistent_workers: true  # 워커 재사용
```

**변경 사항:**
- `per_device_train_batch_size`: 20 → 16 (20% 축소)
- `per_device_eval_batch_size`: 48 → 32 (33% 축소)  
- `dataloader_num_workers`: 32 → 16 (50% 축소)

**예상 효과:**
- DataLoader 메모리: 64GB → 32GB (-32GB)
- 배치 메모리: ~5GB 절약
- 총 절약: ~37GB (84GB → 47GB)

**결과:** 여전히 한계치 근접, 추가 최적화 필요

---

### 2차 축소 (2025-08-01 06:33) - Commit: 58f3c99

```yaml
# 🔥 1단계 극한 최적화 배치 설정
per_device_train_batch_size: 12      # 16→12 메모리 절약
per_device_eval_batch_size: 16       # 32→16 평가 배치 추가 축소
gradient_accumulation_steps: 2       # 유효 배치 48 유지

# 🚀 고성능 DataLoader 설정
dataloader_num_workers: 16           # 16 유지 (최적화 완료)
dataloader_pin_memory: true          # 251GB RAM 활용
dataloader_persistent_workers: true  # 워커 재사용
```

**변경 사항:**
- `per_device_train_batch_size`: 16 → 12 (25% 추가 축소)
- `per_device_eval_batch_size`: 32 → 16 (50% 추가 축소)
- `dataloader_num_workers`: 16 유지

**예상 효과:**
- 트레이닝 배치 메모리: ~5GB 절약
- 평가 배치 메모리: ~3GB 절약
- 총 추가 절약: ~8GB (47GB → 39GB)

**최종 결과:** ✅ 성공적 학습 시작
- GPU 메모리: 18.3GB/24GB (76% 활용)
- 시스템 메모리: ~40GB (60GB 제한 내)

## 설정 비교 매트릭스

| 설정 항목 | 초기값 | 1차 축소 | 2차 축소 | 축소율 |
|-----------|--------|----------|----------|--------|
| 트레이닝 배치 | 20 | 16 | 12 | -40% |
| 평가 배치 | 48 | 32 | 16 | -67% |
| DataLoader 워커 | 32 | 16 | 16 | -50% |
| 유효 배치 크기* | 40 | 32 | 24 | -40% |

*유효 배치 크기 = per_device_train_batch_size × gradient_accumulation_steps

## 메모리 사용량 추정

### GPU 메모리 (24GB 총량)

| 구성 요소 | 초기 | 1차 축소 | 2차 축소 |
|-----------|------|----------|----------|
| 모델 파라미터 | ~8GB | ~8GB | ~8GB |
| 옵티마이저 상태 | ~4GB | ~4GB | ~4GB |
| 그래디언트 | ~4GB | ~4GB | ~4GB |
| 배치 데이터 | ~8GB | ~6GB | ~4GB |
| **총 사용량** | ~24GB | ~22GB | ~18GB |
| **여유 공간** | 0GB | 2GB | 6GB |

### 시스템 메모리 (60GB 제한)

| 구성 요소 | 초기 | 1차 축소 | 2차 축소 |
|-----------|------|----------|----------|
| 기본 프로세스 | ~20GB | ~20GB | ~20GB |
| DataLoader 워커 | ~64GB | ~32GB | ~32GB |
| 기타 오버헤드 | ~5GB | ~5GB | ~5GB |
| **총 사용량** | ~89GB | ~57GB | ~57GB |
| **제한 준수** | ❌ | ❌ | ✅ |

## 학습 성능 영향 분석

### 처리량 (Throughput)
- **초기 목표**: 6.25 samples/sec (20 × 2 × 8 steps/min)
- **최종 달성**: 4.17 samples/sec (12 × 2 × 8 steps/min)
- **성능 감소**: 33% (메모리 안정성과 트레이드오프)

### 학습 시간
- **스텝당 처리 시간**: 3.24초 (안정적)
- **총 예상 시간**: 2시간 45분 (3120 스텝)
- **GPU 활용률**: 100% (최적)

## 최적화 교훈

### 효과적인 최적화 순서
1. **DataLoader 워커 수** (가장 큰 메모리 절약 효과)
2. **평가 배치 크기** (학습에 미치는 영향 최소)
3. **트레이닝 배치 크기** (학습 성능에 직접적 영향)

### 메모리 모니터링 포인트
- 콘테이너 메모리 제한 vs 실제 사용량
- GPU 메모리 활용률 (70-80% 목표)
- DataLoader 워커당 메모리 사용량 (~2GB)

### 안정성 확보 방법
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- GPU 메모리 완전 정리 후 재시작
- 점진적 배치 크기 증가 테스트

---

*변경 이력 추적: Git commits 743d281, 58f3c99*  
*최종 성공 확인: 2025-08-01 06:50*
