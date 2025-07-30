# 🚀 mT5 XLSum 성능 개선 실험 계획서

**목표**: 세계 대회 우승 모델 `csebuetnlp/mT5_multilingual_XLSum`의 성능을 ROUGE-1 25% 이상으로 복원

---

## 📊 현재 상황 분석

### 🚨 문제점
- **현재 성능**: ROUGE-1 10.23%, ROUGE-2 2.42% (세계 대회 수준 대비 -70%)
- **주요 원인**: 하이퍼파라미터 최적화 부족, 도메인 미스매치, 한국어 특화 부족

### 🎯 목표 성능
| 지표 | 현재 | 목표 | 세계 대회 수준 |
|------|------|------|----------------|
| ROUGE-1 | 10.23% | **25%+** | 35-40% |
| ROUGE-2 | 2.42% | **8%+** | 15-20% |
| ROUGE-L | 9.42% | **22%+** | 30-35% |

---

## 🔬 3단계 실험 전략

### 🔥 1단계: 즉시 개선 (우선순위 최고)
**파일**: `config/experiments/01_mt5_xlsum_optimized_v2.yaml`

#### 🎯 핵심 개선사항
```yaml
# 학습률 대폭 증가 (세계 대회 수준)
learning_rate: 8.0e-05  # 기존: 3e-05 → 167% 증가

# 배치 크기 최적화 (AIStages 활용)
per_device_train_batch_size: 4  # 기존: 2 → 100% 증가
gradient_accumulation_steps: 4  # 기존: 8 → 50% 감소

# 한국어 특화 prefix
input_prefix: "요약: "  # 영어 → 한국어

# 향상된 LoRA 설정
lora_rank: 32   # 기존: 16 → 100% 증가
lora_alpha: 64  # 기존: 32 → 100% 증가
```

#### 📈 예상 성능 향상
- **ROUGE-1**: 10% → **18-20%** (80-100% 향상)
- **실행 시간**: ~45분 (AIStages 고성능)

---

### 🔧 2단계: 중급 개선 (도메인 적응)
**파일**: `config/experiments/01_mt5_xlsum_korean_adapted.yaml`

#### 🎯 핵심 개선사항
```yaml
# 극한 학습률 (도메인 적응)
learning_rate: 1.2e-04  # 1단계 대비 50% 추가 증가

# 한국어 대화 특화
input_prefix: "한국어 대화를 요약하세요: "
encoder_max_len: 768    # 긴 대화 처리
decoder_max_len: 150    # 충분한 요약 길이

# 극한 LoRA 설정
lora_rank: 64     # 기존: 32 → 100% 증가
lora_alpha: 128   # 기존: 64 → 100% 증가
```

#### 📈 예상 성능 향상
- **ROUGE-1**: 18-20% → **22-25%** (20-25% 추가 향상)
- **실행 시간**: ~1시간 (8 epoch)

---

### 🚀 3단계: 극한 최적화 (세계 대회 도전)
**파일**: `config/experiments/01_mt5_xlsum_ultimate.yaml`

#### 🎯 핵심 개선사항
```yaml
# 극한 학습률 (위험하지만 고성능)
learning_rate: 2.0e-04  # 최대 학습률

# 최대 배치 및 컨텍스트
per_device_train_batch_size: 8
encoder_max_len: 1024   # 최대 컨텍스트

# 극한 LoRA (최대 표현력)
lora_rank: 128    # 최대 rank
lora_alpha: 256   # 최대 alpha
target_modules: ["q", "k", "v", "o", "wi", "wo", "lm_head", "embed_tokens"]

# 세계 대회 수준 생성 설정
generation_num_beams: 12  # 극한 빔 서치
```

#### 📈 예상 성능 향상
- **ROUGE-1**: 22-25% → **28-32%** (세계 대회 근접)
- **실행 시간**: ~2시간 (12 epoch)

---

## 🏃‍♂️ 실행 순서

### 📅 Day 1: 즉시 개선
```bash
# 1단계 실험 실행
python code/auto_experiment_runner.py --config config/experiments/01_mt5_xlsum_optimized_v2.yaml

# 결과 확인 후 2단계 실행 여부 결정
```

### 📅 Day 2: 중급 + 극한 개선
```bash
# 2단계 실험 실행 (1단계 성공 시)
python code/auto_experiment_runner.py --config config/experiments/01_mt5_xlsum_korean_adapted.yaml

# 3단계 실험 실행 (시간 허용 시)
python code/auto_experiment_runner.py --config config/experiments/01_mt5_xlsum_ultimate.yaml
```

---

## 📊 성공 지표

### ✅ 최소 성공 기준
- **ROUGE-1**: 25% 이상
- **ROUGE-2**: 8% 이상  
- **안정성**: 그래디언트 NaN 없음

### 🏆 이상적 성공 기준
- **ROUGE-1**: 30% 이상 (세계 대회 수준 근접)
- **ROUGE-2**: 12% 이상
- **실용성**: 실제 대화 요약 품질 우수

---

## 🔧 추가 최적화 팁

### 💡 실험 중 모니터링 포인트
1. **Step 200-300**: 빠른 수렴 확인
2. **Gradient Norm**: NaN 발생 시 학습률 조정
3. **Eval Loss**: 과적합 징후 모니터링
4. **ROUGE 추이**: 지속적 향상 확인

### 🚨 실패 시 대안
1. **학습률 조정**: 너무 높으면 50% 감소
2. **배치 크기 조정**: 메모리 부족 시 감소
3. **LoRA 파라미터**: rank/alpha 조정

---

## 📈 예상 결과

### 🎯 성공 시나리오 (80% 확률)
- **1단계**: ROUGE-1 18-20% 달성
- **2단계**: ROUGE-1 22-25% 달성  
- **3단계**: ROUGE-1 28%+ 달성 (세계 대회 수준 근접)

### ⚠️ 부분 성공 시나리오 (20% 확률)
- **1-2단계만 성공**: ROUGE-1 20-22% 달성
- **추가 튜닝**: 3단계 파라미터 조정 후 재실험

---

**다음 실행**: 1단계 실험부터 즉시 시작 권장 🚀
