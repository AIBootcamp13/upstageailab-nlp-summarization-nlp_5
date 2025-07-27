```

#### 2.2. 가상환경 재구성

```bash
# 기존 가상환경 정리
rm -rf .venv

# UV로 새 환경 생성
uv venv --python 3.11
source .venv/bin/activate

# 또는 conda 환경 (AIStages)
conda create -n nlp-sum-latest python==3.11 -y
source activate nlp-sum-latest
```

#### 2.3. 핵심 라이브러리 업그레이드

```bash
# 핵심 라이브러리 설치
uv pip install torch==2.6.0 transformers==4.54.0 pytorch_lightning==2.5.2

# 추가 의존성
uv pip install datasets accelerate pandas

# unsloth (Linux/Ubuntu 환경)
# uv pip install unsloth
```

#### 2.4. 설정 파일 업데이트

기존 config.yaml에 다음 성능 최적화 설정을 추가:

```yaml
# 성능 최적화 설정
decoder_max_len: 200  # 기존 100에서 확장
generate_max_length: 200
generation_max_length: 200

# 평가 전략 개선
eval_strategy: steps  # 기존 epoch에서 변경
eval_steps: 400
save_steps: 400

# 메모리 최적화
gradient_checkpointing: true
torch_empty_cache_steps: 10
gradient_checkpointing_kwargs:
  use_reentrant: false

# QLoRA 설정 (새로 추가)
qlora:
  use_unsloth: false  # 환경에 따라 설정
  use_qlora: true
  lora_rank: 16
  lora_alpha: 32
  target_modules: ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"
```

## 검증 및 테스트

### 환경 검증

```bash
# 환경 상태 확인
./check_env.sh

# 예상 출력:
# ✅ Python 3.11.x
# ✅ torch==2.6.0
# ✅ transformers==4.54.0
# ✅ pytorch_lightning==2.5.2
# ✅ QLoRA 지원 확인
```

### 코드 호환성 테스트

```bash
# Python 환경에서 테스트
python -c "
from trainer import DialogueSummarizationTrainer
from inference_engine import InferenceEngine
print('✅ 모든 클래스 정상 로드')
"

# 설정 파일 로드 테스트
python -c "
from config_manager import ConfigManager
config = ConfigManager('config.yaml')
print(f'✅ 설정 로드 성공: decoder_max_len={config.get(\"decoder_max_len\")}')
"
```

## 성능 최적화 효과

### 메모리 효율성

| 모드 | 메모리 사용량 | 절약률 |
|------|-------------|-------|
| 기본 모드 | 100% | 0% |
| QLoRA 모드 | 50-70% | 30-50% |
| unsloth 모드 | 25% | 75% |

### 학습 성능

- **학습 속도**: 20-30% 향상 (group_by_length, dataloader_num_workers)
- **요약 품질**: decoder_max_len 200으로 더 상세한 요약 생성
- **모니터링**: steps 기반 평가로 실시간 성능 추적

## 조장님 실전 경험 기반 팁

### 1. 학습 안정성 확보

- `gradient_checkpointing` 활성화로 메모리 절약
- `eval_strategy: steps`로 세밀한 모니터링
- `torch_empty_cache_steps: 10`으로 메모리 누수 방지

### 2. 성능 최적화

- `group_by_length: true`로 배치 효율성 극대화
- `dataloader_num_workers: 8`로 병렬 처리
- `decoder_max_len: 200`으로 고품질 요약 생성

---

**마이그레이션 완료 후 반드시 확인할 체크리스트:**

- [ ] `./check_env.sh` 실행하여 모든 라이브러리 버전 확인
- [ ] 기존 trainer 클래스 정상 로드 확인
- [ ] config.yaml에서 `decoder_max_len: 200` 설정 확인
- [ ] QLoRA 설정 활성화 확인
- [ ] 첫 번째 학습 실행하여 메모리 사용량 모니터링

> **조장님 프로젝트와의 완전한 통합 달성** 🎉
