# 📊 WandB 통합 가이드 - NLP 요약 프로젝트

## 🔍 기존 분석: nlp-sum-song 프로젝트

### WandB 사용 현황
기존 nlp-sum-song 프로젝트에서 WandB를 다음과 같이 사용하고 있습니다:

```python
# trainer_base.py에서 WandB 초기화
wandb.init(
    entity="skiersong",     # 팀 공유 계정
    project="nlp-5",        # 프로젝트명
    name="실험명_타임스탬프"  # 개별 실험 구분
)
```

### WandB 환경 변수 설정
```python
# 모델 체크포인트 저장 최적화
os.environ["WANDB_LOG_MODEL"]="false"  # storage 절약
os.environ["WANDB_WATCH"]="false"      # 메모리 절약
```

## 🚀 NLP 요약 프로젝트에 WandB 연동하기

### 1단계: WandB 설치 확인

```bash
# UV 환경에서 WandB 설치
uv add wandb

# 또는 requirements.txt에 추가됨
wandb==0.18.5  # Weights & Biases for experiment tracking
```

### 2단계: .env 파일에 WandB API Key 설정

```bash
# .env 파일 생성 (서버에서)
cat >> .env << EOF
# WandB Configuration
WANDB_API_KEY=your_wandb_api_key_here
WANDB_ENTITY=skiersong  # 팀 공유 entity
WANDB_PROJECT=nlp-5     # 프로젝트명
EOF
```

### 3단계: config.yaml에 WandB 설정 업데이트

#### eenzeenee 모델 WandB 설정 수정
```yaml
eenzeenee:
  # ... 기존 설정 ...
  
  # WandB 설정 (팀 공유 계정)
  wandb:
    entity: skiersong  # 팀 공유 entity (nlp-sum-song과 동일)
    project: nlp-5     # 프로젝트명 (nlp-sum-song과 동일)
    name: eenzeenee_korean_summarization
    tags:
      - eenzeenee
      - T5-base
      - Korean
      - Summarization
    notes: "eenzeenee T5 한국어 요약 모델 실험"
```

#### mT5 모델 WandB 설정 수정  
```yaml
xlsum_mt5:
  # ... 기존 설정 ...
  
  # WandB 설정 (팀 공유 계정)
  wandb:
    entity: skiersong
    project: nlp-5
    name: xlsum_mt5_korean_summarization
    tags:
      - mT5
      - XL-Sum
      - Korean
      - Large-Model
      - LoRA
    notes: "XL-Sum mT5 한국어 요약 모델 (LoRA)"
```

## 🚀 실행 방법

### 환경 설정
```bash
# 1. UV 환경에서 의존성 설치
uv sync

# 2. .env 파일 설정
cp .env.template .env
# .env에 팀 WandB API Key 입력

# 3. WandB 로그인 확인
uv run wandb whoami
```

### 실험 실행
```bash
# eenzeenee 모델 실험 (WandB 자동 추적)
uv run python code/trainer.py \
    --config config.yaml \
    --config-section eenzeenee \
    --train-data data/train.csv \
    --val-data data/dev.csv

# mT5 모델 실험 (WandB 자동 추적)
uv run python code/trainer.py \
    --config config.yaml \
    --config-section xlsum_mt5 \
    --train-data data/train.csv \
    --val-data data/dev.csv
```

## 📊 하이퍼파라미터 튜닝 (Sweep)

### Sweep 생성 및 실행
```bash
# eenzeenee 모델 Sweep 생성
uv run wandb sweep config/sweeps/sweep_eenzeenee.yaml

# mT5 모델 Sweep 생성  
uv run wandb sweep config/sweeps/sweep_mt5.yaml

# Sweep 에이전트 실행 (출력된 명령어 사용)
uv run wandb agent skiersong/nlp-5/sweep_id
```

## 📈 추적되는 지표

### 자동 추적 지표
- **손실 함수**: train_loss, eval_loss
- **ROUGE 점수**: rouge-1, rouge-2, rouge-l
- **학습률**: learning rate scheduling
- **시스템 지표**: GPU 사용률, 메모리 사용량
- **하이퍼파라미터**: 모든 설정값 자동 기록

### 실험 비교
WandB 대시보드에서 다음을 비교할 수 있습니다:
- 모델별 성능 비교 (eenzeenee vs mT5)
- 하이퍼파라미터 영향 분석
- 학습 곡선 시각화
- 리소스 사용량 분석

## 🎯 팀 협업 규칙

### 공통 설정
1. **Entity**: 모두 `skiersong` 사용
2. **Project**: 모두 `nlp-5` 사용
3. **API Key**: 팀에서 제공하는 공유 키 사용

### 실험 명명 규칙
- **eenzeenee**: `eenzeenee_korean_summarization`
- **mT5**: `xlsum_mt5_korean_summarization`
- 개별 실험은 timestamp로 자동 구분

### 태그 규칙
- 모델명 (eenzeenee, mT5)
- 기술 스택 (T5-base, LoRA, etc.)
- 언어 (Korean)
- 태스크 (Summarization)

## 🔧 문제 해결

### WandB 로그인 문제
```bash
# 로그인 상태 확인
uv run wandb whoami

# 재로그인
uv run wandb login
```

### API Key 문제
```bash
# 환경변수 직접 설정
export WANDB_API_KEY=your_api_key_here
export WANDB_ENTITY=skiersong
export WANDB_PROJECT=nlp-5
```

### 네트워크 문제 (서버 환경)
```bash
# 오프라인 모드 (임시)
export WANDB_MODE=offline

# 나중에 동기화
uv run wandb sync wandb/offline-run-*
```

## 📚 생성된 파일

본 가이드를 따라 다음 파일들이 생성/수정됩니다:

### 새로 생성된 파일
- ✅ `WANDB_INTEGRATION_GUIDE.md` - 완전한 WandB 연동 가이드
- ✅ `config/sweeps/sweep_eenzeenee.yaml` - eenzeenee 모델 Sweep 설정
- ✅ `config/sweeps/sweep_mt5.yaml` - mT5 모델 Sweep 설정

### 수정된 파일
- ✅ `requirements.txt` - WandB 의존성 추가
- ✅ `config.yaml` - 팀 계정 WandB 설정 완료
- ✅ `.env.template` - WandB API Key 설정 추가

## 🌟 다음 단계

1. **개발 환경에서 테스트**: 로컬에서 작은 데이터셋으로 WandB 연동 테스트
2. **서버 환경 설정**: GPU 서버에서 .env 파일 설정 및 API Key 추가
3. **실험 실행**: eenzeenee와 mT5 모델 각각 실험 시작
4. **Sweep 실행**: 하이퍼파라미터 튜닝으로 최적 설정 탐색
5. **결과 분석**: WandB 대시보드에서 모델 성능 비교 분석

팀 공유 WandB 계정을 통해 모든 실험을 체계적으로 추적하고 비교할 수 있습니다! 📊🚀
