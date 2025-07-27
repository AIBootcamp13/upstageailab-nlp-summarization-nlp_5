# AIStages 최신 기술 스택 환경 설정 가이드

이 문서는 AI 부트캠프 AIStages 서버에서 조장님의 최신 기술 스택(torch 2.6.0, transformers 4.54.0, unsloth)을 적용한 프로젝트 환경을 설정하는 완전한 가이드입니다.

## 🚀 주요 업그레이드 사항
- **torch**: >=2.0.0 → 2.6.0
- **transformers**: 4.35.2 → 4.54.0 (19버전 대폭 업그레이드!)
- **pytorch_lightning**: 2.1.2 → 2.5.2
- **unsloth 지원**: 고성능 파인튜닝 (Linux 환경 최적화)
- **QLoRA 지원**: 4-bit 양자화로 메모리 30-50% 절약
- **예상 효과**: 학습 속도 20-30% 향상, 메모리 75% 절약

## 주요 단계 요약

### 서버 재생성 미필요 (Python 3.11 사용 중)
1. upstream/song에서 새로운 코드 pull
2. 새로운 requirements.txt로 환경 재설치: `uv pip install -r requirements.txt`

### 서버 재생성 필요 (Python 3.10 + --system 사용)
1. 서버 백업 (outputs 폴더 제외)
2. 서버 재생성
3. 백업 파일 복원
4. 최신 버전 코드 가져오기
5. conda 가상환경 생성: `conda create -n [가상환경 이름] python==3.11 -y`
6. requirements.txt 설치: `uv pip install -r requirements.txt`

---

## 1. conda 가상환경 생성 및 활성화

### 1.1 Python 3.11 가상환경 생성
```bash
# Python 3.11 가상환경 생성
conda create -n nlp-sum-latest python==3.11 -y

# 가상환경 활성화
source activate nlp-sum-latest

# Python 버전 확인
python --version  # Python 3.11.x 확인
```

### 1.2 기존 패키지 정리 (필요시)
```bash
# 특정 패키지만 제거 (전체 제거 불필요)
uv pip uninstall torch torchvision torchaudio transformers

# 또는 새로운 가상환경 이름 사용
conda create -n nlp-sum-new python==3.11 -y
source activate nlp-sum-new
```

---

## 2. AIStages Github 설정

### 2.1 Git 설치 및 설정
```bash
# Git 설치
apt update
apt install -y git

# Git 사용자 정보 설정
git config --global credential.helper store
git config --global user.name "여러분의_깃헙_사용자명"
git config --global user.email "여러분의_깃헙_이메일"
git config --global core.pager "cat"

# Vim 편집기 설치 및 설정
apt install -y vim
git config --global core.editor "vim"
```

### 2.2 설정 확인
```bash
# 설정 확인
git config --list
```

---

## 3. 기타 라이브러리 설치

### 3.1 한국어 폰트 설치
```bash
# 한국어 폰트 설치
apt-get update
apt-get install -y fonts-nanum*

# 폰트 파일 위치 확인
ls /usr/share/fonts/truetype/nanum/Nanum*
```

### 3.2 Curl 설치 (UV 설치용)
```bash
# curl 설치
apt-get install -y curl
```

### 3.3 OpenCV 관련 라이브러리
```bash
# OpenCV 의존성 설치
apt-get install -y libgl1-mesa-glx
apt-get install -y libglib2.0-0
```

> 💡 **팁**: `libglib2.0-0` 설치 시 지역 선택 화면이 나오면 **6번 Asia** → **69번 Seoul**을 선택하세요.

---

## 4. Fork & Clone 설정

### 4.1 GitHub Fork
1. 브라우저에서 팀 레포지토리 접속
2. 우측 상단 **Fork** 버튼 클릭
3. 본인 계정에 Fork된 레포지토리 생성 확인

### 4.2 로컬 Clone
```bash
# HOME 디렉토리로 이동
cd ~/
# AIStages의 HOME 경로: /data/ephemeral/home/

# Fork한 레포지토리 Clone
git clone https://github.com/[본인_깃헙_계정]/upstageailab-nlp-summarization-nlp_5.git

# 프로젝트 디렉토리로 이동
cd upstageailab-nlp-summarization-nlp_5
```

### 4.3 Upstream 설정
```bash
# 팀 레포지토리를 upstream으로 설정
git remote add upstream https://github.com/AIBootcamp13/upstageailab-nlp-summarization-nlp_5.git

# Push 방지 설정 (실수 방지)
git remote set-url --push upstream no-push

# Remote 확인
git remote -v
```

### 4.4 Upstream 동기화
```bash
# 팀 레포지토리의 최신 변경사항 가져오기
git fetch upstream main && git merge FETCH_HEAD
```

---

## 5. UV 설정 및 사용

### 5.1 UV 설치
```bash
# HOME 디렉토리로 이동
cd ~/

# UV 설치 스크립트 실행
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 5.2 환경변수 설정
```bash
# .bashrc 파일 편집
vim ~/.bashrc

# 다음 줄 추가 (i 키를 눌러 편집 모드)
export PATH="/data/ephemeral/home/.local/bin:$PATH"

# 저장 및 종료 (ESC → :wq)

# 변경사항 적용
source ~/.bashrc
```

### 5.3 설치 확인
```bash
# UV 버전 확인
uv --version
```

### 5.4 최신 기술 스택 설치 (필수)

> 📌 **중요**: conda 가상환경에 설치합니다. `--system` 옵션 절대 사용 금지!

#### 최신 기술 스택 설치
```bash
# 프로젝트 폴더로 이동
cd ~/upstageailab-nlp-summarization-nlp_5/nlp-sum-lyj

# conda 가상환경에 최신 라이브러리 설치
uv pip install -r requirements.txt

# 설치 위치 확인 - conda 환경에 설치되는지 확인
# /opt/conda/envs/[nlp-sum-latest]/lib/python3.11/site-packages/
```

### 5.5 최신 기술 스택 설치 확인
```bash
# 최신 라이브러리 버전 및 CUDA 지원 확인
python -c "
import torch, transformers, pytorch_lightning
print(f'✅ torch: {torch.__version__}')
print(f'✅ transformers: {transformers.__version__}')
print(f'✅ pytorch_lightning: {pytorch_lightning.__version__}')
print(f'✅ CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
"

# QLoRA 지원 확인
python -c "
try:
    import peft, bitsandbytes
    print('✅ QLoRA 지원 (peft + bitsandbytes)')
except ImportError:
    print('❌ QLoRA 지원 없음')
"

# unsloth 고성능 파인튜닝 확인
python -c "
try:
    import unsloth
    print('✅ unsloth 지원 (고성능 파인튜닝 활성화)')
except ImportError:
    print('❌ unsloth 설치 실패, QLoRA 모드 사용')
"

# 전체 환경 검증
./check_env.sh
```

---

## 6. Config 파일 설정

### 6.1 Config 파일 생성
```bash
# Jupyter Notebook 또는 Python 스크립트로 실행
jupyter notebook src/configs/generate_config.ipynb
```

### 6.2 주요 설정 항목

#### 6.2.1 프로젝트 경로
```yaml
project_dir: "/data/ephemeral/home/upstageailab-nlp-summarization-nlp_5"
output_dir: "./outputs/exp_baseline_001"
```

#### 6.2.2 일반 설정
```yaml
general:
  data_path: "./data/"
  model_name: "digit82/kobart-summarization"
  output_path: "./outputs/"
```

#### 6.2.3 Tokenizer 설정
```yaml
tokenizer:
  encoder_max_len: 1024  # 모델에 따라 조정
  decoder_max_len: 128
  special_tokens:
    - "#Person1#"
    - "#Person2#"
    # ... 추가 토큰
```

#### 6.2.4 학습 설정
```yaml
training:
  seed: 42
  save_total_limit: 3
  save_eval_log_steps: 500
  num_train_epochs: 20
  per_device_train_batch_size: 16  # GPU 메모리에 따라 조정
  generation_max_length: 128
  early_stopping_patience: 3
  learning_rate: 3e-5
  warmup_ratio: 0.1
```

#### 6.2.5 WandB 설정
```yaml
wandb:
  entity: "your_wandb_team"
  project: "dialogue-summarization"
  name: "kobart-baseline-v1"
  group: "baseline"
  notes: "KoBART baseline with special tokens"
```

#### 6.2.6 추론 설정
```yaml
inference:
  batch_size: 32
  num_beams: 4
  no_repeat_ngram_size: 2
```

---

## 7. Main 파일 실행

### 7.1 학습 실행
```bash
# 프로젝트 루트 디렉토리에서 실행
python src/main_base.py --config config_base_0725140000.yaml
```

### 7.2 추론만 실행
```bash
# 이미 학습된 모델로 추론
python src/main_base.py --config config_base_0725140000.yaml --inference True
```

### 7.3 실행 모니터링
```bash
# 로그 실시간 확인
tail -f outputs/exp_baseline_001/train.log

# GPU 사용량 모니터링
watch -n 1 nvidia-smi
```

---

## 8. 기존 프로젝트와의 통합

### 8.1 현재 프로젝트 구조와의 매핑

현재 `nlp-sum-lyj` 프로젝트 구조:
```
nlp-sum-lyj/
├── code/
│   ├── baseline.ipynb      # 기존 베이스라인
│   ├── config.yaml         # 기존 설정
│   └── requirements.txt    # 의존성
├── data/
└── docs/
```

새로운 구조 통합 방안:
```
nlp-sum-lyj/
├── src/                    # 새로 추가
│   ├── main_base.py
│   ├── configs/
│   │   └── generate_config.ipynb
│   └── ...
├── code/                   # 기존 유지
├── data/
└── docs/
    └── setup_guides/       # 새로 추가
        └── aistages_environment_setup.md
```

### 8.2 기존 코드 적용 방법

#### 8.2.1 기존 baseline.ipynb 사용
```python
# 기존 코드의 config 수정
config = {
    'general': {
        'data_path': './data/',
        'model_name': 'digit82/kobart-summarization',
        'output_path': './outputs/'
    },
    # ... 새로운 config 형식에 맞춰 수정
}
```

#### 8.2.2 UV 환경에서 실행
```bash
# UV로 설치한 환경에서 Jupyter 실행
jupyter notebook code/baseline.ipynb
```

### 8.3 권장 워크플로우

1. **환경 설정**: 이 문서의 1-5단계 진행
2. **기존 코드 실행**: `code/baseline.ipynb`로 초기 실험
3. **새 구조 적용**: `src/main_base.py` 구조로 전환
4. **실험 관리**: WandB로 체계적인 실험 추적

### 8.4 트러블슈팅

#### UV 설치 실패
```bash
# 대안: pip으로 UV 설치
pip install uv
```

#### CUDA 오류
```bash
# PyTorch CUDA 버전 확인
python -c "import torch; print(torch.cuda.is_available())"
```

#### 메모리 부족
```yaml
# config에서 batch size 감소
per_device_train_batch_size: 8
gradient_accumulation_steps: 2
```

---

## 요약

이 가이드는 AIStages 환경에서 NLP 요약 프로젝트를 설정하는 완전한 과정을 담고 있습니다. 주요 차이점:

1. **UV 사용**: pip보다 10-100배 빠른 패키지 설치
2. **시스템 설치**: Docker 환경이므로 `--system` 옵션 사용
3. **경로 설정**: AIStages의 특수 경로 구조 고려
4. **Git 워크플로우**: Fork & Clone 방식의 협업

기존 프로젝트 구조를 유지하면서 새로운 환경 설정을 적용할 수 있으며, 필요에 따라 점진적으로 마이그레이션할 수 있습니다.
