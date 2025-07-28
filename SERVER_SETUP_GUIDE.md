# 🖥️ AIStages Ubuntu 서버 설정 가이드 (UV 환경)

## 📋 개요

맥(Mac)에서 개발한 프로젝트를 AIStages Ubuntu 서버로 전송하여 실험을 실행하는 가이드입니다.
**UV 패키지 매니저**를 사용하여 의존성을 관리하며, pytorch_model.bin 파일을 포함한 대용량 모델 파일들은 자동 다운로드를 통해 처리합니다.

## 🔄 Git 전송 워크플로우

### 1단계: 로컬에서 Git 준비

```bash
# 현재 상태 확인
git status

# 변경사항 스테이징 (pytorch_model.bin은 .gitignore로 제외됨)
git add .

# 커밋
git commit -m "feat: eenzeenee 모델 완전 통합 및 문서화 완료

- eenzeenee_utils.py 전용 유틸리티 모듈 추가 (12개 함수)
- 정확한 모델명으로 수정: eenzeenee/t5-base-korean-summarization  
- config.yaml 최적 설정값 적용 (64토큰, 3빔)
- EENZEENEE_INTEGRATION_REPORT.md 상세 통합 보고서
- EENZEENEE_GUIDE.md 완전 사용자 가이드 
- 4/4 통합 테스트 통과"

# 원격 저장소에 푸시
git push origin main
```

### 2단계: AIStages 서버에서 받기

```bash
# 서버에 SSH 접속
ssh username@your-aistages-server

# 프로젝트 클론 (처음) 또는 풀 (기존)
git clone [your-repository-url]
# 또는
git pull origin main

# 프로젝트 디렉토리로 이동
cd nlp-sum-lyj
```

### 3단계: UV를 사용한 서버 환경 설정

```bash
# UV가 설치되어 있는지 확인
uv --version

# UV가 없다면 설치
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # 또는 ~/.zshrc

# UV를 사용하여 가상환경 생성 및 의존성 설치
uv sync

# 또는 기존 방식으로도 가능
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# uv.lock 파일이 있다면 (권장)
uv sync --frozen

# 환경 변수 설정 (필요시)
cp .env.template .env
# .env 파일 편집
```

## 🚀 UV 명령어 참조

### 기본 UV 명령어
```bash
# 의존성 설치
uv add package-name

# 개발 의존성 설치  
uv add --dev package-name

# 의존성 동기화
uv sync

# 스크립트 실행
uv run python script.py

# 가상환경 활성화
source .venv/bin/activate
# 또는
uv shell
```

### 프로젝트별 UV 사용법
```bash
# 실험 실행시 UV 사용
uv run python code/trainer.py --config config.yaml

# 테스트 실행
uv run python test_eenzeenee_integration.py

# 유틸리티 함수 테스트
uv run python -c "from code.utils.eenzeenee_utils import *; print('UV 환경 테스트 성공!')"
```

## 🤖 모델 자동 다운로드 처리

### pytorch_model.bin 파일 특성
- **mT5 모델**: 2.2GB
- **eenzeenee 모델**: ~800MB 
- **자동 다운로드**: 첫 실행시 Hugging Face Hub에서 자동 다운로드
- **캐시 위치**: `~/.cache/huggingface/transformers/`

### UV를 사용한 자동 다운로드 테스트

```bash
# UV 환경에서 모델 다운로드 테스트 (실제 학습 전 권장)
uv run python -c "
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

print('📥 mT5 모델 다운로드 중...')
mT5_model = AutoModelForSeq2SeqLM.from_pretrained('csebuetnlp/mT5_multilingual_XLSum')
mT5_tokenizer = AutoTokenizer.from_pretrained('csebuetnlp/mT5_multilingual_XLSum')
print('✅ mT5 모델 다운로드 완료')

print('📥 eenzeenee 모델 다운로드 중...')  
eenzeenee_model = AutoModelForSeq2SeqLM.from_pretrained('eenzeenee/t5-base-korean-summarization')
eenzeenee_tokenizer = AutoTokenizer.from_pretrained('eenzeenee/t5-base-korean-summarization') 
print('✅ eenzeenee 모델 다운로드 완료')

print('🎉 모든 모델 준비 완료!')
"
```

## 🧪 UV를 사용한 통합 테스트 실행

### 기본 통합 테스트

```bash
# UV 환경에서 eenzeenee 모델 통합 테스트
uv run python test_eenzeenee_integration.py

# 출력 예시:
# 🎉 모든 통합 테스트가 성공했습니다!
# eenzeenee 모델이 프로젝트에 성공적으로 통합되었습니다.
```

### UV를 사용한 유틸리티 함수 테스트

```bash
# UV 환경에서 eenzeenee_utils 기능 테스트
uv run python -c "
import sys
sys.path.append('code/utils')
from eenzeenee_utils import *

print('=== UV 환경에서 eenzeenee_utils 테스트 ===')
info = get_eenzeenee_model_info()
print(f'모델명: {info[\"model_name\"]}')
print(f'파라미터: {info[\"parameters\"]}')

text = 'UV 환경에서 테스트하는 텍스트입니다.'
processed = preprocess_for_eenzeenee(text)
print(f'전처리 결과: {processed}')
print('✅ UV 환경에서 정상 작동!')
"
```

## 🚀 UV를 사용한 실험 실행

### 1. eenzeenee 모델 실험

```bash
# 스크립트 권한 확인
chmod +x run_eenzeenee_experiment.sh

# UV 환경에서 설정 확인 모드 (안전 테스트)
uv run ./run_eenzeenee_experiment.sh

# UV 환경에서 실제 학습 실행
EENZEENEE_RUN_ACTUAL=true uv run ./run_eenzeenee_experiment.sh

# 또는 직접 trainer 실행
uv run python code/trainer.py \
    --config config.yaml \
    --config-section eenzeenee \
    --train-data data/train.csv \
    --val-data data/dev.csv \
    --test-data data/test.csv
```

### 2. mT5 모델 실험

```bash
# UV 환경에서 mT5 실험 실행
uv run python code/trainer.py \
    --config config.yaml \
    --config-section xlsum_mt5 \
    --train-data data/train.csv \
    --val-data data/dev.csv \
    --test-data data/test.csv
```

### 3. 다중 모델 비교 실험

```bash
# UV 환경에서 여러 모델 동시 비교
chmod +x run_multi_model_experiments.sh
uv run ./run_multi_model_experiments.sh
```

### 4. UV를 활용한 병렬 실험

```bash
# UV 환경에서 백그라운드 실험 실행
uv run python code/trainer.py --config config.yaml --config-section eenzeenee &
uv run python code/trainer.py --config config.yaml --config-section xlsum_mt5 &
wait  # 모든 백그라운드 작업 완료 대기
```

## 📊 실험 결과 확인

### 실험 출력 디렉토리

```bash
# eenzeenee 실험 결과
ls outputs/eenzeenee_experiment_*/
# experiment_info.json  training.log  results/

# 다른 모델 실험 결과
ls outputs/
```

### 로그 확인

```bash
# 실시간 로그 모니터링
tail -f outputs/eenzeenee_experiment_*/training.log

# 실험 메타데이터 확인
cat outputs/eenzeenee_experiment_*/experiment_info.json
```

## ⚙️ UV 환경 최적화

### pyproject.toml 활용

```toml
# pyproject.toml에서 스크립트 정의 (있다면)
[project.scripts]
train-eenzeenee = "code.trainer:main"
test-integration = "test_eenzeenee_integration:main"

# UV로 스크립트 실행
uv run train-eenzeenee --config config.yaml --config-section eenzeenee
```

### UV 의존성 관리

```bash
# 새로운 패키지 추가
uv add transformers torch

# 개발 전용 패키지 추가
uv add --dev pytest black

# 특정 버전 설치
uv add "torch>=2.0.0"

# 의존성 업데이트
uv lock --upgrade

# 의존성 정보 확인
uv tree
```

## ⚠️ 주의사항

### UV 환경 관리

1. **가상환경 위치**: UV는 프로젝트 루트에 `.venv` 생성
2. **Lock 파일**: `uv.lock` 파일로 정확한 버전 관리
3. **캐시 관리**: UV는 글로벌 캐시 사용으로 빠른 설치

### 네트워크 및 다운로드

1. **인터넷 연결 필수**: 첫 실행시 모델 다운로드를 위해 인터넷 연결 필요
2. **다운로드 시간**: 
   - mT5 (2.2GB): 약 5-10분
   - eenzeenee (800MB): 약 2-5분
3. **디스크 공간**: 최소 5GB 여유 공간 권장

### GPU 메모리 관리

```bash
# GPU 상태 확인
nvidia-smi

# UV 환경에서 GPU 테스트
uv run python -c "
import torch
print(f'CUDA 사용 가능: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU 개수: {torch.cuda.device_count()}')
    print(f'현재 GPU: {torch.cuda.get_device_name()}')
"

# 메모리 부족시 배치 크기 조정
# config.yaml에서 inference.batch_size 감소 (8 → 4 → 2)
```

## 🔧 UV 환경 문제 해결

### 일반적인 오류

#### 1. UV 설치 오류
```bash
❌ uv: command not found
```
**해결방법**:
```bash
# UV 설치
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# 또는 pip로 설치
pip install uv
```

#### 2. 의존성 충돌
```bash
❌ ResolutionImpossible: Could not resolve dependencies
```
**해결방법**:
```bash
# Lock 파일 재생성
rm uv.lock
uv lock

# 또는 강제 재설치
uv sync --reinstall
```

#### 3. 가상환경 오류
```bash
❌ No virtual environment found
```
**해결방법**:
```bash
# 가상환경 재생성
rm -rf .venv
uv venv
uv sync
```

#### 4. 모델 다운로드 실패
```bash
❌ Connection timeout / HTTP 403 error
```
**해결방법**:
```bash
# Hugging Face 토큰 설정 (필요시)
uv run huggingface-cli login

# 또는 환경변수 설정
export HUGGINGFACE_HUB_TOKEN=your_token
```

#### 5. GPU 메모리 부족
```bash
❌ CUDA out of memory
```
**해결방법**:
```yaml
# config.yaml 수정
inference:
  batch_size: 2  # 8에서 2로 감소
```

## 📈 UV 환경 성능 최적화

### 서버 리소스 활용

```bash
# 시스템 정보 확인
uv run python -c "
import os, psutil, torch
print(f'CPU 코어: {os.cpu_count()}')
print(f'메모리: {psutil.virtual_memory().total // (1024**3)}GB')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name()}')
    print(f'GPU 메모리: {torch.cuda.get_device_properties(0).total_memory // (1024**3)}GB')
"

# 최적 배치 크기 설정
# GPU 메모리에 따라 조정:
# - 8GB: batch_size=4-8
# - 16GB: batch_size=8-16
# - 24GB+: batch_size=16+
```

### UV 캐시 최적화

```bash
# UV 캐시 정보 확인
uv cache info

# 캐시 정리 (필요시)
uv cache clean

# 캐시 디렉토리 확인
uv cache dir
```

## 🔄 개발 워크플로우

### 로컬-서버 동기화

```bash
# 🖥️ 로컬(맥)에서 개발
uv add new-package        # 새 패키지 추가
uv sync                   # 로컬 환경 동기화
git add uv.lock pyproject.toml
git commit -m "deps: 새 패키지 추가"
git push origin main

# 🐧 서버에서 동기화
git pull origin main
uv sync                   # 서버 환경 동기화
```

### 실험 자동화

```bash
# UV를 사용한 실험 자동화 스크립트 예시
cat > run_experiments.sh << 'EOF'
#!/bin/bash
set -e

echo "🚀 UV 환경에서 자동 실험 시작"

# 환경 확인
uv --version
uv sync

# 모델별 실험 실행
echo "📊 eenzeenee 모델 실험"
uv run python code/trainer.py --config config.yaml --config-section eenzeenee

echo "📊 mT5 모델 실험" 
uv run python code/trainer.py --config config.yaml --config-section xlsum_mt5

echo "✅ 모든 실험 완료"
EOF

chmod +x run_experiments.sh
uv run ./run_experiments.sh
```

## 📚 추가 리소스

### UV 관련 문서
- **UV 공식 문서**: https://docs.astral.sh/uv/
- **pyproject.toml 가이드**: https://peps.python.org/pep-0621/

### 프로젝트 문서
- **프로젝트 문서**: `README.md`
- **eenzeenee 가이드**: `EENZEENEE_GUIDE.md`  
- **통합 보고서**: `EENZEENEE_INTEGRATION_REPORT.md`
- **mT5 보고서**: `MT5_INTEGRATION_REPORT.md`

---

## 🎯 UV 환경 요약

### ✅ UV 사용의 장점
1. **빠른 패키지 설치**: Rust 기반으로 pip보다 10-100배 빠름
2. **정확한 의존성 관리**: `uv.lock` 파일로 재현 가능한 환경
3. **간편한 명령어**: `uv run`, `uv sync` 등 직관적 명령어
4. **프로젝트 격리**: 프로젝트별 독립적인 가상환경

### 📋 핵심 명령어 요약
```bash
# 환경 설정
uv sync                    # 의존성 동기화
uv add package            # 패키지 추가
uv run python script.py   # 스크립트 실행

# 실험 실행
uv run python code/trainer.py --config config.yaml --config-section eenzeenee
uv run python test_eenzeenee_integration.py

# 환경 관리
uv cache clean            # 캐시 정리
uv tree                   # 의존성 트리 확인
```

## 🎉 최종 정리

1. **pytorch_model.bin 파일은 Git에 포함하지 않음** (.gitignore에 이미 설정)
2. **UV를 사용하여 빠르고 안정적인 의존성 관리**
3. **서버에서 첫 실행시 모델 자동 다운로드** (2-10분 소요)
4. **`uv run` 명령어로 모든 실험 실행**

**이제 UV 환경에서 AIStages 서버에서 안전하고 효율적으로 실험을 실행할 수 있습니다! 🚀**
