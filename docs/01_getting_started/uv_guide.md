# Python 패키지 관리자 uv 가이드

## 목차
1. [uv란 무엇인가?](#1-uv란-무엇인가)
2. [왜 uv를 사용해야 하는가?](#2-왜-uv를-사용해야-하는가)
3. [설치 방법](#3-설치-방법)
4. [기본 사용법](#4-기본-사용법)
5. [고급 기능](#5-고급-기능)
6. [venv vs uv 상세 비교](#6-venv-vs-uv-상세-비교)
7. [프로젝트 적용 예시](#7-프로젝트-적용-예시)
8. [성능 벤치마크](#8-성능-벤치마크)
9. [트러블슈팅](#9-트러블슈팅)

---

## 1. uv란 무엇인가?

**uv**는 Astral(Ruff를 만든 회사)에서 2024년에 개발한 **초고속 Python 패키지 관리자**입니다. Rust로 작성되어 있으며, pip와 pip-tools를 대체하는 것을 목표로 합니다.

### 핵심 특징
- ⚡ **초고속**: pip보다 10-100배 빠른 속도
- 🔧 **올인원**: 패키지 관리 + 가상환경 + Python 버전 관리
- 🔒 **안정성**: 의존성 해결 및 lock 파일 지원
- 💾 **효율적 캐싱**: 중복 다운로드 최소화
- 🔄 **호환성**: pip 명령어와 대부분 호환

## 2. 왜 uv를 사용해야 하는가?

### 2.1 속도 비교

```bash
# PyTorch 설치 시간 비교
pip install torch       # 약 45초
uv pip install torch    # 약 3초 (15배 빠름!)

# 가상환경 생성 시간
python -m venv myenv    # 약 3-5초
uv venv myenv          # 약 0.1초 (30-50배 빠름!)
```

### 2.2 장점

1. **개발 생산성 향상**
   - 환경 설정 시간 대폭 단축
   - 빠른 실험과 프로토타이핑

2. **CI/CD 최적화**
   - 빌드 시간 단축으로 비용 절감
   - 더 빈번한 배포 가능

3. **일관된 환경**
   - Lock 파일로 정확한 버전 관리
   - 팀원 간 환경 불일치 문제 해결

4. **리소스 효율성**
   - 스마트 캐싱으로 디스크 공간 절약
   - 네트워크 사용량 감소

## 3. 설치 방법

### 3.1 Windows

#### PowerShell (권장)
```powershell
# 관리자 권한으로 실행
irm https://astral.sh/uv/install.ps1 | iex

# 설치 확인
uv --version
```

#### pip 사용
```bash
pip install uv
```

### 3.2 macOS/Linux

#### 스크립트 설치 (권장)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

# 설치 확인
uv --version
```

#### Homebrew (macOS)
```bash
brew install uv
```

### 3.3 설치 후 설정

```bash
# PATH에 추가 (필요한 경우)
export PATH="$HOME/.cargo/bin:$PATH"

# 자동 완성 설정 (선택사항)
uv generate-shell-completion bash >> ~/.bashrc
uv generate-shell-completion zsh >> ~/.zshrc
```

## 4. 기본 사용법

### 4.1 가상환경 관리

```bash
# 가상환경 생성
uv venv myproject

# Python 버전 지정하여 생성
uv venv --python 3.11 myproject

# 가상환경 활성화
# Windows
myproject\Scripts\activate
# macOS/Linux
source myproject/bin/activate

# 가상환경 비활성화
deactivate
```

### 4.2 패키지 설치

```bash
# 단일 패키지 설치
uv pip install pandas

# 여러 패키지 설치
uv pip install pandas numpy matplotlib

# 특정 버전 설치
uv pip install django==4.2.0

# requirements.txt로 설치
uv pip install -r requirements.txt

# 개발 의존성 설치
uv pip install -r requirements-dev.txt
```

### 4.3 패키지 제거

```bash
# 패키지 제거
uv pip uninstall pandas

# 모든 패키지 제거
uv pip uninstall -r requirements.txt
```

### 4.4 패키지 목록 확인

```bash
# 설치된 패키지 목록
uv pip list

# 패키지 정보 확인
uv pip show pandas

# 오래된 패키지 확인
uv pip list --outdated
```

## 5. 고급 기능

### 5.1 의존성 잠금 (Lock Files)

```bash
# requirements.txt를 기반으로 lock 파일 생성
uv pip compile requirements.txt -o requirements.lock

# Python 버전 지정
uv pip compile --python-version 3.11 requirements.txt -o requirements.lock

# lock 파일로 정확한 버전 설치
uv pip sync requirements.lock

# 개발 환경과 프로덕션 환경 분리
uv pip compile requirements.txt -o requirements.lock
uv pip compile requirements-dev.txt -o requirements-dev.lock
```

### 5.2 Python 버전 관리

```bash
# 사용 가능한 Python 버전 확인
uv python list

# Python 설치
uv python install 3.11
uv python install 3.12

# 특정 버전으로 가상환경 생성
uv venv --python 3.11.7 myproject

# 프로젝트별 Python 버전 설정
echo "3.11" > .python-version
```

### 5.3 프로젝트 관리

```bash
# 프로젝트 초기화
uv init myproject
cd myproject

# pyproject.toml 기반 의존성 추가
uv add pandas numpy

# 개발 의존성 추가
uv add --dev pytest black flake8

# 프로젝트 동기화
uv sync

# 프로덕션 의존성만 설치
uv sync --no-dev
```

### 5.4 캐시 관리

```bash
# 캐시 위치 확인
uv cache dir

# 캐시 크기 확인
uv cache info

# 캐시 정리
uv cache clean

# 특정 패키지 캐시 제거
uv cache clean pandas
```

## 6. venv vs uv 상세 비교

### 6.1 기능 비교표

| 기능 | venv | pip | uv |
|------|------|-----|-----|
| 가상환경 생성 | ✅ | ❌ | ✅ |
| 패키지 설치 | ❌ | ✅ | ✅ |
| Python 버전 관리 | ❌ | ❌ | ✅ |
| Lock 파일 | ❌ | ❌ | ✅ |
| 병렬 다운로드 | ❌ | ❌ | ✅ |
| 스마트 캐싱 | ❌ | 제한적 | ✅ |
| 의존성 해결 속도 | - | 느림 | 빠름 |
| 크로스 플랫폼 | ✅ | ✅ | ✅ |

### 6.2 명령어 비교

| 작업 | venv + pip | uv |
|------|------------|-----|
| 가상환경 생성 | `python -m venv env` | `uv venv env` |
| 패키지 설치 | `pip install package` | `uv pip install package` |
| requirements 설치 | `pip install -r requirements.txt` | `uv pip install -r requirements.txt` |
| 패키지 업그레이드 | `pip install --upgrade package` | `uv pip install --upgrade package` |
| 의존성 동기화 | 지원 안함 | `uv pip sync requirements.lock` |

## 7. 프로젝트 적용 예시

### 7.1 기존 프로젝트 마이그레이션

```bash
# 1. uv 설치
pip install uv

# 2. 기존 requirements.txt 백업
cp requirements.txt requirements.backup.txt

# 3. 새 가상환경 생성 (매우 빠름!)
uv venv venv_new

# 4. 활성화
# Windows
.\venv_new\Scripts\activate
# macOS/Linux
source venv_new/bin/activate

# 5. 의존성 설치 (10-100배 빠름!)
uv pip install -r requirements.txt

# 6. Lock 파일 생성
uv pip compile requirements.txt -o requirements.lock

# 7. 팀원들을 위한 설치 스크립트
echo "uv pip sync requirements.lock" > install.sh
```

### 7.2 새 프로젝트 시작

```bash
# 1. 프로젝트 초기화
uv init dialogue-summarization
cd dialogue-summarization

# 2. Python 버전 설정
uv python install 3.11
echo "3.11" > .python-version

# 3. 기본 의존성 추가
uv add pandas numpy torch transformers

# 4. 개발 도구 추가
uv add --dev jupyter black flake8 pytest

# 5. 환경 동기화
uv sync
```

### 7.3 Dialogue Summarization 프로젝트 예시

```bash
# 프로젝트 클론
git clone https://github.com/AIBootcamp13/upstageailab-nlp-summarization-nlp_5.git
cd upstageailab-nlp-summarization-nlp_5/nlp-sum-lyj

# uv로 환경 설정 (기존 방법보다 20배 빠름!)
uv venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\activate

# 의존성 설치 (매우 빠름!)
uv pip install -r code/requirements.txt

# Lock 파일 생성 (팀 협업용)
uv pip compile code/requirements.txt -o code/requirements.lock
```

## 8. 성능 벤치마크

### 8.1 실제 측정 결과

```bash
# 테스트 환경: Intel i7, 16GB RAM, SSD, 100Mbps 인터넷

# 가상환경 생성
time python -m venv test_venv      # 3.2초
time uv venv test_uv               # 0.08초 (40배 빠름)

# PyTorch 설치 (첫 번째 설치)
time pip install torch             # 52초
time uv pip install torch          # 4.3초 (12배 빠름)

# PyTorch 설치 (캐시된 상태)
time pip install torch             # 8초
time uv pip install torch          # 0.4초 (20배 빠름)

# 전체 requirements.txt 설치 (15개 패키지)
time pip install -r requirements.txt    # 89초
time uv pip install -r requirements.txt # 7.2초 (12배 빠름)
```

### 8.2 프로젝트 규모별 성능

| 프로젝트 규모 | 패키지 수 | pip 시간 | uv 시간 | 속도 향상 |
|--------------|----------|---------|---------|----------|
| 소규모 | 5-10 | 15초 | 2초 | 7.5배 |
| 중규모 | 20-50 | 60초 | 5초 | 12배 |
| 대규모 | 100+ | 300초 | 15초 | 20배 |

## 9. 트러블슈팅

### 9.1 일반적인 문제

#### Windows에서 실행 정책 오류
```powershell
# 오류: "스크립트를 실행할 수 없습니다"
# 해결:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### PATH 문제
```bash
# uv 명령을 찾을 수 없음
# 해결:
export PATH="$HOME/.cargo/bin:$PATH"
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
```

#### 권한 문제
```bash
# 권한 거부 오류
# 해결:
sudo chown -R $(whoami) ~/.cache/uv
```

### 9.2 호환성 문제

#### 특정 패키지 설치 실패
```bash
# 일부 패키지가 uv로 설치되지 않는 경우
# 임시 해결책:
pip install problematic-package
uv pip install -r requirements.txt
```

#### Lock 파일 충돌
```bash
# 팀원 간 lock 파일 충돌
# 해결:
uv pip compile --upgrade requirements.txt -o requirements.lock
git add requirements.lock
git commit -m "Update lock file"
```

### 9.3 성능 최적화

#### 병렬 다운로드 설정
```bash
# 동시 다운로드 수 증가
export UV_CONCURRENT_DOWNLOADS=10
```

#### 캐시 위치 변경
```bash
# SSD로 캐시 이동
export UV_CACHE_DIR=/path/to/ssd/.cache/uv
```

## 마무리

uv는 Python 개발 환경을 혁신적으로 개선하는 도구입니다. 특히 대규모 프로젝트나 CI/CD 환경에서 그 진가를 발휘합니다. 

### 추천 사용 시나리오
- ✅ 많은 의존성을 가진 프로젝트
- ✅ CI/CD 파이프라인
- ✅ 자주 환경을 재생성하는 경우
- ✅ 팀 협업 프로젝트
- ✅ 빠른 프로토타이핑

### 주의 사항
- ⚠️ 아직 발전 중인 도구 (2024년 출시)
- ⚠️ 일부 엣지 케이스에서 pip와 다를 수 있음
- ⚠️ 팀원 교육 필요

그러나 이러한 단점을 감안하더라도, uv의 속도와 기능은 충분히 매력적입니다. 특히 "대화 요약" 프로젝트처럼 많은 딥러닝 패키지를 사용하는 경우, 환경 설정 시간을 90% 이상 단축할 수 있습니다!
