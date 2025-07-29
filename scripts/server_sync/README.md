# 🚀 AIStages 서버 동기화 스크립트

AIStages 서버에서 실험 결과를 로컬 맥으로 안전하게 동기화하고 서버 저장소를 정리하는 자동화 스크립트입니다.

## 📋 목차
1. [필수 도구 설치](#-필수-도구-설치)
2. [초기 설정](#-초기-설정)
3. [SSH 설정](#-ssh-설정)
4. [스크립트 설정](#-스크립트-설정)
5. [사용 방법](#-사용-방법)
6. [주요 기능](#-주요-기능)
7. [문제 해결](#-문제-해결)

---

## 📦 필수 도구 설치

이 스크립트는 **rsync**를 사용하여 파일 동기화를 수행합니다. 로컬(맥)과 원격(AIStages 서버) 양쪽에 모두 설치되어야 합니다.

### 🍎 로컬 맥에서 rsync 설치

#### 방법 1: Homebrew 사용 (권장)
```bash
# Homebrew가 설치되어 있지 않다면 먼저 설치
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# rsync 설치
brew install rsync

# 설치 확인
rsync --version
```

#### 방법 2: MacPorts 사용
```bash
# MacPorts가 설치되어 있다면
sudo port install rsync +universal
```

#### 방법 3: 기본 rsync 확인
```bash
# 맥에는 기본적으로 rsync가 설치되어 있을 수 있음
which rsync
rsync --version

# 출력 예시:
# rsync  version 3.2.7  protocol version 31
```

### 🐧 AIStages 서버에서 rsync 설치

서버에 SSH로 접속하여 rsync를 설치해야 합니다.

#### Ubuntu/Debian 계열 (일반적)
```bash
# 서버 접속
ssh aistages

# 패키지 업데이트
sudo apt update

# rsync 설치
sudo apt install -y rsync

# 설치 확인
rsync --version
```

#### CentOS/RHEL 계열
```bash
# 서버 접속  
ssh aistages

# rsync 설치
sudo yum install -y rsync
# 또는 (최신 버전)
sudo dnf install -y rsync

# 설치 확인
rsync --version
```

#### Alpine Linux
```bash
# 서버 접속
ssh aistages

# rsync 설치
sudo apk add rsync

# 설치 확인
rsync --version
```

### ✅ 설치 확인 방법

**로컬에서 확인:**
```bash
rsync --version
# 출력되면 설치 완료
```

**서버에서 확인:**
```bash
ssh aistages "rsync --version"
# 출력되면 서버에도 설치 완료
```

### 🚨 rsync가 없을 때 나타나는 오류

```bash
# 로컬에 rsync가 없는 경우
bash: rsync: command not found

# 서버에 rsync가 없는 경우  
ssh: rsync: command not found
rsync: connection unexpectedly closed (0 bytes received so far) [sender]
rsync error: error in rsync protocol data stream (code 12)
```

이런 오류가 발생하면 위의 설치 가이드를 따라 rsync를 설치하세요.

---

## 🔧 초기 설정

### 1단계: SSH 설정 (모든 팀원 공통)

AIStages 서버에 접속하기 위해 SSH 설정을 구성해야 합니다.

#### 1.1 SSH 키 파일 준비
팀장으로부터 받은 SSH 키 파일(`MyKey.pem`)을 다음 위치에 저장:
```bash
~/.ssh/MyKey.pem
```

#### 1.2 SSH 키 권한 설정
```bash
chmod 600 ~/.ssh/MyKey.pem
```

#### 1.3 SSH Config 파일 설정
`~/.ssh/config` 파일을 편집하여 다음 내용 추가:
```bash
# AIStages 서버 설정
Host aistages
    HostName 10.196.197.34
    Port 32145
    User root
    IdentityFile ~/.ssh/MyKey.pem
    StrictHostKeyChecking no
```

#### 1.4 SSH 연결 테스트
다음 명령으로 서버 접속이 되는지 확인:
```bash
ssh aistages
```
성공하면 `root@aistages:~#` 프롬프트가 나타납니다.

---

## ⚙️ 스크립트 설정

### 2단계: 설정 파일 생성

#### 2.1 템플릿 복사
```bash
cd scripts/server_sync/
cp config.conf.template config.conf
```

#### 2.2 개인 설정 수정
`config.conf` 파일을 열어 다음 항목들을 **본인 환경에 맞게 수정**:

```bash
# =================================================================
# 경로 설정 (각자 환경에 맞게 수정 필요)
# =================================================================

# 로컬 맥 프로젝트 경로 (각자 다름 - 반드시 수정)
LOCAL_BASE="/Users/YOUR_USERNAME/Developer/Projects/upstageailab-nlp-summarization-nlp_5/nlp-sum-lyj"

# AIStages 서버 프로젝트 경로 (각자 다름 - 반드시 수정)  
REMOTE_BASE="/data/ephemeral/home/upstageailab-nlp-summarization-nlp_5"
```

#### 2.3 팀원별 설정 예시

**팀장 (jayden) 설정:**
```bash
LOCAL_BASE="/Users/jayden/Developer/Projects/upstageailab-nlp-summarization-nlp_5/nlp-sum-lyj"
REMOTE_BASE="/data/ephemeral/home/upstageailab-nlp-summarization-nlp_5"
```

**팀원 A 설정 예시:**
```bash
LOCAL_BASE="/Users/teamA/Documents/nlp-project/nlp-sum-lyj"  
REMOTE_BASE="/data/ephemeral/home/upstageailab-nlp-summarization-nlp_5"
```

**팀원 B 설정 예시:**
```bash
LOCAL_BASE="/Users/teamB/workspace/nlp-summarization/nlp-sum-lyj"
REMOTE_BASE="/data/ephemeral/home/upstageailab-nlp-summarization-nlp_5"
```

### 📁 중요한 경로 확인 방법

#### 로컬 경로 확인:
```bash
pwd  # 현재 프로젝트 폴더에서 실행
```

#### 서버 경로 확인:
```bash
ssh aistages
cd /data/ephemeral/home/
ls -la  # 프로젝트 폴더 찾기
```

---

## 🚀 사용 방법

### 기본 동기화 실행
```bash
./scripts/server_sync/sync_aistages_results.sh
```

### 미리보기 모드 (실제 작업 없이 확인만)
```bash
./scripts/server_sync/sync_aistages_results.sh --dry-run
```

### 도움말 보기
```bash
./scripts/server_sync/sync_aistages_results.sh --help
```

---

## ✨ 주요 기능

### 🔄 지능적 동기화
- **증분 동기화**: 새로운 파일만 복사 (시간 절약)
- **중복 방지**: 이미 존재하는 파일은 건너뛰기
- **진행률 표시**: rsync 진행률 실시간 표시

### 📊 동기화 대상
```
서버 → 로컬 동기화 구조:
├── outputs/                    # 메인 실험 결과
│   ├── dialogue_summarization_*/ 
│   ├── auto_experiments/
│   └── *_20250729_*/
├── logs/                       # 실행 로그
├── wandb/                      # WandB 실험 로그
└── models/                     # 저장된 모델
```

### 🔍 무결성 검증
- **파일 수 비교**: 로컬 vs 서버 파일 개수 확인
- **크기 검증**: 주요 파일 크기 비교
- **자동 재시도**: 실패 시 자동으로 재시도

### 🧹 안전한 서버 정리
- **검증 후 삭제**: 동기화 완료 확인 후에만 서버에서 삭제
- **사용자 확인**: 삭제 전 사용자 승인 요청
- **로그 보관**: 모든 작업 내역 기록

### 📋 상세한 보고서
동기화 완료 후 다음 정보를 포함한 보고서 생성:
- 동기화된 실험 목록
- 파일 개수 및 크기 정보
- 주요 파일 위치 안내
- 타임스탬프 기록

---

## 🔧 고급 설정

### 동기화 옵션 커스터마이징
`config.conf`에서 다음 설정들을 조정할 수 있습니다:

```bash
# 동기화 후 자동 삭제 여부
AUTO_DELETE_AFTER_SYNC=false

# 백업 보관 기간 (일)
BACKUP_RETENTION_DAYS=30

# 포함할 파일 확장자
INCLUDE_EXTENSIONS="*.json,*.txt,*.log,*.csv,*.png,*.jpg,*.pt,*.pth,*.bin"

# 제외할 패턴
EXCLUDE_PATTERNS="__pycache__,*.pyc,*.tmp,*.cache"

# rsync 추가 옵션
RSYNC_ADDITIONAL_OPTIONS="--compress --partial --progress"
```

---

## 🛠️ 문제 해결

### 자주 발생하는 문제들

#### 1. rsync 명령어 없음 오류
```bash
# 에러: bash: rsync: command not found
# 또는: rsync: connection unexpectedly closed
```
**해결 방법:**
- 위의 [필수 도구 설치](#-필수-도구-설치) 섹션을 참고하여 rsync 설치
- 로컴과 서버 양쪽 모두에 설치 필요

#### 2. 한글 디렉토리 이름 오류
```bash
# 에러: rsync: change_dir "/root//디렉토리" failed: No such file or directory
# 또는: LC_ALL: cannot change locale (ko_KR.UTF-8)
```
**해결 방법:**
- 스크립트에서 자동으로 처리됨 (LC_ALL=C 설정)
- 로그 메시지가 디렉토리명으로 잘못 인식되는 문제 해결
- 디렉토리 이름 유효성 검증 기능 추가됨
#### 3. SSH 연결 실패
```bash
# 에러: ssh: connect to host aistages port 32145: Connection refused
```
**해결 방법:**
- SSH 키 파일 권한 확인: `ls -la ~/.ssh/MyKey.pem` (권한이 600이어야 함)
- SSH config 파일 확인: `cat ~/.ssh/config`
- 서버 상태 확인: 팀장에게 문의

#### 4. 설정 파일 오류
```bash  
# 에러: 필수 설정이 누락되었습니다
```
**해결 방법:**
- `config.conf` 파일이 존재하는지 확인
- `LOCAL_BASE`, `REMOTE_BASE`, `REMOTE_HOST` 설정 확인
- 경로에 공백이나 특수문자가 없는지 확인

#### 5. 권한 오류
```bash
# 에러: permission denied
```
**해결 방법:**
- 스크립트 실행 권한 부여: `chmod +x sync_aistages_results.sh`
- 로컬 디렉토리 쓰기 권한 확인

#### 6. 동기화 실패
```bash
# 에러: rsync error
```
**해결 방법:**
- 네트워크 연결 확인
- 원격 서버의 디스크 공간 확인
- `--dry-run` 옵션으로 미리 테스트

### 로그 확인 방법
모든 작업 로그는 다음 위치에 저장됩니다:
```bash
ls -la sync_report_*.txt  # 동기화 보고서
tail -f /var/log/sync.log  # 실시간 로그 (있는 경우)
```

---

## 📞 지원 및 문의

- **스크립트 관련 문제**: 팀장 (jayden)에게 문의
- **서버 접속 문제**: SSH 키 또는 서버 상태 관련 문의
- **rsync 설치 문제**: 운영체제별 설치 가이드 참고
- **설정 도움**: 다른 팀원들과 설정 공유

---

## 📝 변경 이력

- **2025-07-29**: 초기 버전 생성
  - SSH 자동 설정 지원
  - 개인별 경로 설정 지원
  - 무결성 검증 기능 추가
  - 한글 인터페이스 적용
  - rsync 설치 가이드 추가

---

## 🚨 주의사항

1. **rsync 필수**: 로컬과 서버 양쪽에 모두 rsync가 설치되어야 합니다
2. **설정 파일 보안**: `config.conf`에는 개인 경로 정보가 포함되므로 Git에 커밋하지 마세요
3. **서버 정리**: 동기화 검증 후에만 서버에서 파일을 삭제하세요
4. **디스크 공간**: 로컬 디스크 공간이 충분한지 미리 확인하세요
5. **백업**: 중요한 실험 결과는 추가로 백업해두세요

---

*이 스크립트는 팀의 실험 결과를 안전하고 효율적으로 관리하기 위해 만들어졌습니다. 문제가 있거나 개선 사항이 있으면 언제든지 알려주세요!* 🎯