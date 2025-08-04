# 🚀 AIStages 서버 동기화 및 GPU 최적화 도구

AIStages 서버와의 실험 결과 동기화 및 GPU 메모리 최적화를 위한 통합 도구 모음입니다.

## 📋 목차

1. [주요 스크립트 개요](#-주요-스크립트-개요)
2. [서버 동기화 도구](#-서버-동기화-도구)
## 📋 목차

1. [주요 스크립트 개요](#-주요-스크립트-개요)
2. [서버 동기화 도구](#-서버-동기화-도구)
3. [GPU 최적화 도구](#-gpu-최적화-도구)
4. [설정 방법 (실행 권한 포함)](#-설정-방법)
5. [사용법](#-사용법)
6. [실험 워크플로우](#-실험-워크플로우)
7. [문제 해결](#-문제-해결)

---

## 🎯 주요 스크립트 개요

### **서버 동기화 도구 (2개)**
- **`sync_aistages_results.sh`** - 서버 → 로컬 실험 결과 동기화
- **`cleanup_all_experiments.sh`** - 안전한 실험 결과 정리 (3단계 확인)
- **`quick_cleanup.sh`** - 빠른 실험 결과 정리 (즉시 실행)

### **GPU 최적화 도구 (2개)**  
- **`gpu_memory_optimizer.sh`** - 종합적인 GPU 메모리 최적화
- **`quick_gpu_cleanup.sh`** - 빠른 GPU 메모리 정리

---

## 📡 서버 동기화 도구

### **sync_aistages_results.sh** - 메인 동기화 도구

**용도**: AIStages 서버에서 로컬로 실험 결과를 안전하게 동기화

```bash
# 기본 사용법
./scripts/server_sync/sync_aistages_results.sh

# 특정 실험만 동기화  
./scripts/server_sync/sync_aistages_results.sh --experiment my_experiment

# 건조 실행 (미리보기)
./scripts/server_sync/sync_aistages_results.sh --dry-run
```

**주요 기능:**
- ✅ 8개 실험 관련 폴더 동기화 (data 제외)
- ✅ 실험 결과 자동 백업
- ✅ 중복 파일 스마트 처리
- ✅ 상세한 동기화 리포트 생성
- ✅ 안전한 rsync 기반 전송

**동기화 대상:**
```
outputs/              # 실험 결과
logs/                 # 로그 파일  
prediction/           # 채점용 CSV (가장 중요!)
checkpoints/          # 모델 체크포인트
models/               # 저장된 모델
wandb/                # WandB 로그
validation_logs/      # 검증 로그
analysis_results/     # 분석 결과
final_submission/     # 최종 제출 파일
```

---

### **cleanup_all_experiments.sh** - 안전한 정리 도구

**용도**: 로컬 및 원격 서버의 실험 결과를 안전하게 삭제

```bash
# 전체 정리 (3단계 확인)
./scripts/server_sync/cleanup_all_experiments.sh

# 로컬만 정리
./scripts/server_sync/cleanup_all_experiments.sh --local-only

# 원격 서버만 정리  
./scripts/server_sync/cleanup_all_experiments.sh --remote-only

# 도움말
./scripts/server_sync/cleanup_all_experiments.sh --help
```

**안전 장치:**
- 🛡️ **3단계 확인**: "yes" → "DELETE" → "DELETE" 입력 필요
- 🛡️ **상세 분석**: 삭제 전 파일 수, 크기 상세 분석
- 🛡️ **보호 대상**: prediction, data 폴더는 삭제 제외
- 🛡️ **에러 처리**: 각 단계별 안전 검증

---

### **quick_cleanup.sh** - 빠른 정리 도구

**용도**: 확인 절차 없이 빠르게 실험 결과 정리

```bash
# 즉시 정리 (확인 없음)
./scripts/server_sync/quick_cleanup.sh
```

**특징:**
- ⚡ **즉시 실행**: 확인 절차 없음
- ⚡ **빠른 속도**: 5-10초 내 완료
- ⚡ **개발용**: 연속 실험 사이 빠른 정리
- 🛡️ **안전성**: prediction, data 폴더 보호 유지

---

## 🎮 GPU 최적화 도구

### **gpu_memory_optimizer.sh** - 종합 GPU 최적화

**용도**: 포괄적인 GPU 메모리 분석 및 최적화

```bash
# 상태 확인만
./scripts/server_sync/gpu_memory_optimizer.sh --check-only

# 자동 최적화 (권장)
./scripts/server_sync/gpu_memory_optimizer.sh --auto

# 강력한 정리 (문제 발생 시)
./scripts/server_sync/gpu_memory_optimizer.sh --deep-clean

# 도움말
./scripts/server_sync/gpu_memory_optimizer.sh --help
```

**최적화 기능:**
- 🔍 **GPU 상태 분석**: 메모리, 온도, 활용률 상세 분석
- 🧹 **Python 프로세스 정리**: GPU 사용 프로세스 스마트 정리
- 🗑️ **캐시 정리**: PyTorch, HuggingFace 캐시 정리
- 🔄 **CUDA 재시작**: 장치 컨텍스트 재설정 (deep-clean)
- 📊 **상세 로그**: 최적화 전후 비교 리포트

**실행 모드:**
- `--check-only`: 분석만, 정리 안함
- `--auto`: 안전한 자동 정리 (기본값)
- `--deep-clean`: 모든 GPU 프로세스 종료 + 강력한 정리

---

### **quick_gpu_cleanup.sh** - 빠른 GPU 정리

**용도**: 실험 직전 빠른 GPU 메모리 정리

```bash
# 즉시 GPU 정리
./scripts/server_sync/quick_gpu_cleanup.sh
```

**특징:**
- ⚡ **즉시 실행**: 5-15초 내 완료
- 🧹 **핵심 정리**: CUDA 캐시 + 가비지 컬렉션
- 📊 **간단 리포트**: 정리 전후 메모리 상태
- 🔧 **실험 통합**: 실험 스크립트와 연계 사용

---

## ⚙️ 설정 방법

### **1. 초기 설정**

```bash
# 설정 파일 생성
cd /Users/jayden/Developer/Projects/nlp-5/nlp-sum-lyj/scripts/server_sync/
cp config.conf.template config.conf
## ⚙️ 설정 방법

### **1. 실행 권한 설정 (필수)**

스크립트 실행 전에 반드시 실행 권한을 설정해야 합니다:

```bash
# 스크립트 디렉토리로 이동
cd /Users/jayden/Developer/Projects/nlp-5/nlp-sum-lyj/scripts/server_sync/

# 모든 스크립트에 실행 권한 부여
chmod +x *.sh

# 또는 개별적으로 설정
chmod +x sync_aistages_results.sh
chmod +x cleanup_all_experiments.sh
chmod +x quick_cleanup.sh
chmod +x gpu_memory_optimizer.sh
chmod +x quick_gpu_cleanup.sh
```

**⚠️ 중요**: 실행 권한이 없으면 `Permission denied` 오류가 발생합니다!

### **2. 초기 설정**
vim config.conf
```

### **2. 현재 설정 상태 (즉시 사용 가능)**

```bash
# 경로 설정 (완료)
LOCAL_BASE="/Users/jayden/Developer/Projects/nlp-5/nlp-sum-lyj"
REMOTE_BASE="/data/ephemeral/home/nlp-5/nlp-sum-lyj"
REMOTE_HOST="aistages"

# 활성화된 동기화 대상 (8개)
OUTPUTS_PATH="outputs"              ✅
LOGS_PATH="logs"                    ✅
PREDICTION_PATH="prediction"        ✅ (가장 중요!)
CHECKPOINTS_PATH="checkpoints"      ✅
MODELS_PATH="models"                ✅
## 🚀 사용법

### **시작 전 준비**

```bash
# 0. 실행 권한 설정 (최초 1회만)
cd /Users/jayden/Developer/Projects/nlp-5/nlp-sum-lyj/scripts/server_sync/
chmod +x *.sh

# 1. GPU 상태 종합 점검
FINAL_SUBMISSION_PATH="final_submission" ✅

# 보호 대상
DATA_PATH=""                        🛡️ 비활성화 (안전)
```

---

## 🚀 사용법

### **실험 전 준비**

```bash
# 1. GPU 상태 종합 점검
./scripts/server_sync/gpu_memory_optimizer.sh --check-only

# 2. 필요시 GPU 최적화
./scripts/server_sync/gpu_memory_optimizer.sh --auto

# 3. 이전 실험 결과 정리 (선택)
./scripts/server_sync/quick_cleanup.sh
```

### **실험 실행**

```bash
# 4. 실험 시작
bash run_main_5_experiments.sh

# 또는 빠른 테스트
bash run_main_5_experiments.sh -1
```

### **실험 후 결과 관리**

```bash
# 5. 서버 결과 동기화
./scripts/server_sync/sync_aistages_results.sh

# 6. 결과 확인
ls prediction/
cat prediction/experiment_index.csv

# 7. 불필요한 결과 정리 (필요시)
./scripts/server_sync/cleanup_all_experiments.sh --local-only
```

---

## 🔄 실험 워크플로우

### **완전한 실험 사이클**

```bash
# 📊 1단계: 환경 점검
./scripts/server_sync/gpu_memory_optimizer.sh --check-only

# 🧹 2단계: 환경 정리
./scripts/server_sync/quick_gpu_cleanup.sh
./scripts/server_sync/quick_cleanup.sh

# 🚀 3단계: 실험 실행
bash run_main_5_experiments.sh

# 📡 4단계: 결과 동기화
./scripts/server_sync/sync_aistages_results.sh

# 📋 5단계: 결과 확인
ls prediction/latest_output.csv
```

### **연속 실험**

```bash
# 실험 1
./scripts/server_sync/quick_gpu_cleanup.sh && bash run_main_5_experiments.sh -1
./scripts/server_sync/sync_aistages_results.sh

# 실험 2  
./scripts/server_sync/quick_cleanup.sh
./scripts/server_sync/quick_gpu_cleanup.sh && bash run_main_5_experiments.sh -1
./scripts/server_sync/sync_aistages_results.sh
```

### **문제 발생 시 복구**

```bash
# GPU 메모리 과부하
./scripts/server_sync/gpu_memory_optimizer.sh --deep-clean

# 실험 결과 충돌
./scripts/server_sync/cleanup_all_experiments.sh --local-only
./scripts/server_sync/sync_aistages_results.sh

# 전체 초기화
./scripts/server_sync/cleanup_all_experiments.sh
./scripts/server_sync/gpu_memory_optimizer.sh --deep-clean
```

---

## 🎯 스크립트별 특징 요약

| **스크립트** | **용도** | **속도** | **안전성** | **상세도** |
|-------------|---------|---------|-----------|-----------|
| **sync_aistages_results.sh** | 서버 동기화 | 중간 | 높음 | 상세 |
| **cleanup_all_experiments.sh** | 안전한 정리 | 느림 | 매우 높음 | 매우 상세 |
| **quick_cleanup.sh** | 빠른 정리 | 빠름 | 보통 | 간단 |
| **gpu_memory_optimizer.sh** | GPU 종합 최적화 | 느림 | 높음 | 매우 상세 |
| **quick_gpu_cleanup.sh** | 빠른 GPU 정리 | 매우 빠름 | 보통 | 간단 |

---

## 📁 중요 파일 위치

### **설정 파일**
- `config.conf` - 개인 설정 (Git 제외)
- `config.conf.template` - 설정 템플릿 (Git 포함)

### **실험 결과**
- `prediction/latest_output.csv` - 최신 채점용 파일
- `prediction/experiment_index.csv` - 실험 추적 파일
### **권한 문제**

```bash
# 실행 권한 확인
ls -la scripts/server_sync/*.sh

# 실행 권한이 없는 경우 (-rw-r--r-- 표시)
# 모든 스크립트에 실행 권한 설정
cd scripts/server_sync/
chmod +x *.sh

# 권한 설정 후 확인 (-rwxr-xr-x 표시되어야 함)
ls -la *.sh

# Permission denied 오류가 계속 발생하는 경우
sudo chmod +x *.sh
```

**💡 팁**: 새로 다운로드하거나 복사한 스크립트는 항상 실행 권한을 확인하세요!

---

## 🛠️ 문제 해결

### **동기화 문제**

```bash
# SSH 연결 확인
ssh aistages "echo 'Connected'"

# 설정 파일 확인
cat scripts/server_sync/config.conf

# 건조 실행으로 미리보기
./scripts/server_sync/sync_aistages_results.sh --dry-run
```

### **GPU 메모리 문제**

```bash
# 현재 상태 확인
nvidia-smi

# 종합 분석
./scripts/server_sync/gpu_memory_optimizer.sh --check-only

# 강력한 정리
./scripts/server_sync/gpu_memory_optimizer.sh --deep-clean
```

### **권한 문제**

```bash
# 실행 권한 확인
ls -la scripts/server_sync/*.sh

# 권한 설정
chmod +x scripts/server_sync/*.sh
```

### **로그 확인**

```bash
# 최신 동기화 로그
ls -la logs/sync_report_*.txt | tail -1

# 최신 GPU 최적화 로그  
ls -la logs/gpu_optimizer_*.log | tail -1

# 실험 로그
ls -la logs/main_experiments_*/
```

---

## 🎉 완벽한 실험 환경 구성 완료!

**이제 4개의 강력한 도구로 효율적인 AI 실험이 가능합니다:**

1. **🔄 자동 동기화** - 실험 결과 안전한 백업
2. **🧹 스마트 정리** - 실험 환경 최적 유지  
3. **🎮 GPU 최적화** - 메모리 효율성 극대화
4. **⚡ 빠른 워크플로우** - 개발 생산성 향상

**모든 스크립트는 프로젝트 내 어느 위치에서든 바로 사용 가능합니다!** 🚀
