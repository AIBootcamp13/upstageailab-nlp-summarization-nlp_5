# 실험 환경 검증 스크립트 가이드

## 개요

이 문서는 Ubuntu 서버(aistages)에서 NLP 대화 요약 실험을 실행하기 전에 환경을 검증하고 잠재적인 문제를 사전에 감지하는 방법을 설명합니다.

## 스크립트 구성

### 1. `validate_experiment_env.py` - 상세 환경 검증
가장 포괄적인 검증 스크립트로, 시스템 환경부터 코드 무결성까지 모든 것을 점검합니다.

**주요 기능:**
- 시스템 정보 (OS, CPU, 메모리, GPU)
- Python 환경 및 버전
- 프로젝트 구조 완전성
- 필수 라이브러리 설치 및 버전
- 데이터 파일 유효성
- 설정 파일 파싱
- 리소스 가용성
- 파일 권한
- 코드 import 테스트

**사용법:**
```bash
# 기본 실행
python validate_experiment_env.py

# 검증 보고서 저장
python validate_experiment_env.py --save-report

# 특정 프로젝트 경로 지정
python validate_experiment_env.py --project-root /path/to/project
```

### 2. `prerun_test.py` - 사전 실행 테스트
실제 실행 환경을 시뮬레이션하여 주요 기능들을 빠르게 테스트합니다.

**주요 테스트:**
- 설정 파일 로딩
- 데이터 로딩 및 처리
- 모델 초기화
- 토크나이저 동작
- 학습 단계 시뮬레이션
- 실험 추적 시스템
- 디바이스 감지
- 메트릭 계산

**사용법:**
```bash
# 전체 테스트
python prerun_test.py

# 빠른 테스트 (필수 항목만)
python prerun_test.py --quick

# 테스트 결과 저장
python prerun_test.py --save-report
```

### 3. `validate_before_run.sh` - 통합 검증 스크립트
Bash 기반의 통합 검증 스크립트로, 서버 환경에 최적화되어 있습니다.

**특징:**
- 색상 코드로 결과 시각화
- 자동 권한 수정
- 로그 파일 자동 생성
- 시스템 리소스 실시간 확인

**사용법:**
```bash
# 실행 권한 부여 (최초 1회)
chmod +x validate_before_run.sh

# 검증 실행
./validate_before_run.sh
```

## 권장 실행 순서

1. **서버 접속 후 첫 실행:**
   ```bash
   # 통합 검증 스크립트로 전반적인 환경 확인
   ./validate_before_run.sh
   ```

2. **문제 발생 시 상세 진단:**
   ```bash
   # Python 기반 상세 검증
   python validate_experiment_env.py --save-report
   ```

3. **실험 직전 최종 확인:**
   ```bash
   # 빠른 기능 테스트
   python prerun_test.py --quick
   ```

## 일반적인 문제 해결

### 1. Python 패키지 누락
```bash
# requirements.txt 기반 설치
pip install -r requirements.txt

# CUDA 지원 PyTorch 설치 (GPU 사용 시)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. 메모리 부족
- 배치 크기 감소: `config.yaml`에서 `per_device_train_batch_size` 조정
- Gradient accumulation 사용: `gradient_accumulation_steps` 증가
- Mixed precision 활성화: `fp16: true` 설정

### 3. 디스크 공간 부족
```bash
# 불필요한 체크포인트 정리
find ./outputs -name "checkpoint-*" -type d -mtime +7 -exec rm -rf {} +

# 오래된 로그 정리
find ./logs -name "*.log" -mtime +30 -delete
```

### 4. GPU 메모리 오류
```python
# config.yaml 수정
training:
  per_device_train_batch_size: 8  # 줄이기
  gradient_checkpointing: true     # 활성화
  fp16: true                       # 활성화
```

### 5. 파일 권한 문제
```bash
# 스크립트 실행 권한
chmod +x *.sh

# 디렉토리 쓰기 권한
chmod -R 755 outputs logs models
```

## 검증 결과 해석

### 성공 표시
- ✓ (녹색): 테스트 통과
- ℹ (파란색): 정보성 메시지

### 문제 표시
- ⚠ (노란색): 경고 - 실행은 가능하나 주의 필요
- ✗ (빨간색): 오류 - 반드시 해결 필요

### 로그 파일
- `validation_logs/`: 검증 스크립트 로그
- `validation_report_*.json`: 상세 검증 보고서
- `prerun_test_report_*.json`: 기능 테스트 결과

## 실험 실행 체크리스트

□ 1. 가상환경 활성화 확인
```bash
source .venv/bin/activate  # 또는 conda activate myenv
```

□ 2. 최신 코드 동기화
```bash
git pull origin main
```

□ 3. 환경 검증 실행
```bash
./validate_before_run.sh
```

□ 4. 데이터 파일 확인
```bash
ls -la data/*.csv
```

□ 5. GPU 상태 확인
```bash
nvidia-smi
```

□ 6. 실험 설정 확인
```bash
ls -la config/experiments/*.yaml
```

□ 7. 디스크 공간 확인
```bash
df -h .
```

□ 8. 실험 실행
```bash
./run_auto_experiments.sh
```

## 추가 도구

### 실시간 모니터링
```bash
# GPU 사용률 모니터링
watch -n 1 nvidia-smi

# 시스템 리소스 모니터링
htop

# 디스크 사용률 모니터링
watch -n 60 df -h .
```

### 로그 확인
```bash
# 실시간 로그 확인
tail -f logs/auto_experiments.log

# 오류만 필터링
grep -i error logs/*.log
```

## 문의 및 지원

문제가 지속되면 다음 정보와 함께 팀에 문의하세요:
1. 검증 보고서 (`validation_report_*.json`)
2. 오류 로그
3. 시스템 정보 (OS, Python 버전, GPU 정보)
4. 실행한 명령어와 출력 결과

---

**Note:** 이 검증 스크립트들은 실험의 안정성을 높이고 시간을 절약하기 위해 설계되었습니다. 
실험 전 반드시 검증을 실행하여 잠재적인 문제를 사전에 해결하시기 바랍니다.
