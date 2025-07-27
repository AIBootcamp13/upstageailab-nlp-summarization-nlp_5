# 검증 스크립트 모음

이 디렉토리는 실험 환경 및 결과물을 검증하는 스크립트들을 포함합니다.

## 스크립트 목록

### 1. validate_experiment_env.py
- **목적**: 실험 환경 전체를 상세하게 검증
- **검증 항목**: 시스템, Python, 라이브러리, 데이터, 설정 파일 등
- **사용 시기**: 새로운 환경 설정 후, 주요 변경 사항 적용 후

### 2. prerun_test.py
- **목적**: 실제 실행 전 주요 기능 테스트
- **검증 항목**: 데이터 로딩, 모델 초기화, 학습 시뮬레이션 등
- **사용 시기**: 실험 직전 최종 확인

### 3. validate_before_run.sh
- **목적**: Ubuntu 서버 환경에서의 통합 검증
- **검증 항목**: 시스템 리소스, 권한, 전반적인 환경
- **사용 시기**: 서버 접속 후 첫 실행

### 4. validate_submission.py
- **목적**: 대회 제출 파일 검증
- **검증 항목**: 파일 형식, 필수 컬럼, 데이터 유효성
- **사용 시기**: 최종 제출 전

## 권장 실행 순서

1. **초기 환경 설정 후**:
   ```bash
   python scripts/validation/validate_experiment_env.py --save-report
   ```

2. **서버 실행 전**:
   ```bash
   bash scripts/validation/validate_before_run.sh
   ```

3. **실험 직전**:
   ```bash
   python scripts/validation/prerun_test.py --quick
   ```

4. **제출 전**:
   ```bash
   python scripts/validation/validate_submission.py
   ```

## 상세 문서

자세한 사용법과 문제 해결 방법은 다음 문서를 참조하세요:
- [검증 스크립트 상세 가이드](../../docs/06_troubleshooting/validation/VALIDATION_README.md)
