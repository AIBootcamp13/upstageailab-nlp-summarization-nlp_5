# UV 기반 환경 리셋 스크립트

이 스크립트는 UV를 사용하여 Python 환경을 완전히 리셋하고 재설치하는 빠른 방법을 제공합니다.

## 사용법

### 1. 전체 환경 리셋
```bash
# Base 환경 활성화
conda activate base

# 모든 패키지 제거
uv pip freeze | xargs -n 1 uv pip uninstall -y

# 필수 패키지만 재설치
uv pip install -U pip setuptools wheel --system
```

### 2. 프로젝트 의존성 재설치
```bash
# requirements.txt로부터 설치
uv pip install -r requirements.txt --system
```

### 3. 선택적 패키지 제거
특정 패키지만 제거하고 싶을 때:
```bash
# 특정 패키지 제거
uv pip uninstall torch torchvision

# 패키지와 의존성 함께 제거
uv pip uninstall --all-dependencies torch
```

### 4. 캐시 정리
```bash
# UV 캐시 정리
uv cache clean

# 특정 패키지 캐시만 정리
uv cache clean torch
```

## 장점

1. **속도**: pip보다 10-100배 빠른 제거/설치
2. **안정성**: 의존성 충돌 해결
3. **효율성**: 병렬 처리로 시간 단축

## 주의사항

- `--system` 옵션은 conda base 환경에 직접 설치
- AIStages는 Docker 환경이므로 venv 대신 --system 사용 권장
- 중요한 작업 전 패키지 목록 백업 권장:
  ```bash
  uv pip freeze > packages_backup.txt
  ```
