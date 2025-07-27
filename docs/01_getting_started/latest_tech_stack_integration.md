# 최신 기술 스택 통합 가이드

이 문서는 조장님(nlp-sum-song)의 최신 기술 스택을 nlp-sum-lyj에 완전히 통합하는 과정을 설명합니다.

## 📋 업그레이드 요약

### 핵심 라이브러리 업그레이드
- **torch**: >=2.0.0 → 2.6.0
- **transformers**: 4.35.2 → 4.54.0 (19버전 대폭 업그레이드!)  
- **pytorch_lightning**: 2.1.2 → 2.5.2
- **accelerate**: 1.9.0 (새로 추가)

### 고급 기능 추가
- **unsloth**: 고성능 파인튜닝 (Linux 환경)
- **QLoRA**: 4-bit 양자화 메모리 최적화
- **gradient checkpointing**: 메모리 사용량 최적화

### 성능 최적화 설정
- **decoder_max_len**: 100 → 200 (더 긴 요약)
- **eval_strategy**: epoch → steps (정밀한 모니터링)
- **dataloader_num_workers**: 8 (병렬 처리)

## 🚀 빠른 시작

### 자동화 스크립트 사용 (권장)

```bash
# 완전 자동화 마이그레이션
./scripts/migrate_to_latest_stack.sh

# AIStages 환경 (교육 플랫폼)
./scripts/setup_aistages.sh

# 환경 확인
./check_env.sh
```

### 수동 설정

자세한 수동 설정 방법은 개별 가이드를 참조하세요:

1. **환경 리셋**: [`docs/01_getting_started/environment_reset.md`](./01_getting_started/environment_reset.md)
2. **UV 사용법**: [`docs/01_getting_started/uv_guide.md`](./01_getting_started/uv_guide.md)  
3. **AIStages 설정**: [`docs/01_getting_started/aistages_setup.md`](./01_getting_started/aistages_setup.md)

## 📊 예상 성능 향상

### 학습 성능
- **학습 속도**: 20-30% 향상 (torch 2.6.0 최적화)
- **메모리 효율성**: 30-75% 감소 (QLoRA + unsloth)
- **요약 품질**: 더 긴 요약 생성 (200 토큰)

### 개발 효율성
- **환경 설정**: 90% 시간 단축 (UV + 자동화)
- **패키지 설치**: 10-100배 빠른 속도
- **안정성**: 최신 버전으로 호환성 향상

## 🔧 환경별 지원

### 완전 지원 환경
- **Ubuntu 20.04+**: 모든 기능 (unsloth 포함)
- **AIStages**: 교육 플랫폼 완전 최적화
- **Google Colab**: 클라우드 환경 지원

### 호환 환경  
- **macOS**: QLoRA 모드 (unsloth 제외)
- **Windows**: 기본 모드 지원

## 🗂️ 문서 구조

```
docs/01_getting_started/
├── environment_reset.md    # UV 기반 환경 리셋 (최신 버전 반영)
├── uv_guide.md            # UV 패키지 매니저 상세 가이드
├── aistages_setup.md       # AIStages 환경 설정 (최신 스택)
└── integration_guide.md   # 기존 프로젝트 통합 방법
```

## 💡 핵심 특징

### 자동화된 마이그레이션
- 8단계 체계적 프로세스
- 자동 백업 및 복구 지원
- 환경별 최적화 적용

### 안전한 롤백
- 완전한 백업 시스템
- 원클릭 복구 스크립트
- 단계별 검증 과정

### 크로스 플랫폼 호환성
- 환경 자동 감지
- 조건부 기능 설치
- 플랫폼별 최적화

## 🆘 문제 해결

### 일반적인 문제
- **unsloth 설치 실패 (macOS)**: QLoRA 모드 자동 대체
- **CUDA 버전 불일치**: 호환 버전 자동 설치
- **메모리 부족**: gradient checkpointing 자동 활성화

### 지원 리소스
- **환경 진단**: `./check_env.sh`
- **복구 스크립트**: `./scripts/restore_environment.sh`
- **상세 가이드**: 개별 문서 참조

---

*이 가이드는 조장님의 최신 기술 스택을 완전히 통합하기 위한 종합 참조 문서입니다.*
