# 🤖 모델 학습

대화 요약 모델의 학습 방법과 성능 최적화 가이드입니다.

## 🎆 최신 고성능 기능 (2024.12)

### ✨ [unsloth QLoRA 고성능 파인튜닝](./unsloth_qlora_guide.md)
- 메모리 75% 절약 고효율 학습
- 4-bit 양자화 및 gradient checkpointing
- Linux/macOS 환경별 최적화 가이드
- 성능 벤치마크 및 트러블슈팅

## 📋 포함 문서

### 🏁 [베이스라인 학습](./baseline_training.md)
- KoBART 모델 기본 학습 과정 상세 분석
- 베이스라인 코드 구조 및 주요 컴포넌트
- 학습 파이프라인 이해와 실행 방법

### ⚙️ [하이퍼파라미터 튜닝](./hyperparameter_tuning.md)
- 성능 향상을 위한 핵심 파라미터 조정
- 학습률, 배치 크기, 에포크 최적화
- 체계적인 튜닝 전략 및 실험 설계

## 🎯 학습 워크플로우

1. **최신 기술 스택 업그레이드** - torch 2.6.0, transformers 4.54.0 적용
2. **unsloth QLoRA 설정** - 고성능 파인튜닝 환경 구성
3. **베이스라인 이해** - 기본 모델 구조 및 학습 과정 파악
4. **초기 학습 실행** - 최적화된 설정으로 모델 학습 및 성능 확인
5. **성능 분석** - 학습 곡선 분석 및 개선점 식별
## 📈 성능 목표

- **기존 베이스라인**: ROUGE-F1 47.12%
- **최신 기술 스택 적용**: ROUGE-F1 49-51% (최적화된 설정)
- **unsloth QLoRA 활용**: ROUGE-F1 52-55% (고효율 파인튜닝)
- **최종 목표**: ROUGE-F1 55-60%
- **튜닝 후 목표**: ROUGE-F1 50-52%
- **최종 목표**: ROUGE-F1 55-60%

## 🔗 관련 단계

- **이전**: [데이터 분석](../data_analysis/README.md)
- **다음**: [실험 관리](../experiment_management/README.md)
- **평가**: [성능 평가](../evaluation/README.md)

---
📍 **위치**: `docs/02_user_guides/model_training/`
