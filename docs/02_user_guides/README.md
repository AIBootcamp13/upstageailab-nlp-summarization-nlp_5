# 📜 사용자 가이드

일반 사용자를 위한 주요 작업별 상세 가이드를 제공합니다.

## 🎆 새로운 기능 (2024.12 업데이트)

### 고성능 파인튜닝
- **unsloth QLoRA 활용법** - 메모리 75% 절약 고효율 학습
- **4-bit 양자화** - QLoRA 기반 메모리 최적화
- **gradient checkpointing** - 대용량 모델 학습 지원

### 성능 최적화
- **decoder_max_len 200** - 더 긴 요약 생성
- **eval_strategy steps** - 정밀한 실시간 모니터링
- **병렬 데이터 로딩** - dataloader_num_workers 8

## 📁 하위 카테고리

### 📊 data_analysis/
- **데이터셋 분석** - DialogSum 데이터셋 구조 및 특성 분석
- **텍스트 전처리** - 대화 데이터 정제 및 전처리 방법

### 🤖 model_training/
- **베이스라인 학습** - KoBART 모델 기본 학습 방법
- **하이퍼파라미터 튜닝** - 성능 최적화를 위한 파라미터 조정

### ⚡ inference_optimization/
- **고급 추론 엔진** - 배치 처리, 자동 최적화, 다중 입력 형식 지원
- **성능 최적화** - 디바이스별 최적화 및 메모리 관리

### 🧪 experiment_management/
- **WandB 추적** - 실험 결과 모니터링 및 관리
- **자동화 가이드** - 실험 자동화 도구 사용법

### 📈 evaluation/
- **ROUGE 메트릭** - 요약 성능 평가 지표 상세 설명
- **성능 분석** - 모델 성능 분석 및 개선 방향

## 🔄 작업 흐름

1. **데이터 분석** → 2. **모델 학습** → 3. **추론 최적화** → 4. **실험 관리** → 5. **평가**

## 🔗 관련 링크

- [시작하기](../01_getting_started/README.md)
- [기술 문서](../03_technical_docs/README.md)
- [실험 관리](../04_experiments/README.md)

---

✅ **이동 완료된 문서들:**
- `competition_guides/dialogsum_dataset_analysis.md` → [데이터셋 분석](./data_analysis/dataset_analysis.md)
- `competition_guides/text_data_analysis_guide.md` → [텍스트 전처리](./data_analysis/text_preprocessing.md)
- `baseline_code_analysis.md` → [베이스라인 학습](./model_training/baseline_training.md)
- `competition_guides/hyperparameter_tuning_guide.md` → [하이퍼파라미터 튜닝](./model_training/hyperparameter_tuning.md)
- `competition_guides/wandb_experiment_tracking_guide.md` → [WandB 추적](./experiment_management/wandb_tracking.md)
- `competition_guides/competition_integration_guide.md` → [자동화 가이드](./experiment_management/automation_guide.md)
- `rouge_metrics_detail.md` → [ROUGE 메트릭](./evaluation/rouge_metrics.md)
- 새로 추가: [성능 분석](./evaluation/performance_analysis.md)
- 새로 추가: [고급 추론 엔진](./inference_optimization/advanced_inference_guide.md)
