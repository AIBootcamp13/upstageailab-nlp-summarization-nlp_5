# NLP 대화 요약 프로젝트 (lyj 브랜치)

한국어 대화를 자동으로 요약하는 딥러닝 모델 개발 프로젝트입니다.

## 🎯 프로젝트 목표

- 베이스라인 성능(ROUGE-F1 47.12%)을 55-60%로 향상
- 특수 토큰(PII, 화자 정보) 보존율 극대화
- 실용적인 추론 속도 유지

## 📊 목표 성능 (예상치)

> ⚠️ **주의**: 아래 성능 수치는 아직 실제로 달성되지 않은 **목표치**입니다. 실제 실험 후 업데이트 예정입니다.

| 모델 | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-F1 평균 | 상태 |
|------|---------|---------|---------|---------------|------|
| 베이스라인 | 0.5123 | 0.2845 | 0.4756 | 0.4712 | ✅ 확인됨 |
| 1차 개선 (목표) | 0.5456 | 0.3123 | 0.5089 | 0.5056 | 🎯 목표 |
| 2차 통합 (목표) | 0.5821 | 0.3456 | 0.5234 | 0.5504 | 🎯 목표 |
| Solar 앙상블 (목표) | 0.5989 | 0.3612 | 0.5401 | 0.5667 | 🎯 목표 |

## 🤖 지원 모델

프로젝트에서 지원하는 모델 목록:

| 모델명 | 설명 | 특징 | 사용법 |
|-----------|------|------|--------|
| **eenzeenee/xsum-t5-1.7b** | 한국어 요약 T5 모델 | - 1.7B 파라미터<br>- 한국어 최적화<br>- 자동 prefix 처리 | `./run_eenzeenee_experiment.sh` |
| digit82/kobart-summarization | KoBART 요약 모델 | - BART 아키텍처<br>- 한국어 지원 | 전용 스크립트 |
| google/mt5-* | Multilingual T5 | - 다국어 지원<br>- T5 아키텍처 | `--config-section mt5_base` |
| google/flan-t5-* | FLAN-T5 | - 인스트럭션 튤닝<br>- 영어 최적화 | `--config-section flan_t5_base` |

### eenzeenee 모델 특징

- **자동 Prefix 처리**: 'summarize: ' prefix가 모든 입력에 자동으로 추가
- **한국어 최적화**: 한국어 데이터셋으로 사전 학습
- **T5 아키텍처**: sequence-to-sequence 모델로 요약 작업에 최적화
- **기본 설정**: 배치 크기 8, 입력 길이 512, 출력 길이 200

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# Conda 환경 생성
conda create -n nlp-sum python=3.11
conda activate nlp-sum

# 의존성 설치
pip install -r requirements.txt

# KoNLPy 설정 (선택사항)
bash scripts/install_konlpy.sh
```

### 2. eenzeenee 모델 실험 실행

```bash
# 단일 eenzeenee 모델 실험
./run_eenzeenee_experiment.sh

# 실제 학습 실행 (설정 후)
EENZEENEE_RUN_ACTUAL=true ./run_eenzeenee_experiment.sh

# 다중 모델 비교 실험 (eenzeenee 포함)
./run_multi_model_experiments.sh
```

### 3. 최종 모델로 추론

```bash
# 간단한 추론 (Fine-tuned 모델만)
python final_submission/run_final_inference.py

# Solar API 앙상블 (최고 성능)
export UPSTAGE_API_KEY="your-api-key"
python final_submission/run_final_inference.py --use_ensemble
```

### 4. 제출 파일 확인

```bash
# 형식 검증
python scripts/validate_submission.py \
    --submission final_submission/submission.csv \
    --sample sample_submission.csv
```

## 📁 프로젝트 구조

```
nlp-sum-lyj/
├── code/
│   ├── core/               # 핵심 모듈
│   ├── data_augmentation/  # 데이터 증강
│   ├── ensemble/           # Solar API 앙상블
│   ├── models/             # 모델 관련 (가중치 손실 등)
│   ├── postprocessing/     # 후처리 파이프라인
│   ├── preprocessing/      # 전처리 (텍스트 정규화)
│   ├── utils/              # 유틸리티
│   └── trainer.py          # 학습 모듈
├── config/
│   └── experiments/        # 실험 설정 YAML 파일
├── data/                   # 데이터 파일
├── docs/                   # 문서
├── final_submission/       # 최종 제출 관련
├── logs/                   # 실험 로그
├── models/                 # 모델 체크포인트
├── outputs/                # 실험 결과
└── scripts/                # 유틸리티 스크립트
```

## 🔧 주요 기능

### 1. 자동화 실험 시스템
- YAML 기반 실험 설정
- 자동 하이퍼파라미터 추적
- WandB 통합 모니퇁링
- **WandB Sweep**: 베이지안 최적화 및 Hyperband 조기 종료 ✅

### 2. 데이터 증강
- 동의어 치환
- 문장 순서 변경
- 백트랜슬레이션 (한→영→한)

### 3. 특수 토큰 가중치
- PII 토큰 2.5배 가중치
- 화자 토큰 2.0배 가중치
- 동적 가중치 조정

### 4. 후처리 파이프라인
- 중복 제거
- 길이 최적화
- 특수 토큰 검증

### 5. Solar API 앙상블 (안정성 강화)
- Fine-tuned 모델 + Solar API
- **오류 처리 강화**: 지수 백오프 재시도, 타임아웃 증가, 연결 테스트
- **폴백 메커니즘**: Solar API 실패 시 Fine-tuned 모델로 자동 전환
- **비용 최적화**: 캐싱, rate limiting, 연속 실패 모니터링
- 동적 가중치 결합
- **현재 상태**: 코드 구현 완료, API 키 필요

## 🚀 하이퍼파라미터 최적화 (WandB Sweep)

### 베이지안 최적화 실행
```bash
# 50개 실험으로 최적 하이퍼파라미터 찾기
python code/sweep_runner.py \
  --base-config config/base_config.yaml \
  --sweep-config hyperparameter_sweep \
  --count 50

# 모델 비교 실험
python code/sweep_runner.py \
  --base-config config/base_config.yaml \
  --sweep-config model_comparison_sweep \
  --count 20
```
✅ **장점**: WandB Sweep 베이지안 최적화 + WandB 실험 추적 완전 통합

## 📨 실험 재현

### 전체 파이프라인 실행

```bash
# 1. 베이스라인 재현
python code/auto_experiment_runner.py \
    --config config/experiments/00_baseline_reproduction.yaml

# 2. 1차 개선 실험
./scripts/experiments/run_auto_experiments.sh phase1

# 3. 2차 통합 실험
./run_phase2_experiments.sh

# 4. Solar 앙상블 (선택사항)
./run_solar_ensemble.sh
```

### 개별 실험 실행

```bash
# 특정 실험만 실행
python code/auto_experiment_runner.py \
    --config config/experiments/10_combination_phase2/10c_all_optimizations.yaml
```

## 🛠️ 고급 설정

### GPU 메모리 최적화
```yaml
# config에서 조정
training:
  per_device_train_batch_size: 8  # 줄이기
  gradient_accumulation_steps: 8   # 늘리기
  fp16: true                       # 필수
  gradient_checkpointing: true     # 필수
```

### 커스텀 후처리
```python
from postprocessing import PostProcessingPipeline, CustomProcessor

pipeline = PostProcessingPipeline()
pipeline.add_processor(CustomProcessor())
```

### WandB 설정
```bash
# 로그인
wandb login

# 프로젝트 설정
export WANDB_PROJECT="nlp-summarization"
export WANDB_ENTITY="your-team"
```

## 📝 계획된 개선사항 상세

> 💡 **안내**: 아래는 계획된 개선 방법들이며, 예상 성능 향상치입니다. 실제 구현 후 결과가 업데이트될 예정입니다.

### 1. 데이터 증강 (예상: ROUGE +2-3%)
- SynonymReplacement: WordNet 기반 동의어 치환 ✅ 구현 완료
- SentenceReorder: 화자 순서 보존하며 재배열 ✅ 구현 완료
- BackTranslation: Google Translate API 활용 ✅ 구현 완료

### 2. 학습 최적화 (예상: ROUGE +1-2%)
- Cosine Annealing with Warm Restarts ✅ 설정 준비
- Learning Rate: 3e-5 → 5e-5 🎯 실험 예정
- Gradient Accumulation 최적화 ✅ 설정 준비

### 3. 특수 토큰 처리 (예상: ROUGE +2-3%)
- Weighted Cross Entropy Loss ✅ 구현 완료
- 동적 가중치 스케줄링 ✅ 구현 완료
- 토큰별 손실 추적 🎯 추가 개발 필요

### 4. 빔 서치 개선 (예상: ROUGE +1%)
- Diverse Beam Search (5 groups) ✅ 설정 준비
- Length Penalty 조정 (1.0 → 1.2) ✅ 설정 준비
- No Repeat N-gram 강화 ✅ 설정 준비

### 5. Solar API 앙상블 (예상: ROUGE +2-3%)
- 가중 평균 결합 ✅ 코드 구현
- 신뢰도 기반 동적 가중치 ✅ 코드 구현
- Few-shot 프롬프트 최적화 🔑 API 키 필요

## 🐛 트러블슈팅

### CUDA Out of Memory
```bash
# 배치 크기 감소
python code/trainer.py --per_device_train_batch_size 4

# Mixed Precision 활성화
python code/trainer.py --fp16 true --fp16_backend amp
```

### 느린 학습 속도
```bash
# 데이터로더 워커 증가
python code/trainer.py --dataloader_num_workers 8

# 캐시 활성화
export TRANSFORMERS_CACHE=/path/to/cache
```

### API Rate Limit
```python
# config에서 조정
solar_api:
  rate_limit_per_minute: 50  # 줄이기
  retry_delay: 10            # 늘리기
```

## 📚 참고 자료

- [프로젝트 문서](docs/)
- [실험 결과 분석](docs/experiment_results/)
- [API 문서](docs/api/)
- [트러블슈팅 가이드](docs/troubleshooting.md)

## 🤝 기여 방법

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 👥 팀

- 개발자: LYJ
- 프로젝트 기간: 2025.01

## 🙏 감사의 말

- Upstage AI Lab for providing the dataset and baseline
- Hugging Face for the excellent transformers library
- The open-source community for various tools and libraries

---

**Note**: Solar API 키가 필요한 기능은 별도 설정이 필요합니다. 자세한 내용은 [Solar API 가이드](code/ensemble/README.md)를 참조하세요.
