# NLP 프로젝트 실험 자동화 시스템 설계 문서

## 개요

본 문서는 6조 CV 프로젝트의 성공적인 실험 자동화 전략을 바탕으로, 현재 NLP 요약 프로젝트에 적합한 자동화 시스템을 설계한 결과입니다. 기존 팀원들의 워크플로우를 방해하지 않으면서 점진적으로 실험 자동화를 도입할 수 있는 구조를 제시합니다.

## 1. 현재 NLP 프로젝트 구조 분석

### 1.1 기존 구조의 강점
```
nlp-sum-lyj/code/
├── config.yaml          # 잘 구조화된 설정 파일
├── baseline.ipynb       # 완전한 학습 파이프라인
├── solar_api.ipynb     # Solar API 활용 방안
└── requirements.txt     # 의존성 관리
```

**강점 분석:**
- ✅ **계층적 설정 구조**: general, tokenizer, training, wandb, inference로 체계적 분류
- ✅ **NLP 특화 고려**: 특수 토큰, 생성 파라미터, ROUGE 평가 등 포함
- ✅ **WandB 통합 준비**: 이미 wandb 설정 섹션 존재
- ✅ **실용적 파라미터**: 대회 특성에 맞는 현실적 설정값

### 1.2 확장 필요성 분석
```yaml
# 현재 config.yaml의 한계점
❌ WandB Sweep 설정 부재
❌ 하이퍼파라미터 탐색 공간 정의 없음
❌ 모델별 특화 설정 분리 없음
❌ 자동화 실험을 위한 메타 설정 부족
❌ 베이지안 최적화 지원 없음
```

## 2. 확장 설계 전략

### 2.1 점진적 확장 원칙
1. **기존 구조 보존**: config.yaml과 baseline.ipynb는 변경 없이 유지
2. **선택적 사용**: 기존 방식과 새로운 자동화 방식 병행 지원
3. **하위 호환성**: 기존 설정 파일 완전 호환
4. **단계적 도입**: 팀원별로 원하는 시점에 새 기능 활용

### 2.2 확장된 폴더 구조
```
nlp-sum-lyj/
├── code/
│   ├── config.yaml                    # 기존 유지 (변경 없음)
│   ├── baseline.ipynb                 # 기존 유지 (변경 없음)
│   ├── solar_api.ipynb               # 기존 유지
│   ├── requirements.txt              # 기존 유지
│   │
│   ├── config/                       # 새로운 확장 설정
│   │   ├── base_config.yaml          # config.yaml 확장 버전
│   │   ├── sweep/                    # WandB Sweep 설정들
│   │   │   ├── hyperparameter_sweep.yaml
│   │   │   ├── model_comparison_sweep.yaml
│   │   │   ├── ablation_study_sweep.yaml
│   │   │   └── quick_test_sweep.yaml
│   │   └── models/                   # 모델별 특화 설정
│   │       ├── kobart.yaml
│   │       ├── kogpt2.yaml
│   │       ├── t5.yaml
│   │       └── solar_api.yaml
│   │
│   ├── utils/                        # 유틸리티 모듈들
│   │   ├── __init__.py
│   │   ├── config_manager.py         # 설정 로딩 및 관리
│   │   ├── data_utils.py            # 데이터 처리 유틸리티
│   │   ├── metrics.py               # ROUGE 평가 시스템
│   │   └── experiment_utils.py      # 실험 관리 도구
│   │
│   ├── modules/                      # 모듈화된 컴포넌트
│   │   ├── __init__.py
│   │   ├── data_module.py           # 데이터셋 및 전처리
│   │   ├── model_module.py          # 모델 로딩 및 설정
│   │   └── trainer_module.py        # 학습 로직
│   │
│   ├── trainer.py                    # baseline.ipynb 모듈화 버전
│   ├── sweep_runner.py              # WandB Sweep 실행기 (6조 방식)
│   ├── inference.py                 # 추론 전용 모듈
│   └── experiment_runner.py         # 통합 실험 실행기
│
└── docs/experiment_automation/       # 문서화
    ├── README.md
    ├── cv_team6_analysis.md         # 6조 분석 (완료)
    ├── nlp_automation_design.md     # 본 문서
    ├── configuration_guide.md
    └── best_practices.md
```

## 3. 설정 파일 확장 설계

### 3.1 base_config.yaml (기존 config.yaml 확장)
```yaml
# 기존 config.yaml의 모든 설정을 포함하되 확장
meta:
  experiment_name: "nlp_summarization"
  version: "1.0"
  description: "Dialogue Summarization with Automated Hyperparameter Tuning"
  
general:
  data_path: "../data/"
  model_name: "digit82/kobart-summarization"
  output_dir: "./"
  seed: 42
  device: "auto"  # auto, cuda, cpu

model:
  architecture: "kobart"  # kobart, kogpt2, t5, solar_api
  checkpoint: "digit82/kobart-summarization"
  load_pretrained: true

tokenizer:
  encoder_max_len: 512
  decoder_max_len: 100
  bos_token: "<s>"
  eos_token: "</s>"
  special_tokens: ['#Person1#', '#Person2#', '#Person3#', '#PhoneNumber#', '#Address#', '#PassportNumber#']
  truncation: true
  padding: "max_length"

training:
  # 기존 training 설정 모두 포함
  overwrite_output_dir: true
  num_train_epochs: 20
  learning_rate: 1.0e-05
  per_device_train_batch_size: 50
  per_device_eval_batch_size: 32
  warmup_ratio: 0.1
  weight_decay: 0.01
  lr_scheduler_type: 'cosine'
  optim: 'adamw_torch'
  gradient_accumulation_steps: 1
  evaluation_strategy: 'epoch'
  save_strategy: 'epoch'
  save_total_limit: 5
  fp16: true
  load_best_model_at_end: true
  logging_dir: "./logs"
  logging_strategy: "epoch"
  predict_with_generate: true
  generation_max_length: 100
  do_train: true
  do_eval: true
  early_stopping_patience: 3
  early_stopping_threshold: 0.001
  report_to: "wandb"
  
  # 새로운 NLP 특화 설정 추가
  label_smoothing: 0.0
  dataloader_num_workers: 4
  remove_unused_columns: false

generation:  # 추론 설정을 별도 섹션으로 분리
  max_length: 100
  min_length: 10
  num_beams: 4
  no_repeat_ngram_size: 2
  early_stopping: true
  length_penalty: 1.0
  repetition_penalty: 1.0
  do_sample: false
  temperature: 1.0
  top_k: 50
  top_p: 1.0

evaluation:
  metrics: ["rouge1", "rouge2", "rougeL"]
  multi_reference: true  # 대회 특성 (3개 정답)
  rouge_use_stemmer: true
  rouge_tokenize_korean: true

wandb:
  entity: "wandb_repo"
  project: "nlp-summarization-auto"
  name: "auto_experiment"
  notes: "Automated hyperparameter tuning experiment"
  tags: ["nlp", "summarization", "automl"]

inference:
  ckt_path: "model_ckt_path"
  result_path: "./prediction/"
  batch_size: 32
  remove_tokens: ['<usr>', '<s>', '</s>', '<pad>']
  output_format: "csv"  # csv, json
```

### 3.2 WandB Sweep 설정 (sweep/hyperparameter_sweep.yaml)
```yaml
# 6조 방식을 NLP에 적용한 베이지안 최적화 설정
project: nlp-summarization-auto
entity: wandb_repo
program: sweep_runner.py

method: bayes  # 베이지안 최적화
metric:
  name: rouge_combined_f1  # ROUGE-1 + ROUGE-2 + ROUGE-L 합계
  goal: maximize

# 조기 종료 설정
early_terminate:
  type: hyperband
  min_iter: 5
  max_iter: 20

parameters:
  # 학습률 관련
  learning_rate:
    distribution: log_uniform_values
    min: 1.0e-6
    max: 1.0e-4
  
  # 배치 크기
  per_device_train_batch_size:
    values: [8, 16, 32, 64]
  
  per_device_eval_batch_size:
    values: [16, 32, 64]
  
  # 토큰 길이 설정 (NLP 특화)
  encoder_max_len:
    values: [256, 512, 1024]
  
  decoder_max_len:
    values: [64, 100, 128, 256]
  
  # 정규화 파라미터
  weight_decay:
    distribution: log_uniform_values
    min: 1.0e-3
    max: 1.0e-1
  
  warmup_ratio:
    values: [0.0, 0.1, 0.2, 0.3]
  
  # 생성 파라미터 (NLP 특화)
  num_beams:
    values: [3, 4, 5, 8]
  
  no_repeat_ngram_size:
    values: [2, 3, 4]
  
  length_penalty:
    distribution: uniform
    min: 0.8
    max: 1.5
  
  # 학습 설정
  num_train_epochs:
    values: [10, 15, 20, 25]
  
  gradient_accumulation_steps:
    values: [1, 2, 4]
  
  # 정규화
  label_smoothing:
    values: [0.0, 0.1, 0.2]
  
  # 조기 종료
  early_stopping_patience:
    values: [3, 5, 7]

# 실험 제약 조건
constraints:
  # 메모리 제약 고려
  - batch_size_memory_constraint:
      if: encoder_max_len > 512
      then: per_device_train_batch_size <= 16
```

### 3.3 모델별 특화 설정 (models/kobart.yaml)
```yaml
# KoBART 특화 설정
model:
  architecture: "kobart"
  checkpoint: "digit82/kobart-summarization"
  config_overrides:
    max_position_embeddings: 1024
    vocab_size: 30000

tokenizer:
  model_max_length: 1024
  special_tokens: ['#Person1#', '#Person2#', '#Person3#', '#PhoneNumber#', '#Address#', '#PassportNumber#']

training:
  # KoBART에 최적화된 기본값
  learning_rate: 3.0e-05
  per_device_train_batch_size: 32
  warmup_ratio: 0.1
  weight_decay: 0.01
  
generation:
  max_length: 128
  num_beams: 5
  length_penalty: 1.2
  no_repeat_ngram_size: 3

# KoBART 특화 하이퍼파라미터 탐색 공간
sweep_parameters:
  learning_rate:
    min: 1.0e-5
    max: 5.0e-5
  length_penalty:
    min: 0.9
    max: 1.5
```

## 4. 모듈화 설계

### 4.1 baseline.ipynb 모듈화 전략

**기존 baseline.ipynb의 주요 섹션들:**
1. **데이터 전처리** → `modules/data_module.py`
2. **모델 로딩 및 설정** → `modules/model_module.py`
3. **학습 루프** → `modules/trainer_module.py`
4. **평가 및 메트릭** → `utils/metrics.py`
5. **추론** → `inference.py`

### 4.2 ConfigManager 설계 (utils/config_manager.py)
```python
class ConfigManager:
    """설정 파일 통합 관리자 - 기존/신규 설정 모두 지원"""
    
    def __init__(self):
        self.config = None
        self.is_legacy = False
    
    def load_config(self, config_path):
        """기존 config.yaml과 새로운 base_config.yaml 모두 지원"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # 기존 config.yaml 형식 감지 및 자동 변환
        if self._is_legacy_format(config):
            config = self._migrate_legacy_config(config)
            self.is_legacy = True
        
        self.config = config
        return config
    
    def merge_sweep_config(self, sweep_params):
        """WandB Sweep 파라미터를 동적으로 병합"""
        # 6조 방식과 동일한 동적 설정 업데이트
        pass
    
    def _is_legacy_format(self, config):
        """기존 config.yaml 형식인지 판단"""
        return "meta" not in config and "general" in config
    
    def _migrate_legacy_config(self, legacy_config):
        """기존 설정을 새로운 형식으로 자동 변환"""
        # 기존 팀원들의 config.yaml을 새 형식으로 투명하게 변환
        pass
```

### 4.3 NLP 특화 메트릭 시스템 (utils/metrics.py)
```python
class MultiReferenceROUGE:
    """대회 특성에 맞는 Multi-reference ROUGE 평가"""
    
    def __init__(self):
        self.rouge = Rouge()
    
    def compute_metrics(self, eval_preds):
        """HuggingFace Trainer와 호환되는 메트릭 계산"""
        predictions, labels = eval_preds
        
        # 3개 참조 요약문 중 최고 점수 계산
        rouge_scores = self._compute_multi_reference_rouge(predictions, labels)
        
        return {
            "rouge1_f1": rouge_scores["rouge1_f1"],
            "rouge2_f1": rouge_scores["rouge2_f1"],
            "rougeL_f1": rouge_scores["rougeL_f1"],
            "rouge_combined_f1": sum(rouge_scores.values())  # WandB Sweep 목표
        }
    
    def _compute_multi_reference_rouge(self, predictions, references):
        """각 예측에 대해 3개 참조 중 최고 점수 반환"""
        pass
```

## 5. 자동화 스크립트 설계

### 5.1 sweep_runner.py (6조 방식 적용)
```python
import wandb
from utils.config_manager import ConfigManager
from modules.trainer_module import NLPTrainer
from utils.metrics import MultiReferenceROUGE

def train():
    """WandB Sweep에서 호출되는 메인 훈련 함수"""
    # 1. 설정 로딩 (6조 방식)
    config_manager = ConfigManager()
    base_config = config_manager.load_config("config/base_config.yaml")
    
    # 2. WandB Sweep 파라미터 동적 병합
    wandb_config = wandb.config
    config = config_manager.merge_sweep_config(wandb_config)
    
    # 3. NLP 특화 모델 및 데이터 설정
    trainer = NLPTrainer(config)
    
    # 4. Multi-reference ROUGE 평가 설정
    metrics = MultiReferenceROUGE()
    
    # 5. 학습 실행
    results = trainer.train()
    
    # 6. 대회 특성에 맞는 최종 점수 계산
    final_score = results["rouge_combined_f1"]
    wandb.log({"final_rouge_score": final_score})
    
    return results

if __name__ == "__main__":
    # 6조 방식과 동일한 WandB Sweep 실행
    sweep_config = yaml.safe_load(open("config/sweep/hyperparameter_sweep.yaml"))
    sweep_id = wandb.sweep(sweep=sweep_config, project="nlp-summarization-auto")
    wandb.agent(sweep_id=sweep_id, function=train, count=20)
```

### 5.2 experiment_runner.py (통합 실험 관리)
```python
class ExperimentRunner:
    """다양한 실험 타입을 통합 관리하는 실행기"""
    
    def __init__(self):
        self.config_manager = ConfigManager()
    
    def run_single_experiment(self, config_path):
        """단일 실험 실행 (기존 방식과 호환)"""
        pass
    
    def run_hyperparameter_sweep(self, sweep_config_path):
        """하이퍼파라미터 탐색 실험"""
        pass
    
    def run_model_comparison(self, models_list):
        """여러 모델 성능 비교 실험"""
        pass
    
    def run_ablation_study(self, ablation_config):
        """소거 연구 실험"""
        pass
```

## 6. 호환성 보장 전략

### 6.1 기존 워크플로우 보존
```python
# 기존 방식 (변경 없음)
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
# 기존 baseline.ipynb 코드 그대로 사용 가능

# 새로운 방식 (선택적 사용)
config_manager = ConfigManager()
config = config_manager.load_config("config/base_config.yaml")
# 또는 기존 config.yaml도 자동 변환하여 사용
config = config_manager.load_config("config.yaml")  # 자동 변환됨
```

### 6.2 점진적 마이그레이션 지원
1. **Phase 1**: 기존 config.yaml + 새로운 utils 모듈 사용
2. **Phase 2**: 일부 실험에만 새로운 설정 구조 적용
3. **Phase 3**: 원하는 팀원만 WandB Sweep 활용
4. **Phase 4**: 완전 자동화 전환 (선택 사항)

## 7. NLP 특화 고려사항

### 7.1 대회 특성 반영
- **Multi-reference 평가**: 3개 정답 요약문 중 최고 점수
- **한국어 토큰화**: 형태소 분석기 기반 ROUGE 계산
- **생성 모델 파라미터**: beam search, length penalty 등 최적화
- **Solar API 통합**: 외부 API 기반 실험도 자동화 지원

### 7.2 메모리 및 성능 최적화
```yaml
# 메모리 제약 고려 설정
constraints:
  memory_optimization:
    if: encoder_max_len > 512
    then: 
      per_device_train_batch_size: max 16
      gradient_accumulation_steps: min 2
  
  performance_optimization:
    dataloader_num_workers: 4
    pin_memory: true
    fp16: true
```

## 8. 구현 우선순위

### 8.1 Phase 1: 기반 구조 구축 (1주차)
1. **ConfigManager 구현**: 기존/신규 설정 호환 시스템
2. **base_config.yaml 작성**: 기존 config.yaml 확장
3. **유틸리티 모듈**: metrics.py, data_utils.py 구현
4. **호환성 테스트**: 기존 코드와의 동작 확인

### 8.2 Phase 2: 모듈화 및 자동화 (2주차)
1. **baseline.ipynb 모듈화**: trainer.py, modules/ 구현
2. **WandB Sweep 설정**: sweep/ 폴더 설정 파일들 작성
3. **sweep_runner.py 구현**: 6조 방식 자동화 스크립트
4. **첫 번째 자동화 실험**: 간단한 하이퍼파라미터 탐색

### 8.3 Phase 3: 고도화 및 최적화 (3주차)
1. **experiment_runner.py**: 통합 실험 관리 시스템
2. **모델 비교 자동화**: 여러 모델 성능 비교
3. **결과 분석 도구**: 실험 결과 자동 분석 및 시각화
4. **Solar API 통합**: 외부 API 기반 실험 자동화

## 9. 기대 효과

### 9.1 즉시 효과 (Phase 1 완료 후)
- ✅ 설정 관리 체계화
- ✅ 실험 재현성 보장
- ✅ 팀 협업 효율성 증대

### 9.2 중기 효과 (Phase 2 완료 후)
- 🚀 하이퍼파라미터 자동 최적화
- 📊 체계적 실험 추적 및 비교
- ⏱️ 실험 시간 단축 (수동 → 자동)

### 9.3 장기 효과 (Phase 3 완료 후)
- 🎯 최적 모델 자동 발견
- 🔄 지속적 성능 개선 시스템
- 📈 대회 순위 향상 기대

## 10. 결론

본 설계는 6조의 성공적인 실험 자동화 전략을 NLP 프로젝트에 맞게 적응시키면서도, 기존 팀원들의 워크플로우를 존중하는 점진적 도입 방식을 제시합니다. 

**핵심 가치:**
- **호환성**: 기존 코드 완전 보존
- **선택성**: 원하는 기능만 선택적 사용
- **확장성**: 미래 요구사항에 유연하게 대응
- **실용성**: 대회 특성과 팀 상황에 최적화

이를 통해 팀의 실험 효율성을 크게 높이면서도 학습 곡선을 최소화할 수 있을 것으로 기대됩니다.
