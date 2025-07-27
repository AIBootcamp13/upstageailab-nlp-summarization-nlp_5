# AIStages 대회 통합 가이드 - 완전 정리

## 목차
1. [프로젝트 현황 분석](#1-프로젝트-현황-분석)
2. [새로운 내용 요약](#2-새로운-내용-요약)
3. [즉시 적용 가능한 개선사항](#3-즉시-적용-가능한-개선사항)
4. [단계별 통합 전략](#4-단계별-통합-전략)
5. [실전 코드 예시](#5-실전-코드-예시)
6. [팀 협업 가이드](#6-팀-협업-가이드)

---

## 1. 프로젝트 현황 분석

### 1.1 기존 문서화 상태

#### 이미 구현/문서화된 내용
- ✅ **기본 베이스라인**: `code/baseline.ipynb`
- ✅ **Solar API 활용**: `code/solar_api.ipynb`
- ✅ **UV 패키지 관리자**: `docs/uv_package_manager_guide.md`
- ✅ **프로젝트 구조**: 체계적인 폴더 구조
- ✅ **기본 config**: `code/config.yaml`

#### 새로 추가된 문서
- 🆕 **AIStages 환경 설정**: 서버 특화 설정
- 🆕 **하이퍼파라미터 튜닝**: WandB Sweep을 통한 베이지안 최적화 (✅ 구현 완료)
- 🆕 **텍스트 데이터 분석**: 전처리 및 시각화
- 🆕 **WandB 실험 관리**: 팀 협업 도구

### 1.2 개선 필요 사항

| 영역 | 현재 상태 | 개선 방안 |
| 하이퍼파라미터 | 수동 조정 | WandB Sweep (베이지안 최적화) ✅ |
| 하이퍼파라미터 최적화 | WandB Sweep | 구현 완료, 즉시 사용 가능 |
| 환경 설정 | 수동 pip 설치 | UV + 자동화 스크립트 |
| 실험 관리 | 로컬 저장 | WandB 통합 |

| 데이터 분석 | 기본 EDA | 체계적 전처리 파이프라인 |

## 2. 새로운 내용 요약

### 2.1 텍스트 데이터 분석
- **개인정보 마스킹**: 8가지 PII 토큰 처리
- **Special Token**: #Person1# ~ #Person7# 추가
- **전처리**: 구어체 표현 정제 (ㅋㅋ→웃음)
- **시각화**: 워드클라우드, TF-IDF 분석

### 2.2 WandB 실험 관리
- **팀 협업**: 실시간 실험 결과 공유
- **자동 추적**: 하이퍼파라미터, 메트릭 기록
- **Sweep**: 자동 하이퍼파라미터 탐색
- **아티팩트**: 모델/데이터 버전 관리
### 2.3 하이퍼파라미터 최적화 (WandB Sweep)
- **현재 상태**: WandB Sweep을 통한 베이지안 최적화 ✅ 구현 완료
- **핵심 기능**: 
  - 베이지안 최적화 (Optuna와 동등)
  - Hyperband 조기 종료
  - 실시간 실험 추적
- **주요 파라미터**:
- **주요 파라미터**: 
  - Learning Rate: 1e-5 ~ 5e-4
  - Batch Size: 8, 16, 32
  - Epochs: 10-30
  - Warmup Ratio: 0.0-0.2

## 3. 즉시 적용 가능한 개선사항
## 3. 즉시 적용 가능한 개선사항

### 3.1 WandB Sweep으로 하이퍼파라미터 최적화 ✅

1. **설치 및 로그인**
   ```bash
   pip install wandb
   wandb login
   ```

2. **Sweep 실행**
   ```bash
   # 하이퍼파라미터 최적화
   python code/sweep_runner.py \
     --base-config config/base_config.yaml \
     --sweep-config hyperparameter_sweep \
     --count 50
   ```

3. **Sweep 설정 파일** (`config/sweep/hyperparameter_sweep.yaml`)
   ```yaml
   method: bayes  # 베이지안 최적화
   metric:
     name: rouge_combined_f1
     goal: maximize
   
   parameters:
     learning_rate:
       distribution: log_uniform_values
       min: 1.0e-6
       max: 1.0e-4
     
     per_device_train_batch_size:
       values: [8, 16, 32, 64]
   ```

4. **장점**
   - Optuna와 동등한 베이지안 최적화
   - WandB 대시보드에서 실시간 모니퇁링
   - 팀 협업 및 결과 공유 용이
tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
```

### 3.3 데이터 전처리
```python
# 구어체 정제
def clean_dialogue(text):
    text = re.sub(r'ㅋ+', '웃음', text)
    text = re.sub(r'ㅎ+', '웃음', text)
    text = re.sub(r'ㅠ+|ㅜ+', '슬픔', text)
    return text.strip()

train['dialogue_clean'] = train['dialogue'].apply(clean_dialogue)
```

## 4. 단계별 통합 전략

### Phase 1: 기초 개선 (1-2일)
1. **UV 환경 설정**
   ```bash
   # UV 설치 및 환경 설정
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv pip install -r requirements.txt --system
   ```

2. **데이터 전처리 적용**
   ```python
   # 전처리 파이프라인 적용
   preprocessor = DialoguePreprocessor()
   train_processed = preprocessor.preprocess(train)
   ```

3. **WandB 설정**
   ```python
   # WandB 초기화
   wandb.init(
       project="dialogue-summarization",
       entity="your-team-name",
       config=config
   )
   ```

### Phase 2: 실헐 최적화 (3-4일)
1. **체계적 하이퍼파라미터 탐색** (현재 관리 및 Grid Search 기반)
   ```python
   # 현재 구현된 Grid Search 방법
   learning_rates = [1e-5, 2e-5, 3e-5, 5e-5]
   batch_sizes = [8, 16, 32]
   
   # auto_experiment_runner.py 활용으로 순차 실헐 실행
   # 미래: Optuna 통합 시 Bayesian Optimization 적용 예정
   # 미래: Optuna 통합 시 Bayesian Optimization 적용 예정
   ```

   **향후 Optuna 통합 예정지**:
   ```python
   # TODO: 향후 구현 예정
   # def optuna_hp_space(trial):
   #     return {
   #         "learning_rate": trial.suggest_loguniform('lr', 1e-5, 5e-4),
   #         "per_device_train_batch_size": trial.suggest_categorical('bs', [8, 16, 32]),
   #         "num_train_epochs": trial.suggest_int('epochs', 10, 30, step=5)
   #     }
   ```

2. **데이터 증강**
   - Paraphrasing
   - Back-translation
   - 노이즈 추가

### Phase 3: 고급 최적화 (5-7일)
1. **모델 앙상블**
2. **Advanced 전처리**
3. **Custom Loss Functions**

## 5. 실전 코드 예시

> **⚠️ 중요 안내**: 아래 코드 예시는 Optuna 통합이 포함되어 있지만, **현재 시점에서는 미구현 상태**입니다. 
> 이는 향후 구현 예정인 기능의 설계 도면으로 참고하시기 바랍니다.
> **현재 사용 가능한 기능**: `auto_experiment_runner.py`를 통한 순차적 실험 실행, WandB Sweep을 통한 하이퍼파라미터 탐색
> **🎉 현재 사용 가능**: WandB Sweep을 통해 Optuna와 동등한 수준의 베이지안 최적화가 가능합니다!
> 아래 코드는 WandB Sweep을 활용한 하이퍼파라미터 최적화 예시입니다.
```python
# train_integrated.py - 향후 구현 예정인 기능들
import wandb
# import optuna  # 향후 추가 예정
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import pandas as pd
import numpy as np
from rouge_score import rouge_scorer

class IntegratedTrainer:
    def __init__(self, config_path="config_integrated.yaml"):
        self.config = self.load_config(config_path)
        self.setup_environment()
        
    def setup_environment(self):
        """환경 설정"""
        # WandB 초기화
        wandb.init(
            project=self.config['wandb']['project'],
            entity=self.config['wandb']['entity'],
            config=self.config
        )
        
        # Tokenizer & Model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model']['name']
        )
        self.add_special_tokens()
        
    def add_special_tokens(self):
        """Special Token 추가"""
        special_tokens = [
            f'#Person{i}#' for i in range(1, 8)
        ] + [
            '#PhoneNumber#', '#Address#', '#DateOfBirth#',
            '#PassportNumber#', '#SSN#', '#CardNumber#',
            '#CarNumber#', '#Email#'
        ]
        
        self.tokenizer.add_special_tokens({
            'additional_special_tokens': special_tokens
        })
        
    def preprocess_data(self, df):
        """데이터 전처리"""
        # 구어체 정제
        df['dialogue'] = df['dialogue'].apply(self.clean_dialogue)
        
        # 토큰화
        def tokenize_function(examples):
            model_inputs = self.tokenizer(
                examples['dialogue'],
                max_length=self.config['model']['max_encoder_length'],
                truncation=True,
                padding="max_length"
            )
            
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    examples['summary'],
                    max_length=self.config['model']['max_decoder_length'],
                    truncation=True,
                    padding="max_length"
                )
            
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        return df.map(tokenize_function, batched=True)
    
    @staticmethod
    def clean_dialogue(text):
        """텍스트 정제"""
        import re
        text = re.sub(r'ㅋ+', '웃음', text)
        text = re.sub(r'ㅎ+', '웃음', text)
        text = re.sub(r'ㅠ+|ㅜ+', '슬픔', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def compute_metrics(self, eval_pred):
        """ROUGE 메트릭 계산"""
        predictions, labels = eval_pred
        
        # 디코딩
        decoded_preds = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )
        decoded_labels = self.tokenizer.batch_decode(
            labels, skip_special_tokens=True
        )
        
        # ROUGE 계산
        scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], use_stemmer=False
        )
        
        scores = []
        for pred, label in zip(decoded_preds, decoded_labels):
            score = scorer.score(label, pred)
            scores.append({
                'rouge1': score['rouge1'].fmeasure,
                'rouge2': score['rouge2'].fmeasure,
                'rougeL': score['rougeL'].fmeasure
            })
        
        # 평균 계산
        result = {
            'rouge1': np.mean([s['rouge1'] for s in scores]),
            'rouge2': np.mean([s['rouge2'] for s in scores]),
            'rougeL': np.mean([s['rougeL'] for s in scores])
        }
        
        # WandB 로깅
        wandb.log(result)
        
        return result
    
    def create_trainer(self, trial=None):
        """Trainer 생성"""
        # 하이퍼파라미터 설정
        if trial:
            hp = self.get_optuna_params(trial)
        else:
            hp = self.config['training']
        
        # Training Arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config['general']['output_dir'],
            num_train_epochs=hp['num_train_epochs'],
            per_device_train_batch_size=hp['per_device_train_batch_size'],
            learning_rate=hp['learning_rate'],
            warmup_ratio=hp.get('warmup_ratio', 0.1),
            weight_decay=hp.get('weight_decay', 0.01),
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=500,
            save_steps=500,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="rougeL",
            greater_is_better=True,
            fp16=True,
            gradient_checkpointing=True,
            predict_with_generate=True,
            generation_max_length=self.config['model']['max_decoder_length'],
            report_to="wandb"
        )
        
        # Model 초기화
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config['model']['name']
        )
        model.resize_token_embeddings(len(self.tokenizer))
        
        # Trainer
        return Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )
    
    def get_optuna_params(self, trial):
        """Optuna 하이퍼파라미터"""
        return {
            'learning_rate': trial.suggest_loguniform('lr', 1e-5, 5e-4),
            'per_device_train_batch_size': trial.suggest_categorical(
                'batch_size', [8, 16, 32]
            ),
            'num_train_epochs': trial.suggest_int('epochs', 10, 30, step=5),
            'warmup_ratio': trial.suggest_float('warmup', 0.0, 0.2),
            'weight_decay': trial.suggest_categorical(
                'weight_decay', [0.0, 0.01, 0.1]
            )
        }
    
    def optuna_objective(self, trial):
        """Optuna 목적 함수"""
        # Trainer 생성
        trainer = self.create_trainer(trial)
        
        # 학습
        trainer.train()
        
        # 평가
        metrics = trainer.evaluate()
        
        # WandB에 trial 정보 기록
        wandb.log({
            f"trial_{trial.number}/rouge1": metrics['eval_rouge1'],
            f"trial_{trial.number}/rouge2": metrics['eval_rouge2'],
            f"trial_{trial.number}/rougeL": metrics['eval_rougeL']
        })
        
        return metrics['eval_rougeL']
    
    def run_hyperparameter_search(self, n_trials=20):
        """하이퍼파라미터 탐색"""
        study = optuna.create_study(
            direction="maximize",
            study_name="dialogue-summarization-hp-search"
        )
        
        study.optimize(self.optuna_objective, n_trials=n_trials)
        
        # 최적 파라미터 저장
        best_params = study.best_params
        wandb.config.update({"best_params": best_params})
        
        print(f"Best parameters: {best_params}")
        print(f"Best ROUGE-L: {study.best_value}")
        
        return best_params
    
    def train_with_best_params(self, best_params=None):
        """최적 파라미터로 최종 학습"""
        if best_params:
            self.config['training'].update(best_params)
        
        # Trainer 생성 및 학습
        trainer = self.create_trainer()
        trainer.train()
        
        # 최종 평가
        final_metrics = trainer.evaluate()
        print(f"Final ROUGE scores: {final_metrics}")
        
        # 모델 저장
        trainer.save_model(f"{self.config['general']['output_dir']}/best_model")
        
        # WandB Artifact 저장
        artifact = wandb.Artifact(
            name="dialogue-summarization-model",
            type="model"
        )
        artifact.add_dir(f"{self.config['general']['output_dir']}/best_model")
        wandb.log_artifact(artifact)
        
        return trainer

# 실행
if __name__ == "__main__":
    # 통합 학습기 초기화
    trainer = IntegratedTrainer()
    
    # 데이터 로드 및 전처리
    train_df = pd.read_csv("data/train.csv")
    eval_df = pd.read_csv("data/dev.csv")
    
    trainer.train_dataset = trainer.preprocess_data(train_df)
    trainer.eval_dataset = trainer.preprocess_data(eval_df)
    
    # 옵션 1: 하이퍼파라미터 탐색
    best_params = trainer.run_hyperparameter_search(n_trials=20)
    
    # 옵션 2: 최적 파라미터로 학습
    trainer.train_with_best_params(best_params)
    
    wandb.finish()
```

### 5.2 Config 파일 (YAML)
```yaml
# config_integrated.yaml
general:
  project_name: "dialogue-summarization"
  data_path: "./data/"
  output_dir: "./outputs/integrated_experiment"
  seed: 42

model:
  name: "gogamza/kobart-base-v2"
  max_encoder_length: 512
  max_decoder_length: 128

training:
  learning_rate: 3e-5
  per_device_train_batch_size: 16
  num_train_epochs: 20
  warmup_ratio: 0.1
  weight_decay: 0.01
  gradient_accumulation_steps: 1
  fp16: true
  save_total_limit: 3
  early_stopping_patience: 3

wandb:
  project: "dialogue-summarization"
  entity: "nlp-team-5"
  tags: ["kobart", "integrated", "planned-optuna"]
  notes: "통합 실험 계획 - 향후 Optuna + WandB + 전처리"

optuna:
  n_trials: 20
  direction: "maximize"
  metric: "rougeL"
  pruner: "MedianPruner"
```

## 6. 팀 협업 가이드

### 6.1 Git 워크플로우
```bash
# 1. 최신 코드 동기화
git fetch upstream main
git merge FETCH_HEAD

# 2. 실험 브랜치 생성
git checkout -b exp/hp-tuning-lr

# 3. 실험 후 커밋
git add -A
git commit -m "feat: Add Optuna hyperparameter tuning"

# 4. Push 및 PR
git push origin exp/hp-tuning-lr
```

### 6.2 실험 명명 규칙
```python
# WandB 실험 이름
experiment_name = f"{model_type}_{data_version}_lr{lr}_bs{bs}_ep{epochs}"
# 예: kobart_v2_lr3e-5_bs16_ep20

# 브랜치 이름
branch_name = f"exp/{feature}-{description}"
# 예: exp/optuna-integration, exp/data-augmentation
```

### 6.3 실험 기록 템플릿
```markdown
## 실험 #001: Optuna 하이퍼파라미터 최적화

**날짜**: 2025-01-27
**실험자**: @username
**WandB Run**: [링크](https://wandb.ai/...)

### 목적
- Learning Rate 최적값 탐색
- Batch Size와 성능 관계 분석

### 설정
- Model: gogamza/kobart-base-v2
- Optuna Trials: 20
- 탐색 범위:
  - LR: 1e-5 ~ 5e-4
  - BS: [8, 16, 32]
  - Epochs: 10-30

### 결과
- Best LR: 2.3e-5
- Best BS: 16
- Best Epochs: 22
- Final ROUGE-L: 0.4856

### 인사이트
1. LR은 2e-5 ~ 3e-5 범위가 최적
2. BS 32는 메모리 부족으로 실패
3. 20 epochs 이후 성능 포화

### 다음 단계
- [ ] Warmup ratio 추가 탐색
- [ ] Learning rate scheduler 실험
- [ ] 데이터 증강 적용
```

### 6.4 체크리스트

#### 일일 체크리스트
- [ ] 코드 동기화 (git pull)
- [ ] WandB 실험 결과 확인
- [ ] 팀 슬랙에 진행상황 공유
- [ ] 다음 실험 계획 수립

#### 실험 전 체크리스트
- [ ] GPU 사용 가능 확인
- [ ] 데이터 경로 확인
- [ ] Config 파일 검증
- [ ] WandB 프로젝트 확인
- [ ] 이전 실험 결과 리뷰

#### 제출 전 체크리스트
- [ ] 모델 성능 검증
- [ ] 추론 코드 테스트
- [ ] 제출 파일 형식 확인
- [ ] 팀원 리뷰 완료

## 결론

이 통합 가이드를 통해:

1. **즉시 적용**: UV 환경 설정, 데이터 전처리, Special Token 추가
2. **단계별 개선**: WandB → Optuna → 고급 기법 순차 적용
3. **체계적 실험**: 명명 규칙과 템플릿으로 일관성 유지
4. **팀 협업 강화**: Git, WandB로 효율적 공유

핵심은 **작은 개선부터 시작**하여 점진적으로 발전시키는 것입니다. 각 단계별로 성능 향상을 측정하고 기록하여 최종적으로 최고의 모델을 만들어내세요!
