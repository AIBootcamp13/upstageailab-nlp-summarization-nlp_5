# WandB (Weights & Biases) 실험 관리 가이드

## 목차
1. [WandB 소개](#1-wandb-소개)
2. [설치 및 초기 설정](#2-설치-및-초기-설정)
3. [프로젝트 및 팀 설정](#3-프로젝트-및-팀-설정)
4. [실험 추적 기본](#4-실험-추적-기본)
5. [하이퍼파라미터 관리](#5-하이퍼파라미터-관리)
6. [실험 결과 시각화](#6-실험-결과-시각화)
7. [프로젝트 통합 코드](#7-프로젝트-통합-코드)

---

## 1. WandB 소개

### 1.1 WandB란?
WandB(Weights & Biases)는 머신러닝 실험을 추적, 비교, 시각화하는 MLOps 도구입니다.

### 1.2 주요 기능
- **실험 추적**: 하이퍼파라미터, 메트릭, 모델 버전 관리
- **실시간 모니터링**: 학습 진행 상황 실시간 확인
- **팀 협업**: 팀원간 실험 결과 공유
- **시각화**: 자동 차트 생성 및 커스텀 대시보드
- **모델 관리**: 모델 아티팩트 저장 및 버전 관리

## 2. 설치 및 초기 설정

### 2.1 WandB 설치
```bash
# pip로 설치
pip install wandb -qU

# conda로 설치 (선택사항)
conda install -c conda-forge wandb
```

### 2.2 계정 생성
1. https://wandb.ai 접속
2. 우측 상단 "Sign Up" 클릭
3. GitHub/Google/Microsoft 계정으로 가입 가능

### 2.3 API Key 설정
```python
import wandb

# 로그인 (처음 한 번만)
wandb.login()
# 브라우저에서 https://wandb.ai/authorize 접속하여 API key 복사
# 터미널에 붙여넣기

# 또는 환경변수로 설정
# export WANDB_API_KEY=your_api_key_here
```

### 2.4 설정 확인
```python
# 로그인 상태 확인
print(f"WandB 사용자: {wandb.api.viewer()['username']}")
```

## 3. 프로젝트 및 팀 설정

### 3.1 프로젝트 생성
1. https://wandb.ai/home 접속
2. 좌측 "My projects" → "Create new project"
3. 프로젝트 정보 입력:
   - **Project name**: dialogue-summarization
   - **Description**: 일상 대화 요약 모델 실험
   - **Visibility**: Private (팀 전용)

### 3.2 팀 생성 및 관리
1. 좌측 하단 "Teams" → "Create new team"
2. 팀 이름 입력 (예: nlp-team-5)
3. 팀원 초대:
   - Settings → Members → Invite
   - 이메일 또는 사용자명으로 초대

### 3.3 팀 프로젝트 설정
```python
# 팀 프로젝트로 초기화
wandb.init(
    project="dialogue-summarization",
    entity="nlp-team-5",  # 팀 이름
    name="kobart-baseline-v1",
    tags=["baseline", "kobart"],
    notes="KoBART 기반 베이스라인 실험"
)
```

## 4. 실험 추적 기본

### 4.1 기본 사용법
```python
import wandb

# 실험 시작
wandb.init(
    project="dialogue-summarization",
    config={
        "learning_rate": 3e-5,
        "batch_size": 16,
        "epochs": 20,
        "model": "gogamza/kobart-base-v2",
        "max_length": 512,
        "optimizer": "adamw"
    }
)

# 하이퍼파라미터 접근
config = wandb.config

# 메트릭 기록
for epoch in range(config.epochs):
    train_loss = train_one_epoch()
    val_loss, rouge_score = evaluate()
    
    # 로그 기록
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "rouge_1": rouge_score['rouge1'],
        "rouge_2": rouge_score['rouge2'],
        "rouge_l": rouge_score['rougeL']
    })

# 실험 종료
wandb.finish()
```

### 4.2 아티팩트 저장
```python
# 모델 저장
artifact = wandb.Artifact(
    name="kobart-dialogue-summary",
    type="model",
    description="Fine-tuned KoBART for dialogue summarization"
)

# 모델 파일 추가
artifact.add_dir("./model_output")
wandb.log_artifact(artifact)

# 데이터셋 저장
data_artifact = wandb.Artifact(
    name="dialogue-dataset",
    type="dataset"
)
data_artifact.add_file("train.csv")
data_artifact.add_file("dev.csv")
wandb.log_artifact(data_artifact)
```

## 5. 하이퍼파라미터 관리

### 5.1 Sweep 설정 (자동 하이퍼파라미터 탐색)
```python
# sweep_config.yaml
sweep_config = {
    'method': 'bayes',  # grid, random, bayes
    'metric': {
        'name': 'val_rouge_l',
        'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 5e-4
        },
        'batch_size': {
            'values': [8, 16, 32]
        },
        'epochs': {
            'values': [15, 20, 25]
        },
        'warmup_ratio': {
            'distribution': 'uniform',
            'min': 0.0,
            'max': 0.2
        },
        'weight_decay': {
            'values': [0.0, 0.01, 0.1]
        }
    }
}

# Sweep 생성
sweep_id = wandb.sweep(sweep_config, project="dialogue-summarization")
```

### 5.2 Sweep 실행
```python
def train_sweep():
    # WandB 초기화
    wandb.init()
    config = wandb.config
    
    # 모델 및 학습 설정
    model = create_model()
    training_args = Seq2SeqTrainingArguments(
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        num_train_epochs=config.epochs,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        # ... 기타 설정
    )
    
    # 학습 실행
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        # ... 기타 설정
    )
    
    # 학습 및 평가
    trainer.train()
    metrics = trainer.evaluate()
    
    # 결과 기록
    wandb.log(metrics)

# Sweep 실행 (10개 실험)
wandb.agent(sweep_id, train_sweep, count=10)
```

## 6. 실험 결과 시각화

### 6.1 커스텀 차트 생성
```python
# 학습 곡선 비교
wandb.log({
    "train_loss": wandb.plot.line_series(
        xs=range(len(train_losses)),
        ys=[train_losses, val_losses],
        keys=["train", "validation"],
        title="Loss Curves",
        xname="epoch"
    )
})

# Rouge 점수 막대 차트
wandb.log({
    "rouge_scores": wandb.plot.bar(
        table=wandb.Table(
            data=[
                ["ROUGE-1", rouge_scores['rouge1']],
                ["ROUGE-2", rouge_scores['rouge2']],
                ["ROUGE-L", rouge_scores['rougeL']]
            ],
            columns=["Metric", "Score"]
        ),
        label="Metric",
        value="Score",
        title="ROUGE Scores"
    )
})
```

### 6.2 실험 비교 대시보드
```python
# 여러 실험 결과 비교
api = wandb.Api()
runs = api.runs("nlp-team-5/dialogue-summarization")

# 최고 성능 실험 찾기
best_run = max(runs, key=lambda x: x.summary.get('val_rouge_l', 0))
print(f"Best run: {best_run.name}")
print(f"Best ROUGE-L: {best_run.summary['val_rouge_l']}")
print(f"Config: {best_run.config}")
```

## 7. 프로젝트 통합 코드

### 7.1 WandB 통합 Trainer
```python
from transformers import Seq2SeqTrainer, TrainingArguments
import wandb

class WandbTrainer(Seq2SeqTrainer):
    """WandB 통합 Trainer 클래스"""
    
    def __init__(self, *args, wandb_project=None, wandb_entity=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        
    def train(self, *args, **kwargs):
        # WandB 초기화
        wandb.init(
            project=self.wandb_project,
            entity=self.wandb_entity,
            config=self.args.to_dict(),
            name=self.args.output_dir.split('/')[-1],
            tags=["training", "dialogue-summarization"]
        )
        
        # 학습 실행
        output = super().train(*args, **kwargs)
        
        # 최종 결과 기록
        final_metrics = self.evaluate()
        wandb.log({
            "final/" + k: v for k, v in final_metrics.items()
        })
        
        # 모델 저장
        if self.args.save_model_as_artifact:
            self._save_model_artifact()
        
        wandb.finish()
        return output
    
    def _save_model_artifact(self):
        """모델을 WandB Artifact로 저장"""
        artifact = wandb.Artifact(
            name=f"model-{wandb.run.id}",
            type="model"
        )
        artifact.add_dir(self.args.output_dir)
        wandb.log_artifact(artifact)
```

### 7.2 실험 설정 통합
```python
# config_wandb.yaml
wandb_config = {
    # WandB 설정
    'wandb': {
        'project': 'dialogue-summarization',
        'entity': 'nlp-team-5',
        'tags': ['kobart', 'baseline'],
        'notes': 'KoBART 베이스라인 실험'
    },
    
    # 모델 설정
    'model': {
        'name': 'gogamza/kobart-base-v2',
        'max_encoder_length': 512,
        'max_decoder_length': 128
    },
    
    # 학습 설정
    'training': {
        'learning_rate': 3e-5,
        'batch_size': 16,
        'epochs': 20,
        'warmup_ratio': 0.1,
        'weight_decay': 0.01,
        'gradient_accumulation_steps': 1,
        'fp16': True,
        'save_model_as_artifact': True
    },
    
    # 평가 설정
    'evaluation': {
        'eval_steps': 500,
        'save_steps': 500,
        'logging_steps': 100,
        'metric_for_best_model': 'rouge_l',
        'greater_is_better': True
    }
}
```

### 7.3 통합 실행 스크립트
```python
# train_with_wandb.py
import wandb
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import yaml

def main():
    # 설정 로드
    with open('config_wandb.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # WandB 초기화
    wandb.init(
        project=config['wandb']['project'],
        entity=config['wandb']['entity'],
        config=config,
        tags=config['wandb']['tags'],
        notes=config['wandb']['notes']
    )
    
    # 모델 및 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    model = AutoModelForSeq2SeqLM.from_pretrained(config['model']['name'])
    
    # 데이터 로드 및 전처리
    dataset = load_dataset('csv', data_files={
        'train': 'train.csv',
        'validation': 'dev.csv'
    })
    
    # Trainer 설정
    trainer = WandbTrainer(
        model=model,
        tokenizer=tokenizer,
        args=create_training_args(config),
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        compute_metrics=compute_metrics,
        wandb_project=config['wandb']['project'],
        wandb_entity=config['wandb']['entity']
    )
    
    # 학습 실행
    trainer.train()
    
    # 최종 평가
    final_results = trainer.evaluate()
    print(f"Final ROUGE-L: {final_results['eval_rouge_l']}")
    
    # 베스트 모델 저장
    trainer.save_model(f"./best_model_{wandb.run.id}")

if __name__ == "__main__":
    main()
```

### 7.4 팀 협업 베스트 프랙티스

#### 실험 명명 규칙
```python
# 일관된 실험 이름 사용
experiment_name = f"{model_name}_{dataset_version}_lr{learning_rate}_bs{batch_size}"

wandb.init(
    name=experiment_name,
    group=f"{model_name}_experiments",  # 모델별 그룹화
    job_type="training"  # 작업 유형 명시
)
```

#### 실험 태그 전략
```python
tags = []
# 모델 태그
tags.append(f"model:{model_name}")
# 데이터 태그
tags.append(f"data:v{dataset_version}")
# 실험 유형
tags.append("baseline" if is_baseline else "experimental")
# 실험자
tags.append(f"experimenter:{username}")

wandb.init(tags=tags)
```

#### 결과 공유 템플릿
```markdown
## 실험 결과 요약

**실험 ID**: {run_id}
**실험자**: {username}
**날짜**: {date}

### 설정
- Model: {model_name}
- Learning Rate: {learning_rate}
- Batch Size: {batch_size}
- Epochs: {epochs}

### 결과
- Best ROUGE-1: {rouge1}
- Best ROUGE-2: {rouge2}
- Best ROUGE-L: {rougeL}
- Training Time: {time}

### 주요 발견사항
- {key_findings}

### 다음 단계
- {next_steps}
```

## 결론

WandB를 활용하면 실험 관리가 체계적이고 효율적으로 변합니다:

1. **실험 추적**: 모든 실험 기록이 자동 저장
2. **팀 협업**: 실시간 결과 공유 및 비교
3. **재현성**: 설정과 결과를 완벽하게 기록
4. **최적화**: Sweep으로 자동 하이퍼파라미터 탐색

팀 프로젝트에서는 일관된 명명 규칙과 태그 전략을 사용하여 실험을 체계적으로 관리하는 것이 중요합니다.
