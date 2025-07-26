# 6조 CV 프로젝트 실험 자동화 시스템 분석 보고서

## 개요

본 문서는 Upstage AI Lab CV Classification 대회에서 2등을 차지한 6조 팀의 실험 자동화 시스템을 상세 분석하여, NLP 요약 프로젝트에 적용할 수 있는 핵심 전략과 구현 방법을 도출한 분석 보고서입니다.

## 1. 6조 프로젝트 구조 분석

### 1.1 전체 디렉토리 구조
```
cv_6/
├── config/                      # 설정 파일 중심 구조
│   ├── main_config.yaml         # 메인 학습 설정
│   ├── sweep-config.yaml        # WandB Sweep 하이퍼파라미터 튜닝
│   ├── transforms_config.yaml   # 데이터 증강 설정
│   ├── inference_config.yaml    # 추론 전용 설정
│   └── *.md                     # 설정 관련 문서들
├── models/                      # 모델 아키텍처
├── datasets/                    # 데이터셋 및 전처리
├── trainer/                     # 학습 로직 모듈화
├── utils/                       # 유틸리티 함수들
├── main.py                      # 기본 학습 스크립트
├── main_sweep_test.py          # 자동화 실험 스크립트
└── inference_*.py              # 다양한 추론 스크립트
```

### 1.2 핵심 성공 요인
1. **완전한 모듈화**: 모든 기능이 독립적인 모듈로 분리
2. **설정 중심 설계**: YAML 파일로 모든 하이퍼파라미터 외부화
3. **자동화 우선**: WandB Sweep을 통한 체계적 실험 관리
4. **재현성 보장**: 시드 고정 및 환경 설정 표준화

## 2. YAML 설정 구조 심층 분석

### 2.1 main_config.yaml - 핵심 학습 설정
```yaml
# 기본 설정
n_splits: 10                     # K-Fold 교차 검증
BATCH_SIZE: 54                   # 배치 크기
EPOCHS: 100                      # 최대 에폭
SEED: 42                         # 재현성을 위한 시드

# 모델 및 데이터 설정
model_type: convnext             # 모델 타입 (efficientnet, resnet, convnext, swin)
DATASET: FastImageDataset        # 데이터셋 클래스
MODEL: ConvNeXtArcFaceModel      # 모델 클래스
training_mode: on_amp            # 학습 모드 (on_amp, normal)

# 학습률 설정
backbone_lr: 0.00001            # 백본 학습률
use_differential_lr: False       # 차등 학습률 사용 여부
use_unfreeze: True              # 언프리징 사용 여부
num_blocks_to_unfreeze: 4       # 언프리징할 블록 수

# 조기 종료
patience: 15                     # 조기 종료 인내
delta: 0.01                     # 개선 판단 기준

# 외부 설정 파일 참조
TRANSFORMS_PATH: config/transforms_config.yaml

# 손실 함수, 옵티마이저, 스케줄러 설정
triplet_loss_weight: 0.3

optimizer:
  name: AdamW
  params:
    lr: 0.0001
    weight_decay: 0.0001

scheduler:
  name: cosine
  params:
    T_max: 100
    eta_min: 0.000001

loss:
  name: FocalLoss
  params:
    gamma: 2.0
    label_smoothing: 0.1
```

**핵심 특징:**
- **계층적 구조**: 관련 설정들을 그룹화하여 관리
- **외부 참조**: transforms_config.yaml을 별도로 분리
- **타입 명시**: 문자열과 숫자를 명확히 구분

### 2.2 sweep-config.yaml - WandB Sweep 설정
```yaml
project: sweep-test
entity: fkjy132
program: main.py                 # 실행할 스크립트

method: bayes                    # 베이지안 최적화

# 최적화 목표
metric:
  name: valid_max_accuracy       # 최적화할 메트릭
  goal: maximize                 # 최대화

# 하이퍼파라미터 탐색 공간
parameters:
  batch_size:
    values: [4, 8, 16, 32]
  
  # 차등 학습률
  head_lr:
    distribution: log_uniform_values
    min: 1.0e-4
    max: 1.0e-2
  backbone_lr:
    distribution: log_uniform_values
    min: 1.0e-5
    max: 1.0e-3
  
  # 정규화
  weight_decay:
    values: [0.01, 0.001]
  
  # 모델 구조
  num_blocks_to_unfreeze:
    values: [1, 2, 3, 4]
  
  # ArcFace 하이퍼파라미터
  arcface_s:
    distribution: uniform
    min: 25.0
    max: 45.0
  arcface_m:
    distribution: uniform
    min: 0.3
    max: 0.55
  
  # 스케줄러
  scheduler_T_0:
    values: [10, 20, 30]
  
  # 손실 함수
  loss_gamma:
    distribution: uniform
    min: 1.5
    max: 3.0
```

**핵심 특징:**
- **다양한 분포**: log_uniform, uniform, discrete values 조합
- **CV 특화**: ArcFace, backbone/head 차등 학습률 등
- **베이지안 최적화**: 효율적인 하이퍼파라미터 탐색

### 2.3 transforms_config.yaml - 데이터 증강 설정
```yaml
transforms:
  train:    # 3단계 파이프라인
    # 1단계: Augraphy (문서 품질 저하 시뮬레이션)
    - backend: augraphy
      name: OneOf
      params:
        p: 0.5
        transforms:
          - backend: augraphy
            name: InkBleed
            params: { intensity_range: [0.1, 0.3] }
          - backend: augraphy
            name: LowInkRandomLines
    
    # 2단계: Albumentations (기하학적/색상 증강)
    - backend: albumentations
      name: ShiftScaleRotate
      params: { p: 0.5, shift_limit: 0.05, scale_limit: 0.05, rotate_limit: 10 }
    
    # 3단계: Torchvision (텐서 변환 및 정규화)
    - backend: torchvision
      name: ToTensor
    - backend: torchvision
      name: Normalize
      params:
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
  
  val:      # 검증용 파이프라인 (더 제한적)
    - backend: augraphy
      name: BleedThrough
    - backend: albumentations
      name: Resize
      params: { height: 384, width: 384 }
    - backend: torchvision
      name: ToTensor
    - backend: torchvision
      name: Normalize
      params:
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
```

**핵심 특징:**
- **다중 백엔드**: Augraphy, Albumentations, Torchvision 조합
- **단계별 처리**: 문서→기하학적→텐서 변환 순서
- **검증 전용**: 검증 데이터에는 보수적 증강 적용

## 3. 자동화 스크립트 분석 (main_sweep_test.py)

### 3.1 핵심 구조
```python
def train():
    # 1. WandB 설정 로딩
    logger = WandbLogger(project_name="document-type-classification", config=cfg)
    config = wandb.config
    
    # 2. 동적 설정 업데이트
    cfg["BATCH_SIZE"] = config.batch_size
    cfg_optimizer["params"]["lr"] = config.head_lr
    cfg_optimizer["params"]["weight_decay"] = config.weight_decay
    
    # 3. 데이터 로더 생성
    train_loader, val_loader, train_dataset, val_dataset = setting_data_loader(cfg, data_path)
    
    # 4. 모델 인스턴스 생성
    model = setting_model(cfg["MODEL"], ModelClass, config).to(device)
    
    # 5. 차등 학습률 설정
    if cfg["use_unfreeze"]:
        params_to_update = setup_optimizer_params(
            model=model,
            model_type=cfg["model_type"], 
            num_layers_to_unfreeze=config.num_blocks_to_unfreeze,
            backbone_lr=config.backbone_lr,
            head_lr=config.head_lr,
            use_differential_lr=cfg["use_differential_lr"]
        )
    
    # 6. 학습 실행
    model, valid_max_accuracy = training_loop(...)

if __name__ == "__main__":
    sweep_configuration = get_yaml("./config/sweep-config.yaml")
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="document-type-classification")
    wandb.agent(sweep_id=sweep_id, function=train, count=10)
```

### 3.2 핵심 설계 패턴

#### 3.2.1 설정 병합 패턴
```python
# 기본 설정 로딩
cfg = load_config("config/main_config.yaml")

# WandB Sweep 파라미터로 동적 오버라이드
config = wandb.config
cfg["BATCH_SIZE"] = config.batch_size
cfg_optimizer["params"]["lr"] = config.head_lr
```

#### 3.2.2 팩토리 패턴
```python
# 모델 팩토리
def setting_model(model_name: str, model_class: nn.Module, config):
    if "ArcFace" in model_name:
        return model_class(num_classes=num_classes, s=config.arcface_s, m=config.arcface_m)
    else:
        return model_class(num_classes=num_classes)

# 데이터 로더 팩토리
def setting_data_loader(cfg: dict, data_path):
    train_transform, val_transform = build_unified_transforms(...)
    return train_loader, val_loader, train_dataset, val_dataset
```

#### 3.2.3 차등 학습률 시스템
```python
def setup_optimizer_params(model, model_type, num_layers_to_unfreeze, backbone_lr, head_lr, use_differential_lr):
    # 1. 모든 파라미터 동결
    for param in model.parameters():
        param.requires_grad = False
    
    # 2. 모델별 마지막 n개 블록 해제
    if model_type_lower.startswith('resnet'):
        all_stages = [backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4]
        stages_to_unfreeze = all_stages[-num_layers_to_unfreeze:]
    elif model_type_lower.startswith(('efficientnet', 'swin', 'convnext')):
        # 모델별 특화 처리
    
    # 3. 차등 학습률 파라미터 그룹 생성
    if use_differential_lr:
        param_groups = [
            {"params": filter(lambda p: p.requires_grad, model.backbone.parameters()), "lr": backbone_lr},
            {"params": filter(lambda p: p.requires_grad, model.head.parameters()), "lr": head_lr}
        ]
```

## 4. WandB 통합 전략

### 4.1 로깅 시스템
```python
class WandbLogger:
    def __init__(self, project_name, config, save_path):
        wandb.init(project=project_name, config=config)
        self.save_path = save_path
    
    def log_metrics(self, metrics_dict, step):
        wandb.log(metrics_dict, step=step)
    
    def save_model(self):
        wandb.save(self.save_path)
    
    def finish(self):
        wandb.finish()
```

### 4.2 메트릭 추적
- **학습 메트릭**: train/loss, train/acc, train/f1
- **검증 메트릭**: val/loss, val/acc, val/f1
- **최적화 메트릭**: valid_max_accuracy (Sweep 목표)
- **학습률 추적**: lr (스케줄러 모니터링)

## 5. 모듈화 전략

### 5.1 설정 관리 (config/config.py)
```python
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
```

### 5.2 팩토리 패턴 활용
- **모델 팩토리**: `models/__init__.py`에서 `get_model()` 함수
- **데이터셋 팩토리**: `datasets/__init__.py`에서 `get_dataset()` 함수
- **옵티마이저/스케줄러/손실함수**: `utils/*_factory.py`

### 5.3 학습 로직 분리
- **trainer/training.py**: 기본/AMP 학습 함수
- **trainer/evaluation.py**: 검증 함수
- **trainer/train_loop.py**: 전체 학습 루프
- **trainer/wandb_logger.py**: WandB 로깅 래퍼

## 6. NLP 프로젝트 적용 방안

### 6.1 직접 적용 가능한 요소

#### 6.1.1 설정 구조
```yaml
# nlp_base_config.yaml
general:
  seed: 42
  num_epochs: 20
  batch_size: 32
  
model:
  name: kobart
  checkpoint: digit82/kobart-summarization
  
training:
  learning_rate: 1.0e-05
  weight_decay: 0.01
  warmup_ratio: 0.1
  
evaluation:
  metrics: ["rouge1", "rouge2", "rougeL"]
  multi_reference: true
```

#### 6.1.2 WandB Sweep for NLP
```yaml
# nlp_sweep_config.yaml
method: bayes
metric:
  name: rouge_combined_f1
  goal: maximize

parameters:
  learning_rate:
    distribution: log_uniform_values
    min: 1.0e-6
    max: 1.0e-4
  
  batch_size:
    values: [8, 16, 32, 64]
  
  max_input_length:
    values: [256, 512, 1024]
  
  max_output_length:
    values: [64, 128, 256]
  
  num_beams:
    values: [3, 5, 8]
  
  weight_decay:
    distribution: log_uniform_values
    min: 1.0e-3
    max: 1.0e-1
```

### 6.2 NLP 특화 적응 필요 요소

#### 6.2.1 메트릭 시스템
```python
def compute_rouge_metrics(predictions, references):
    # Multi-reference ROUGE 계산
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for pred, refs in zip(predictions, references):
        # 3개 참조 요약 중 최고 점수
        best_rouge1 = max([rouge1(pred, ref) for ref in refs])
        best_rouge2 = max([rouge2(pred, ref) for ref in refs])
        best_rougeL = max([rougeL(pred, ref) for ref in refs])
        
        rouge1_scores.append(best_rouge1)
        rouge2_scores.append(best_rouge2)
        rougeL_scores.append(best_rougeL)
    
    return {
        "rouge1_f1": np.mean(rouge1_scores),
        "rouge2_f1": np.mean(rouge2_scores),
        "rougeL_f1": np.mean(rougeL_scores),
        "rouge_combined_f1": np.mean(rouge1_scores) + np.mean(rouge2_scores) + np.mean(rougeL_scores)
    }
```

#### 6.2.2 생성 모델 특화 설정
```python
def setup_generation_params(config):
    generation_params = {
        "max_length": config.max_output_length,
        "num_beams": config.num_beams,
        "no_repeat_ngram_size": config.no_repeat_ngram_size,
        "early_stopping": True,
        "length_penalty": config.length_penalty
    }
    return generation_params
```

### 6.3 기존 코드와의 호환성 보장

#### 6.3.1 점진적 마이그레이션
1. **1단계**: 기존 config.yaml 확장하여 base_config.yaml 생성
2. **2단계**: baseline.ipynb 핵심 로직을 trainer.py로 모듈화
3. **3단계**: WandB Sweep 추가하되 기존 학습 방식 병행
4. **4단계**: 완전 자동화 전환

#### 6.3.2 설정 호환성
```python
class ConfigManager:
    def load_config(self, config_path):
        # 기존 config.yaml과 새로운 구조 모두 지원
        config = yaml.safe_load(open(config_path))
        
        # 기존 구조를 새로운 구조로 자동 변환
        if "general" not in config:
            config = self.migrate_legacy_config(config)
        
        return config
    
    def migrate_legacy_config(self, legacy_config):
        # 기존 config.yaml 구조를 새로운 구조로 변환
        return migrated_config
```

## 7. 구현 우선순위 및 로드맵

### 7.1 Phase 1: 기반 구조 (1주차)
1. **설정 파일 확장**: config.yaml → base_config.yaml + sweep/
2. **모듈화 시작**: baseline.ipynb → trainer.py 분리
3. **유틸리티 구현**: config_manager.py, metrics.py

### 7.2 Phase 2: 자동화 구현 (2주차)
1. **WandB Sweep 통합**: sweep_runner.py 구현
2. **ROUGE 메트릭 시스템**: multi-reference 평가 지원
3. **실험 관리**: 결과 비교 및 분석 도구

### 7.3 Phase 3: 최적화 및 문서화 (3주차)
1. **고급 기능**: 모델 비교, A/B 테스트
2. **성능 최적화**: 병렬 실험, 리소스 관리
3. **종합 문서화**: 사용자 가이드, 베스트 프랙티스

## 8. 결론 및 기대 효과

### 8.1 6조 성공 요인 요약
1. **체계적 실험 관리**: WandB Sweep을 통한 자동화
2. **완전한 모듈화**: 재사용 가능한 컴포넌트 설계
3. **설정 중심 아키텍처**: 유연하고 확장 가능한 구조
4. **재현성 보장**: 시드 고정 및 환경 표준화

### 8.2 NLP 프로젝트 적용 시 기대 효과
1. **실험 효율성 증대**: 수동 실험 → 자동 하이퍼파라미터 튜닝
2. **성능 향상**: 체계적 탐색을 통한 최적 설정 발견
3. **팀 협업 개선**: 표준화된 실험 환경 제공
4. **재현성 확보**: 모든 실험 결과의 완전한 재현 가능

### 8.3 차별화 포인트
- **Multi-reference ROUGE**: 대회 특성에 맞는 평가 시스템
- **점진적 도입**: 기존 워크플로우 방해 최소화
- **NLP 특화**: 생성 모델, 토큰화, 길이 제약 등 고려
- **Solar API 통합**: 외부 API 기반 실험도 자동화 지원

이러한 분석을 바탕으로 NLP 요약 프로젝트에 효과적인 실험 자동화 시스템을 구축할 수 있을 것으로 기대됩니다.
