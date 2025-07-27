# NLP 대화 요약 시스템 - 종합 개발 가이드

## 목차
1. [개발 환경 개요](#개발-환경-개요)
2. [프로젝트 구조 이해](#프로젝트-구조-이해)
3. [개발 환경 설정](#개발-환경-설정)
4. [데이터 처리 워크플로우](#데이터-처리-워크플로우)
5. [모델 개발 과정](#모델-개발-과정)
6. [실험 관리 시스템](#실험-관리-시스템)
7. [성능 최적화 전략](#성능-최적화-전략)
8. [디버깅 및 트러블슈팅](#디버깅-및-트러블슈팅)
9. [코딩 표준 및 베스트 프랙티스](#코딩-표준-및-베스트-프랙티스)
10. [고급 개발 기법](#고급-개발-기법)

---

## 개발 환경 개요

이 종합 개발 가이드는 NLP 대화 요약 시스템을 효과적으로 개발하기 위한 모든 과정을 다룹니다. 초기 환경 설정부터 고급 최적화 기법까지, 개발자가 알아야 할 모든 내용을 포함합니다.

### 개발 철학
- **🔄 반복적 개발**: 빠른 프로토타이핑과 지속적인 개선
- **📊 데이터 중심**: 실험 결과와 메트릭 기반 의사결정
- **🧪 실험 추적**: 모든 실험과 결과의 체계적 관리
- **⚡ 성능 최적화**: 메모리와 속도 효율성 고려
- **🤝 팀 협업**: 명확한 코딩 표준과 문서화

---

## 프로젝트 구조 이해

### 전체 프로젝트 구조

```
nlp-sum-lyj/
├── code/                           # 핵심 소스 코드
│   ├── core/                       # 핵심 기능 모듈
│   │   ├── models/                 # 모델 아키텍처
│   │   ├── training/               # 학습 관련 코드
│   │   └── inference/              # 추론 엔진
│   ├── utils/                      # 유틸리티 함수
│   │   ├── data_utils.py          # 데이터 처리
│   │   ├── metrics.py             # 평가 메트릭
│   │   ├── path_utils.py          # 경로 관리
│   │   └── device_utils.py        # 디바이스 최적화
│   ├── config/                     # 설정 파일
│   ├── trainer.py                  # 메인 학습 스크립트
│   ├── inference.py                # 추론 실행 스크립트
│   └── experiments/                # 실험 관리 코드
├── data/                           # 데이터 파일
│   ├── train.csv                   # 학습 데이터
│   ├── dev.csv                     # 검증 데이터
│   └── test.csv                    # 테스트 데이터
├── outputs/                        # 결과 파일
│   ├── models/                     # 저장된 모델
│   ├── experiments/                # 실험 결과
│   └── submissions/                # 제출 파일
├── docs/                           # 문서
└── requirements.txt                # 의존성 명세
```

### 코드 모듈 설계 원칙

#### 1. 핵심 모듈 분리
```python
# core/models/ - 모델 정의
# core/training/ - 학습 로직
# core/inference/ - 추론 로직
# utils/ - 재사용 가능한 유틸리티
```

#### 2. 설정 기반 개발
```python
# config/base_config.yaml - 기본 설정
# config/models/ - 모델별 설정
# config/experiments/ - 실험별 설정
```

#### 3. 상대 경로 시스템
```python
# 모든 경로는 프로젝트 루트 기준 상대 경로
# 절대 경로 사용 금지 (크로스 플랫폼 호환성)
```

---

## 개발 환경 설정

### 1. 기본 환경 구성

#### Python 환경 설정
```bash
# Python 3.8+ 환경 생성
python -m venv venv

# 환경 활성화
source venv/bin/activate  # Linux/Mac
# venv\\Scripts\\activate   # Windows

# 의존성 설치
pip install -r requirements.txt

# 개발용 추가 패키지
pip install jupyter ipykernel black flake8 pytest
```

#### 핵심 의존성 확인
```python
import torch
import transformers
import pandas as pd
import numpy as np

print(f\"PyTorch: {torch.__version__}\")
print(f\"Transformers: {transformers.__version__}\")
print(f\"CUDA Available: {torch.cuda.is_available()}\")

# MPS (Apple Silicon) 확인
if hasattr(torch.backends, 'mps'):
    print(f\"MPS Available: {torch.backends.mps.is_available()}\")
```

### 2. 개발 도구 설정

#### VS Code 설정
```json
// .vscode/settings.json
{
    \"python.defaultInterpreterPath\": \"./venv/bin/python\",
    \"python.formatting.provider\": \"black\",
    \"python.linting.enabled\": true,
    \"python.linting.flake8Enabled\": true,
    \"files.exclude\": {
        \"**/__pycache__\": true,
        \"**/*.pyc\": true,
        \"outputs/experiments/\": true
    }
}
```

#### Git 설정
```bash
# .gitignore 확인
echo \"outputs/experiments/\" >> .gitignore
echo \"*.pyc\" >> .gitignore
echo \"__pycache__/\" >> .gitignore
echo \".env\" >> .gitignore

# 커밋 훅 설정 (옵션)
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/sh
# 코드 포맷팅 확인
black --check code/
flake8 code/
EOF
chmod +x .git/hooks/pre-commit
```

### 3. 디바이스별 최적화 설정

#### 자동 디바이스 감지 시스템
```python
# utils/device_utils.py (구현 예시)
import torch
import platform

def get_optimal_device():
    \"\"\"최적 디바이스 자동 감지\"\"\"
    
    # CUDA 확인
    if torch.cuda.is_available():
        return \"cuda\"
    
    # MPS (Apple Silicon) 확인
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return \"mps\"
    
    # CPU 사용
    return \"cpu\"

def get_device_config(device: str) -> dict:
    \"\"\"디바이스별 최적화 설정\"\"\"
    
    configs = {
        \"cuda\": {
            \"batch_size\": 16,
            \"fp16\": True,
            \"dataloader_pin_memory\": True,
            \"torch_dtype\": torch.float16
        },
        \"mps\": {
            \"batch_size\": 8,
            \"fp16\": False,  # MPS float16 이슈 회피
            \"dataloader_pin_memory\": False,
            \"torch_dtype\": torch.float32
        },
        \"cpu\": {
            \"batch_size\": 4,
            \"fp16\": False,
            \"dataloader_pin_memory\": False,
            \"torch_dtype\": torch.float32
        }
    }
    
    return configs.get(device, configs[\"cpu\"])

# 사용 예시
device = get_optimal_device()
config = get_device_config(device)
print(f\"감지된 디바이스: {device}\")
print(f\"권장 설정: {config}\")
```

---

## 데이터 처리 워크플로우

### 1. 다중 참조 요약 데이터 처리

#### 데이터 로딩 시스템
```python
# utils/data_utils.py (핵심 기능 설명)
class DataProcessor:
    \"\"\"다중 참조 요약 데이터 전용 프로세서\"\"\"
    
    def __init__(self, project_root=None):
        \"\"\"
        프로젝트 루트 기준 상대 경로 시스템 초기화
        \"\"\"
        # PathManager를 통한 상대 경로 관리
        self.path_manager = PathManager(project_root)
        
    def load_multi_reference_data(self, file_path: str) -> pd.DataFrame:
        \"\"\"
        다중 참조 요약 데이터 로딩
        
        지원 형식:
        1. 개별 컬럼: summary1, summary2, summary3
        2. 구분자 분리: summary 컬럼에 ||| 구분자로 분리
        3. JSON 형식: summary 컬럼에 JSON 배열
        \"\"\"
        # 상대 경로 해결
        full_path = self.path_manager.resolve_path(file_path)
        
        # 파일 존재 확인
        if not full_path.exists():
            raise FileNotFoundError(f\"데이터 파일을 찾을 수 없습니다: {file_path}\")
        
        # CSV 로딩
        df = pd.read_csv(full_path, encoding='utf-8')
        
        # Multi-reference 형식 자동 감지 및 변환
        df = self._detect_and_convert_format(df)
        
        print(f\"📊 Multi-reference 데이터 로딩 완료: {len(df)} 샘플\")
        return df
    
    def _detect_and_convert_format(self, df: pd.DataFrame) -> pd.DataFrame:
        \"\"\"다중 참조 형식 자동 감지 및 표준화\"\"\"
        
        # 형식 1: 개별 컬럼 (summary1, summary2, summary3)
        if all(col in df.columns for col in ['summary1', 'summary2', 'summary3']):
            df['summaries'] = df[['summary1', 'summary2', 'summary3']].apply(
                lambda x: [str(val) if pd.notna(val) else \"\" for val in x], axis=1
            )
            print(\"✅ 개별 컬럼 형식의 multi-reference 데이터 감지\")
            
        # 형식 2: 구분자 분리 (summary 컬럼에 ||| 구분자)
        elif 'summary' in df.columns:
            df['summaries'] = df['summary'].apply(self._parse_multiple_summaries)
            print(\"✅ 구분자 분리 형식의 multi-reference 데이터 감지\")
            
        else:
            raise ValueError(
                \"지원되는 summary 형식이 없습니다. \"
                \"summary1,summary2,summary3 컬럼 또는 summary 컬럼이 필요합니다.\"
            )
        
        return df
    
    def _parse_multiple_summaries(self, summary_text: str) -> list:
        \"\"\"구분자로 분리된 요약문 파싱\"\"\"
        if pd.isna(summary_text):
            return [\"\", \"\", \"\"]
        
        # 다양한 구분자 지원
        separators = ['|||', '##', '---', '\
\
']
        
        for sep in separators:
            if sep in summary_text:
                summaries = [s.strip() for s in summary_text.split(sep)]
                # 3개로 패딩 또는 자르기
                while len(summaries) < 3:
                    summaries.append(\"\")
                return summaries[:3]
        
        # 구분자가 없으면 단일 요약문으로 처리
        return [summary_text.strip(), \"\", \"\"]
```

#### 제출 형식 변환
```python
def export_submission_format(self, 
                            predictions: List[str],
                            fnames: List[str],
                            output_path: str) -> pd.DataFrame:
    \"\"\"
    대회 제출 형식으로 저장
    
    Args:
        predictions: 예측된 요약문 리스트
        fnames: 파일명 리스트
        output_path: 출력 파일 경로 (상대 경로)
    
    Returns:
        submission_df: 제출 형식 데이터프레임
    \"\"\"
    if len(predictions) != len(fnames):
        raise ValueError(f\"예측과 파일명 개수 불일치: {len(predictions)} vs {len(fnames)}\")
    
    # 제출 형식 데이터프레임 생성
    submission_df = pd.DataFrame({
        'fname': fnames,
        'summary': predictions
    })
    
    # 상대 경로 확인 및 해결
    output_path = Path(output_path)
    if output_path.is_absolute():
        raise ValueError(f\"출력 경로는 상대 경로여야 합니다: {output_path}\")
    
    full_output_path = self.path_manager.resolve_path(output_path)
    
    # 출력 디렉토리 생성
    full_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # CSV 저장 (대회 제출 형식)
    submission_df.to_csv(full_output_path, index=False, encoding='utf-8')
    
    print(f\"💾 제출 파일 저장: {output_path}\")
    print(f\"📋 형식: fname, summary ({len(submission_df)} 항목)\")
    
    return submission_df
```

---

## 모델 개발 과정

### 1. 모델 아키텍처 설계

#### 지원 모델 아키텍처
```python
# core/models/model_factory.py (구조 설명)
class ModelFactory:
    \"\"\"모델 팩토리 클래스\"\"\"
    
    SUPPORTED_MODELS = {
        \"kobart\": {
            \"base_model\": \"gogamza/kobart-base-v2\",
            \"tokenizer_class\": \"BartTokenizer\",
            \"model_class\": \"BartForConditionalGeneration\",
            \"max_position_embeddings\": 1024
        },
        \"kogpt2\": {
            \"base_model\": \"skt/kogpt2-base-v2\", 
            \"tokenizer_class\": \"GPT2Tokenizer\",
            \"model_class\": \"GPT2LMHeadModel\",
            \"max_position_embeddings\": 1024
        },
        \"kt5\": {
            \"base_model\": \"KETI-AIR/ke-t5-base\",
            \"tokenizer_class\": \"T5Tokenizer\", 
            \"model_class\": \"T5ForConditionalGeneration\",
            \"max_position_embeddings\": 512
        },
        \"mt5\": {
            \"base_model\": \"google/mt5-base\",
            \"tokenizer_class\": \"T5Tokenizer\",
            \"model_class\": \"MT5ForConditionalGeneration\", 
            \"max_position_embeddings\": 1024
        }
    }
    
    @classmethod
    def create_model_and_tokenizer(cls, model_name: str, device: str):
        \"\"\"모델과 토크나이저 생성\"\"\"
        
        if model_name not in cls.SUPPORTED_MODELS:
            raise ValueError(f\"지원되지 않는 모델: {model_name}\")
        
        config = cls.SUPPORTED_MODELS[model_name]
        
        # 토크나이저 로딩
        tokenizer = getattr(transformers, config[\"tokenizer_class\"]).from_pretrained(
            config[\"base_model\"]
        )
        
        # 모델 로딩 및 디바이스 설정
        model = getattr(transformers, config[\"model_class\"]).from_pretrained(
            config[\"base_model\"]
        )
        
        # 디바이스별 최적화
        device_config = get_device_config(device)
        if device_config.get(\"torch_dtype\"):
            model = model.to(dtype=device_config[\"torch_dtype\"])
        
        model = model.to(device)
        
        return model, tokenizer, config
```

#### 커스텀 모델 구성
```python
# core/models/custom_models.py (설계 가이드)
class DialogueSummarizationModel:
    \"\"\"대화 요약 전용 모델 래퍼\"\"\"
    
    def __init__(self, model_name: str, device: str = \"auto\"):
        \"\"\"
        대화 요약에 최적화된 모델 초기화
        \"\"\"
        if device == \"auto\":
            device = get_optimal_device()
        
        self.device = device
        self.model, self.tokenizer, self.config = ModelFactory.create_model_and_tokenizer(
            model_name, device
        )
        
        # 대화 요약 특화 설정
        self._setup_generation_config()
        self._setup_special_tokens()
    
    def _setup_generation_config(self):
        \"\"\"생성 설정 최적화\"\"\"
        self.generation_config = {
            \"max_length\": 128,
            \"min_length\": 10,
            \"num_beams\": 4,
            \"length_penalty\": 1.0,
            \"no_repeat_ngram_size\": 3,
            \"early_stopping\": True,
            \"do_sample\": False
        }
    
    def _setup_special_tokens(self):
        \"\"\"특수 토큰 설정\"\"\"
        # 대화 구분자 추가 (필요시)
        special_tokens = [\"<speaker1>\", \"<speaker2>\", \"<turn>\"]
        
        if hasattr(self.tokenizer, 'add_special_tokens'):
            self.tokenizer.add_special_tokens({
                \"additional_special_tokens\": special_tokens
            })
            self.model.resize_token_embeddings(len(self.tokenizer))
    
    def preprocess_dialogue(self, dialogue: str) -> str:
        \"\"\"대화 전처리\"\"\"
        
        # 기본 정리
        dialogue = dialogue.strip()
        
        # 화자 구분 정규화 (옵션)
        dialogue = re.sub(r'화자\\s*(\\d+)\\s*:', r'<speaker\\1>:', dialogue)
        
        # 과도한 공백 정리
        dialogue = re.sub(r'\\s+', ' ', dialogue)
        
        return dialogue
    
    def generate_summary(self, dialogue: str, **generation_kwargs) -> str:
        \"\"\"요약 생성\"\"\"
        
        # 전처리
        processed_dialogue = self.preprocess_dialogue(dialogue)
        
        # 토크나이징
        inputs = self.tokenizer(
            processed_dialogue,
            max_length=self.config[\"max_position_embeddings\"],
            truncation=True,
            padding=True,
            return_tensors=\"pt\"
        ).to(self.device)
        
        # 생성 설정 병합
        final_config = {**self.generation_config, **generation_kwargs}
        
        # 요약 생성
        with torch.no_grad():
            if self.device == \"mps\":
                # MPS 최적화
                with torch.autocast(device_type=\"cpu\", enabled=False):
                    outputs = self.model.generate(**inputs, **final_config)
            else:
                outputs = self.model.generate(**inputs, **final_config)
        
        # 디코딩
        summary = self.tokenizer.decode(
            outputs[0], 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        return summary.strip()
```

### 2. 학습 과정 최적화

#### 학습 설정 관리
```python
# core/training/trainer_config.py (설정 예시)
class TrainingConfig:
    \"\"\"학습 설정 관리 클래스\"\"\"
    
    def __init__(self, device: str = \"auto\"):
        if device == \"auto\":
            device = get_optimal_device()
        
        self.device = device
        self.device_config = get_device_config(device)
        
        # 기본 학습 설정
        self.base_config = {
            \"learning_rate\": 5e-5,
            \"num_train_epochs\": 3,
            \"warmup_ratio\": 0.1,
            \"weight_decay\": 0.01,
            \"logging_steps\": 100,
            \"save_steps\": 500,
            \"eval_steps\": 500,
            \"save_total_limit\": 3,
            \"load_best_model_at_end\": True,
            \"metric_for_best_model\": \"rouge_combined_f1\",
            \"greater_is_better\": True,
        }
        
        # 디바이스별 최적화 병합
        self.training_args = {**self.base_config, **self.device_config}
    
    def get_training_arguments(self, output_dir: str):
        \"\"\"TrainingArguments 객체 생성\"\"\"
        from transformers import TrainingArguments
        
        return TrainingArguments(
            output_dir=output_dir,
            **self.training_args
        )
    
    def update_config(self, **kwargs):
        \"\"\"설정 업데이트\"\"\"
        self.training_args.update(kwargs)
```

#### 데이터셋 클래스
```python
# core/training/dataset.py (구조 가이드)
class DialogueSummarizationDataset(torch.utils.data.Dataset):
    \"\"\"대화 요약 데이터셋 클래스\"\"\"
    
    def __init__(self, 
                 dialogues: List[str], 
                 summaries: List[str],
                 tokenizer,
                 max_input_length: int = 512,
                 max_target_length: int = 128):
        
        self.dialogues = dialogues
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
    
    def __len__(self):
        return len(self.dialogues)
    
    def __getitem__(self, idx):
        dialogue = self.dialogues[idx]
        summary = self.summaries[idx]
        
        # 입력 토크나이징
        model_inputs = self.tokenizer(
            dialogue,
            max_length=self.max_input_length,
            truncation=True,
            padding=\"max_length\",
            return_tensors=\"pt\"
        )
        
        # 타겟 토크나이징
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                summary,
                max_length=self.max_target_length,
                truncation=True,
                padding=\"max_length\",
                return_tensors=\"pt\"
            )
        
        model_inputs[\"labels\"] = labels[\"input_ids\"]
        
        # 배치 차원 제거
        return {k: v.squeeze() for k, v in model_inputs.items()}
```

---

## 실험 관리 시스템

### 1. 실험 추적 도구

#### 실험 메타데이터 관리
```python
# utils/experiment_utils.py (핵심 기능)
class ExperimentTracker:
    \"\"\"실험 추적 및 관리 클래스\"\"\"
    
    def __init__(self, experiments_dir: str = \"outputs/experiments\"):
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_experiment = None
        self.experiment_data = {}
    
    def start_experiment(self, 
                        name: str, 
                        description: str,
                        config: dict) -> str:
        \"\"\"새 실험 시작\"\"\"
        
        experiment_id = self._generate_experiment_id()
        timestamp = datetime.now().isoformat()
        
        self.experiment_data = {
            \"id\": experiment_id,
            \"name\": name,
            \"description\": description,
            \"config\": config,
            \"start_time\": timestamp,
            \"status\": \"running\",
            \"device\": get_optimal_device(),
            \"metrics\": [],
            \"checkpoints\": [],
            \"final_results\": None
        }
        
        self.current_experiment = experiment_id
        self._save_experiment_data()
        
        print(f\"🧪 실험 시작: {name} (ID: {experiment_id[:8]})\")
        return experiment_id
    
    def log_metrics(self, metrics: dict, step: int = None):
        \"\"\"메트릭 로깅\"\"\"
        if not self.current_experiment:
            raise ValueError(\"활성 실험이 없습니다\")
        
        metric_entry = {
            \"step\": step or len(self.experiment_data[\"metrics\"]),
            \"timestamp\": datetime.now().isoformat(),
            **metrics
        }
        
        self.experiment_data[\"metrics\"].append(metric_entry)
        self._save_experiment_data()
    
    def log_checkpoint(self, checkpoint_path: str, metrics: dict):
        \"\"\"체크포인트 로깅\"\"\"
        checkpoint_entry = {
            \"path\": checkpoint_path,
            \"timestamp\": datetime.now().isoformat(),
            \"metrics\": metrics
        }
        
        self.experiment_data[\"checkpoints\"].append(checkpoint_entry)
        self._save_experiment_data()
    
    def end_experiment(self, 
                      final_metrics: dict, 
                      status: str = \"completed\"):
        \"\"\"실험 종료\"\"\"
        if not self.current_experiment:
            raise ValueError(\"활성 실험이 없습니다\")
        
        self.experiment_data.update({
            \"end_time\": datetime.now().isoformat(),
            \"status\": status,
            \"final_results\": final_metrics
        })
        
        self._save_experiment_data()
        
        print(f\"🏁 실험 완료: {self.experiment_data['name']}\")
        print(f\"📊 최종 성능: {final_metrics}\")
        
        self.current_experiment = None
    
    def _generate_experiment_id(self) -> str:
        \"\"\"실험 ID 생성\"\"\"
        import uuid
        return str(uuid.uuid4())
    
    def _save_experiment_data(self):
        \"\"\"실험 데이터 저장\"\"\"
        if not self.current_experiment:
            return
        
        file_path = self.experiments_dir / f\"{self.current_experiment}.json\"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.experiment_data, f, indent=2, ensure_ascii=False)
    
    def load_experiment(self, experiment_id: str) -> dict:
        \"\"\"실험 데이터 로딩\"\"\"
        file_path = self.experiments_dir / f\"{experiment_id}.json\"
        
        if not file_path.exists():
            raise FileNotFoundError(f\"실험을 찾을 수 없습니다: {experiment_id}\")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_experiment_summary(self) -> pd.DataFrame:
        \"\"\"모든 실험 요약 조회\"\"\"
        experiments = []
        
        for exp_file in self.experiments_dir.glob(\"*.json\"):
            try:
                with open(exp_file, 'r', encoding='utf-8') as f:
                    exp_data = json.load(f)
                
                # 최고 성능 메트릭 추출
                best_rouge = 0
                if exp_data.get('metrics'):
                    rouge_scores = [m.get('rouge_combined_f1', 0) for m in exp_data['metrics']]
                    best_rouge = max(rouge_scores) if rouge_scores else 0
                
                experiments.append({
                    'id': exp_data['id'][:8],
                    'name': exp_data['name'],
                    'status': exp_data['status'],
                    'device': exp_data.get('device', 'unknown'),
                    'start_time': exp_data['start_time'],
                    'best_rouge_combined_f1': best_rouge
                })
                
            except Exception as e:
                print(f\"⚠️ 실험 파일 로딩 실패 {exp_file}: {e}\")
        
        df = pd.DataFrame(experiments)
        return df.sort_values('best_rouge_combined_f1', ascending=False)
```

### 2. 모델 레지스트리

#### 모델 버전 관리
```python
# utils/model_registry.py (구조 가이드)
class ModelRegistry:
    \"\"\"모델 버전 및 성능 관리\"\"\"
    
    def __init__(self, registry_path: str = \"outputs/model_registry.json\"):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.models = self._load_registry()
    
    def register_model(self,
                      name: str,
                      architecture: str,
                      config: dict,
                      performance: dict,
                      model_path: str = None,
                      experiment_id: str = None) -> str:
        \"\"\"모델 등록\"\"\"
        
        model_id = self._generate_model_id()
        
        model_info = {
            \"id\": model_id,
            \"name\": name,
            \"architecture\": architecture,
            \"config\": config,
            \"performance\": performance,
            \"model_path\": model_path,
            \"experiment_id\": experiment_id,
            \"created_at\": datetime.now().isoformat(),
            \"tags\": []
        }
        
        self.models[model_id] = model_info
        self._save_registry()
        
        print(f\"📝 모델 등록: {name} (ID: {model_id[:8]})\")
        return model_id
    
    def get_best_model(self, 
                      architecture: str = None,
                      metric: str = \"rouge_combined_f1\") -> dict:
        \"\"\"최고 성능 모델 조회\"\"\"
        
        filtered_models = self.models.values()
        
        if architecture:
            filtered_models = [
                m for m in filtered_models 
                if m[\"architecture\"] == architecture
            ]
        
        if not filtered_models:
            return None
        
        # 성능 기준 정렬
        best_model = max(
            filtered_models,
            key=lambda m: m[\"performance\"].get(metric, 0)
        )
        
        return best_model
    
    def get_models_summary(self) -> pd.DataFrame:
        \"\"\"모델 요약 테이블\"\"\"
        
        if not self.models:
            return pd.DataFrame()
        
        model_data = []
        for model_info in self.models.values():
            model_data.append({
                'id': model_info['id'][:8],
                'name': model_info['name'],
                'architecture': model_info['architecture'],
                'rouge_combined_f1': model_info['performance'].get('rouge_combined_f1', 0),
                'created_at': model_info['created_at'][:10]  # 날짜만
            })
        
        df = pd.DataFrame(model_data)
        return df.sort_values('rouge_combined_f1', ascending=False)
```

---

## 성능 최적화 전략

### 1. 메모리 최적화

#### 배치 크기 동적 조정
```python
# utils/optimization.py (최적화 가이드)
class AdaptiveBatchProcessor:
    \"\"\"동적 배치 크기 조정\"\"\"
    
    def __init__(self, device: str, initial_batch_size: int = 8):
        self.device = device
        self.current_batch_size = initial_batch_size
        self.max_batch_size = self._get_max_batch_size()
        self.min_batch_size = 1
        
        # 성능 추적
        self.success_count = 0
        self.oom_count = 0
    
    def _get_max_batch_size(self) -> int:
        \"\"\"디바이스별 최대 배치 크기\"\"\"
        if self.device == \"mps\":
            return 8  # MPS 메모리 제약
        elif self.device == \"cuda\":
            # GPU 메모리 기반 추정
            try:
                total_memory = torch.cuda.get_device_properties(0).total_memory
                memory_gb = total_memory / (1024**3)
                return min(32, int(memory_gb * 2))
            except:
                return 16
        else:
            return 4  # CPU 제약
    
    def process_batch(self, data_loader, process_fn):
        \"\"\"적응형 배치 처리\"\"\"
        
        results = []
        
        for batch in data_loader:
            try:
                # 배치 처리 시도
                batch_results = process_fn(batch)
                results.extend(batch_results)
                
                self.success_count += 1
                
                # 성공 시 배치 크기 증가 고려
                if (self.success_count % 10 == 0 and 
                    self.current_batch_size < self.max_batch_size):
                    self.current_batch_size = min(
                        self.current_batch_size + 1, 
                        self.max_batch_size
                    )
                    print(f\"📈 배치 크기 증가: {self.current_batch_size}\")
                
            except torch.cuda.OutOfMemoryError:
                # OOM 시 배치 크기 감소
                self.oom_count += 1
                old_size = self.current_batch_size
                self.current_batch_size = max(
                    self.current_batch_size // 2, 
                    self.min_batch_size
                )
                
                print(f\"⚠️ GPU 메모리 부족, 배치 크기 {old_size} → {self.current_batch_size}\")
                
                # 메모리 정리
                if self.device == \"cuda\":
                    torch.cuda.empty_cache()
                
                # 줄어든 배치로 재시도
                continue
        
        return results
```

#### 메모리 모니터링
```python
def monitor_memory_usage(func):
    \"\"\"메모리 사용량 모니터링 데코레이터\"\"\"
    
    def wrapper(*args, **kwargs):
        import psutil
        import gc
        
        # 시작 메모리
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        if torch.cuda.is_available():
            gpu_memory_before = torch.cuda.memory_allocated() / 1024 / 1024
        else:
            gpu_memory_before = 0
        
        try:
            # 함수 실행
            result = func(*args, **kwargs)
            
            # 종료 메모리
            memory_after = process.memory_info().rss / 1024 / 1024
            memory_diff = memory_after - memory_before
            
            if torch.cuda.is_available():
                gpu_memory_after = torch.cuda.memory_allocated() / 1024 / 1024
                gpu_memory_diff = gpu_memory_after - gpu_memory_before
            else:
                gpu_memory_diff = 0
            
            print(f\"💾 {func.__name__} 메모리 사용량:\")
            print(f\"  System: {memory_diff:+.1f} MB\")
            if gpu_memory_diff:
                print(f\"  GPU: {gpu_memory_diff:+.1f} MB\")
            
            return result
            
        finally:
            # 가비지 컬렉션
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return wrapper
```

### 2. 학습 속도 최적화

#### 데이터 로딩 최적화
```python
# utils/data_loading.py (최적화 가이드)
def create_optimized_dataloader(dataset, 
                               batch_size: int,
                               device: str,
                               num_workers: int = None) -> DataLoader:
    \"\"\"최적화된 데이터로더 생성\"\"\"
    
    # 디바이스별 워커 수 자동 설정
    if num_workers is None:
        if device == \"cuda\":
            num_workers = min(8, os.cpu_count())
        elif device == \"mps\":
            num_workers = 4  # MPS는 제한적
        else:
            num_workers = 2  # CPU
    
    # 디바이스별 설정
    device_config = get_device_config(device)
    pin_memory = device_config.get(\"dataloader_pin_memory\", False)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else 2
    )
```

---

## 디버깅 및 트러블슈팅

### 1. 일반적인 문제 해결

#### 디바이스 관련 문제
```python
# utils/debug.py (디버깅 도구)
def diagnose_device_issues():
    \"\"\"디바이스 관련 문제 진단\"\"\"
    
    print(\"🔍 디바이스 진단 시작\")
    print(\"=\" * 50)
    
    # 플랫폼 정보
    import platform
    print(f\"플랫폼: {platform.system()} {platform.machine()}\")
    print(f\"Python: {platform.python_version()}\")
    
    # PyTorch 정보
    print(f\"PyTorch: {torch.__version__}\")
    
    # CUDA 진단
    print(f\"\
🖥️ CUDA 정보:\")
    print(f\"  사용 가능: {torch.cuda.is_available()}\")
    if torch.cuda.is_available():
        print(f\"  버전: {torch.version.cuda}\")
        print(f\"  디바이스 수: {torch.cuda.device_count()}\")
        print(f\"  현재 디바이스: {torch.cuda.current_device()}\")
        print(f\"  GPU 이름: {torch.cuda.get_device_name()}\")
        
        # 메모리 정보
        memory = torch.cuda.get_device_properties(0).total_memory
        print(f\"  총 메모리: {memory / 1024**3:.1f} GB\")
    
    # MPS 진단 (Apple Silicon)
    if hasattr(torch.backends, 'mps'):
        print(f\"\
🍎 MPS 정보:\")
        print(f\"  사용 가능: {torch.backends.mps.is_available()}\")
        print(f\"  Built: {torch.backends.mps.is_built()}\")
    
    # 권장 설정
    optimal_device = get_optimal_device()
    config = get_device_config(optimal_device)
    
    print(f\"\
✅ 권장 설정:\")
    print(f\"  디바이스: {optimal_device}\")
    print(f\"  배치 크기: {config['batch_size']}\")
    print(f\"  FP16: {config['fp16']}\")
    print(f\"  Pin Memory: {config['dataloader_pin_memory']}\")

def test_model_loading(model_name: str = \"kobart\"):
    \"\"\"모델 로딩 테스트\"\"\"
    
    print(f\"🧪 모델 로딩 테스트: {model_name}\")
    
    try:
        device = get_optimal_device()
        model, tokenizer, config = ModelFactory.create_model_and_tokenizer(
            model_name, device
        )
        
        print(\"✅ 모델 로딩 성공\")
        
        # 간단한 추론 테스트
        test_input = \"화자1: 안녕하세요\
화자2: 안녕하세요\"
        inputs = tokenizer(test_input, return_tensors=\"pt\").to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=50)
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f\"✅ 추론 테스트 성공: {result[:50]}...\")
        
    except Exception as e:
        print(f\"❌ 테스트 실패: {e}\")
        import traceback
        traceback.print_exc()
```

#### 데이터 관련 문제
```python
def validate_data_pipeline(data_path: str = \"data/train.csv\"):
    \"\"\"데이터 파이프라인 검증\"\"\"
    
    print(f\"📊 데이터 파이프라인 검증: {data_path}\")
    
    try:
        # 데이터 로딩 테스트
        processor = DataProcessor()
        df = processor.load_multi_reference_data(data_path)
        
        print(f\"✅ 데이터 로딩 성공: {len(df)} 샘플\")
        
        # 기본 통계
        print(f\"📋 데이터 정보:\")
        print(f\"  컬럼: {list(df.columns)}\")
        print(f\"  대화 평균 길이: {df['dialogue'].str.len().mean():.1f}\")
        
        if 'summaries' in df.columns:
            # Multi-reference 요약문 체크
            sample_summaries = df['summaries'].iloc[0]
            print(f\"  요약문 개수: {len(sample_summaries)}\")
            print(f\"  첫 번째 요약: {sample_summaries[0][:50]}...\")
        
        # 데이터 품질 검증
        validator = DataValidator()
        validation_results = validator.validate_dataset(df)
        validator.print_validation_report(validation_results)
        
    except Exception as e:
        print(f\"❌ 데이터 검증 실패: {e}\")
        import traceback
        traceback.print_exc()
```

### 2. 성능 문제 분석

#### 프로파일링 도구
```python
def profile_training_step(trainer, data_loader, num_steps: int = 10):
    \"\"\"학습 단계 프로파일링\"\"\"
    
    import time
    import cProfile
    import pstats
    
    print(f\"📊 학습 단계 프로파일링 ({num_steps} 스텝)\")
    
    # 프로파일러 설정
    profiler = cProfile.Profile()
    
    # 시작 시간
    start_time = time.time()
    
    profiler.enable()
    
    try:
        # 지정된 스텝 수만큼 학습
        for i, batch in enumerate(data_loader):
            if i >= num_steps:
                break
            
            # 한 스텝 실행
            trainer.training_step(trainer.model, batch)
            
            if i % 5 == 0:
                print(f\"  스텝 {i+1}/{num_steps} 완료\")
    
    finally:
        profiler.disable()
    
    # 결과 분석
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f\"⏱️ 총 시간: {total_time:.2f}초\")
    print(f\"🚀 스텝당 평균: {total_time/num_steps:.2f}초\")
    
    # 프로파일 결과 출력
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # 상위 10개 함수
```

---

## 코딩 표준 및 베스트 프랙티스

### 1. 코드 스타일 가이드

#### 함수 작성 표준
```python
# ✅ 권장 패턴
def process_dialogue_data(input_path: Union[str, Path], 
                         output_path: Union[str, Path],
                         batch_size: int = 16,
                         device: str = \"auto\") -> int:
    \"\"\"
    대화 데이터 처리 함수
    
    Args:
        input_path: 입력 파일 경로 (상대 경로)
        output_path: 출력 파일 경로 (상대 경로) 
        batch_size: 배치 크기
        device: 처리 디바이스 (\"auto\", \"cuda\", \"mps\", \"cpu\")
    
    Returns:
        int: 처리된 샘플 수
        
    Raises:
        ValueError: 경로가 절대 경로인 경우
        FileNotFoundError: 입력 파일이 없는 경우
        
    Example:
        >>> count = process_dialogue_data(\"data/train.csv\", \"outputs/processed.csv\")
        >>> print(f\"처리된 샘플: {count}\")
    \"\"\"
    # 1. 입력 검증
    if isinstance(input_path, str):
        input_path = Path(input_path)
    if isinstance(output_path, str):
        output_path = Path(output_path)
    
    # 2. 절대 경로 금지
    if input_path.is_absolute() or output_path.is_absolute():
        raise ValueError(\"상대 경로를 사용하세요\")
    
    # core/ensemble/ensemble_model.py (앙상블 가이드)
    class EnsemblePredictor:
        """다중 모델 앙상블 예측기"""
        
        def __init__(self, model_configs: List[dict]):
            """
            Args:
                model_configs: 앙상블할 모델 설정 리스트
                              [{'path': 'model1_path', 'weight': 0.4}, 
                               {'path': 'model2_path', 'weight': 0.6}]
            """
            self.models = []
            self.weights = []
            
            for config in model_configs:
                # 모델 로딩
                model = DialogueSummarizationModel.load_from_checkpoint(config['path'])
                weight = config.get('weight', 1.0)
                
                self.models.append(model)
                self.weights.append(weight)
            
            # 가중치 정규화
            total_weight = sum(self.weights)
            self.weights = [w / total_weight for w in self.weights]
            
            print(f"🔗 앙상블 모델 로딩 완료: {len(self.models)}개 모델")
        
        def predict_ensemble(self, dialogue: str, strategy: str = "weighted_vote") -> str:
            """앙상블 예측"""
            
            # 각 모델에서 예측 생성
            predictions = []
            for model in self.models:
                pred = model.generate_summary(dialogue)
                predictions.append(pred)
            
            # 앙상블 전략에 따른 결합
            if strategy == "weighted_vote":
                # 가중 평균 기반 (단순화)
                return self._weighted_average(predictions)
            elif strategy == "longest":
                # 가장 긴 예측 선택
                return max(predictions, key=len)
            elif strategy == "shortest":
                # 가장 짧은 예측 선택
                return min(predictions, key=len)
            else:
                # 첫 번째 모델 결과
                return predictions[0]
        
        def _weighted_average(self, predictions: List[str]) -> str:
            """가중 평균 기반 앙상블 (단순화)"""
            
            # 실제 구현에서는 토큰 레벨에서 확률 결합
            # 여기서는 가장 높은 가중치 모델의 예측 사용
            best_idx = self.weights.index(max(self.weights))
            return predictions[best_idx]
        
        def predict_batch_ensemble(self, dialogues: List[str]) -> List[str]:
            """배치 앙상블 예측"""
            
            results = []
            for dialogue in dialogues:
                ensemble_pred = self.predict_ensemble(dialogue)
                results.append(ensemble_pred)
            
            return results
    
    # 사용 예시
    model_configs = [
        {'path': 'outputs/kobart_model', 'weight': 0.4},
        {'path': 'outputs/kt5_model', 'weight': 0.6}
    ]
    
    ensemble = EnsemblePredictor(model_configs)
    prediction = ensemble.predict_ensemble("화자1: 안녕하세요\n화자2: 안녕하세요")
    ```
    
    ### 4. 고급 데이터 증강
    
    #### 대화 데이터 증강 기법
    ```python
    # utils/data_augmentation.py (데이터 증강)
    class DialogueAugmenter:
        """대화 데이터 증강 클래스"""
        
        def __init__(self, augmentation_ratio: float = 0.2):
            """
            Args:
                augmentation_ratio: 증강할 데이터 비율
            """
            self.augmentation_ratio = augmentation_ratio
            
            # 동의어 사전 (간단한 예시)
            self.synonyms = {
                '안녕하세요': ['안녕', '반갑습니다', '안녕하십니까'],
                '감사합니다': ['고맙습니다', '고마워요', '감사해요'],
                '네': ['예', '그렇습니다', '맞습니다'],
                '아니요': ['아닙니다', '아니에요', '그렇지 않습니다']
            }
        
        def augment_dataset(self, dialogues: List[str], summaries: List[str]) -> Tuple[List[str], List[str]]:
            """데이터셋 증강"""
            
            augmented_dialogues = dialogues.copy()
            augmented_summaries = summaries.copy()
            
            num_to_augment = int(len(dialogues) * self.augmentation_ratio)
            
            # 랜덤하게 선택하여 증강
            import random
            indices = random.sample(range(len(dialogues)), num_to_augment)
            
            for idx in indices:
                original_dialogue = dialogues[idx]
                original_summary = summaries[idx]
                
                # 다양한 증강 기법 적용
                aug_techniques = [
                    self._synonym_replacement,
                    self._speaker_permutation,
                    self._sentence_reordering
                ]
                
                for technique in aug_techniques:
                    try:
                        aug_dialogue = technique(original_dialogue)
                        if aug_dialogue != original_dialogue:
                            augmented_dialogues.append(aug_dialogue)
                            augmented_summaries.append(original_summary)
                    except Exception as e:
                        print(f"⚠️ 증강 실패: {e}")
                        continue
            
            print(f"🔄 데이터 증강 완료: {len(dialogues)} → {len(augmented_dialogues)}")
            return augmented_dialogues, augmented_summaries
        
        def _synonym_replacement(self, dialogue: str) -> str:
            """동의어 치환"""
            
            augmented = dialogue
            
            for original, synonyms in self.synonyms.items():
                if original in augmented:
                    import random
                    synonym = random.choice(synonyms)
                    augmented = augmented.replace(original, synonym, 1)  # 첫 번째만 치환
            
            return augmented
        
        def _speaker_permutation(self, dialogue: str) -> str:
            """화자 순서 변경 (2명 대화인 경우)"""
            
            lines = dialogue.split('\n')
            
            # 화자1과 화자2 교체
            augmented_lines = []
            for line in lines:
                if line.startswith('화자1:'):
                    augmented_lines.append(line.replace('화자1:', '화자2:'))
                elif line.startswith('화자2:'):
                    augmented_lines.append(line.replace('화자2:', '화자1:'))
                else:
                    augmented_lines.append(line)
            
            return '\n'.join(augmented_lines)
        
        def _sentence_reordering(self, dialogue: str) -> str:
            """문장 순서 재배치 (주의: 의미 변경 가능)"""
            
            lines = dialogue.split('\n')
            
            if len(lines) <= 2:
                return dialogue  # 너무 짧으면 그대로
            
            # 인접한 문장 2개씩 순서 바꾸기
            import random
            if len(lines) >= 4 and random.random() < 0.5:
                # 중간 2개 문장 순서 바꾸기
                mid = len(lines) // 2
                if mid > 0 and mid < len(lines) - 1:
                    lines[mid], lines[mid + 1] = lines[mid + 1], lines[mid]
            
            return '\n'.join(lines)
    ```
    
    ### 5. 모델 해석 및 분석
    
    #### 어텐션 시각화
    ```python
    # utils/model_interpretation.py (모델 해석)
    class AttentionVisualizer:
        """어텐션 가중치 시각화"""
        
        def __init__(self, model, tokenizer):
            self.model = model
            self.tokenizer = tokenizer
        
        def visualize_attention(self, dialogue: str, summary: str, save_path: str = None):
            """어텐션 가중치 시각화"""
            
            # 토크나이징
            inputs = self.tokenizer(dialogue, return_tensors="pt")
            
            # 모델 실행 (어텐션 가중치 포함)
            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True)
            
            # 어텐션 가중치 추출
            attention_weights = outputs.attentions[-1]  # 마지막 레이어
            
            # 시각화
            self._plot_attention_heatmap(
                attention_weights[0].mean(dim=0),  # 헤드 평균
                self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]),
                save_path
            )
        
        def _plot_attention_heatmap(self, attention_matrix, tokens, save_path):
            """어텐션 히트맵 생성"""
            
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.figure(figsize=(12, 8))
            
            # 히트맵 생성
            sns.heatmap(
                attention_matrix.cpu().numpy(),
                xticklabels=tokens,
                yticklabels=tokens,
                cmap='Blues',
                cbar=True
            )
            
            plt.title('Attention Weights Heatmap')
            plt.xlabel('Input Tokens')
            plt.ylabel('Output Tokens')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📊 어텐션 시각화 저장: {save_path}")
            
            plt.show()
        
        def analyze_token_importance(self, dialogue: str) -> dict:
            """토큰 중요도 분석"""
            
            inputs = self.tokenizer(dialogue, return_tensors="pt")
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True)
            
            # 모든 레이어의 어텐션 평균
            all_attention = torch.stack(outputs.attentions)
            avg_attention = all_attention.mean(dim=(0, 2))  # 레이어, 헤드 평균
            
            # 각 토큰의 중요도 (받은 어텐션의 합)
            token_importance = avg_attention.sum(dim=0).cpu().numpy()
            
            # 토큰-중요도 매핑
            importance_dict = {
                token: float(importance) 
                for token, importance in zip(tokens, token_importance)
            }
            
            # 중요도 순 정렬
            sorted_importance = sorted(
                importance_dict.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            return {
                'token_importance': importance_dict,
                'top_tokens': sorted_importance[:10],
                'avg_importance': float(token_importance.mean())
            }
    ```
    
    ---
    
    ## 실전 개발 워크플로우
    
    ### 1. 프로젝트 시작 체크리스트
    
    #### 초기 설정 (30분)
    ```bash
    # 1. 환경 확인
    python --version  # 3.8+
    git --version
    
    # 2. 가상환경 생성
    python -m venv venv
    source venv/bin/activate
    
    # 3. 의존성 설치
    pip install -r requirements.txt
    
    # 4. 디바이스 확인
    python -c "from utils.device_utils import get_optimal_device; print(get_optimal_device())"
    
    # 5. 데이터 검증
    python validate_data.py
    ```
    
    #### 개발 환경 검증
    ```python
    # quick_test.py
    if __name__ == "__main__":
        # 핵심 모듈 임포트 테스트
        try:
            from utils.data_utils import DataProcessor
            from utils.metrics import RougeCalculator
            from utils.device_utils import get_optimal_device
            
            print("✅ 모든 모듈 임포트 성공")
            
            # 디바이스 감지
            device = get_optimal_device()
            print(f"✅ 감지된 디바이스: {device}")
            
            # 간단한 데이터 로딩 테스트
            processor = DataProcessor()
            print("✅ DataProcessor 초기화 성공")
            
            print("🎉 개발 환경 준비 완료!")
            
        except Exception as e:
            print(f"❌ 환경 설정 문제: {e}")
            exit(1)
    ```
    
    ### 2. 일일 개발 루틴
    
    #### 오전 (모델 실험)
    ```python
    # daily_experiment.py
    from utils.experiment_utils import ExperimentTracker
    from datetime import datetime
    
    def run_daily_experiment():
        """일일 실험 실행"""
        
        tracker = ExperimentTracker()
        
        # 오늘 날짜로 실험명 생성
        today = datetime.now().strftime("%Y%m%d")
        experiment_name = f"daily_experiment_{today}"
        
        # 실험 설정
        config = {
            "model": "kobart",
            "learning_rate": 5e-5,
            "batch_size": 16,
            "epochs": 3
        }
        
        # 실험 시작
        exp_id = tracker.start_experiment(
            name=experiment_name,
            description=f"{today} 일일 실험",
            config=config
        )
        
        try:
            # 실제 학습 실행
            results = run_training(config)
            
            # 최종 결과 기록
            tracker.end_experiment(results, "completed")
            
            print(f"✅ {experiment_name} 완료")
            return results
            
        except Exception as e:
            tracker.end_experiment({"error": str(e)}, "failed")
            print(f"❌ 실험 실패: {e}")
            return None
    
    if __name__ == "__main__":
        run_daily_experiment()
    ```
    
    #### 오후 (성능 분석)
    ```python
    # analyze_results.py
    from utils.experiment_utils import ExperimentTracker
    from utils.model_registry import ModelRegistry
    
    def analyze_daily_progress():
        """일일 진행 상황 분석"""
        
        tracker = ExperimentTracker()
        registry = ModelRegistry()
        
        # 실험 요약
        exp_summary = tracker.get_experiment_summary()
        print("📊 실험 요약:")
        print(exp_summary.head())
        
        # 모델 성능 순위
        model_summary = registry.get_models_summary()
        print("\n🏆 모델 성능 순위:")
        print(model_summary.head())
        
        # 최고 성능 모델
        best_model = registry.get_best_model()
        if best_model:
            print(f"\n🥇 최고 성능 모델:")
            print(f"  이름: {best_model['name']}")
            print(f"  성능: {best_model['performance']['rouge_combined_f1']:.4f}")
        
        return exp_summary, model_summary
    
    if __name__ == "__main__":
        analyze_daily_progress()
    ```
    
    ### 3. 주간 리뷰 및 계획
    
    #### 주간 성과 리포트
    ```python
    # weekly_report.py
    import pandas as pd
    from datetime import datetime, timedelta
    
    def generate_weekly_report():
        """주간 성과 리포트 생성"""
        
        tracker = ExperimentTracker()
        registry = ModelRegistry()
        
        # 이번 주 실험 필터링
        week_ago = datetime.now() - timedelta(days=7)
        exp_summary = tracker.get_experiment_summary()
        
        if not exp_summary.empty:
            exp_summary['start_time'] = pd.to_datetime(exp_summary['start_time'])
            this_week = exp_summary[exp_summary['start_time'] > week_ago]
            
            # 리포트 생성
            report = {
                "주간 실험 수": len(this_week),
                "성공한 실험": len(this_week[this_week['status'] == 'completed']),
                "최고 성능": this_week['best_rouge_combined_f1'].max(),
                "평균 성능": this_week['best_rouge_combined_f1'].mean(),
                "사용된 디바이스": this_week['device'].value_counts().to_dict()
            }
            
            # 리포트 출력
            print("📈 주간 성과 리포트")
            print("=" * 50)
            for key, value in report.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")
            
            # 다음 주 계획
            print("\n📋 다음 주 계획:")
            print("1. 하이퍼파라미터 최적화")
            print("2. 새로운 모델 아키텍처 실험")
            print("3. 데이터 증강 기법 적용")
            
            return report
        else:
            print("⚠️ 이번 주 실험 데이터 없음")
            return {}
    
    if __name__ == "__main__":
        generate_weekly_report()
    ```
    
    ---
    
    ## 마무리 및 다음 단계
    
    ### 개발 완료 체크리스트
    
    #### 기능 완성도 확인
    - [ ] **데이터 처리**: Multi-reference 형식 완전 지원
    - [ ] **모델 학습**: 모든 지원 아키텍처 정상 작동
    - [ ] **성능 평가**: ROUGE 메트릭 정확한 계산
    - [ ] **실험 관리**: 체계적인 추적 및 분석
    - [ ] **추론 시스템**: 배치 처리 및 제출 형식 지원
    - [ ] **크로스 플랫폼**: Mac MPS / Ubuntu CUDA 호환
    
    #### 코드 품질 확인
    - [ ] **상대 경로**: 모든 경로가 프로젝트 루트 기준
    - [ ] **에러 처리**: 포괄적인 예외 처리 구현
    - [ ] **문서화**: 함수/클래스 docstring 완성
    - [ ] **타입 힌트**: 모든 함수에 타입 annotation
    - [ ] **테스트**: 핵심 기능 검증 스크립트 작성
    
    #### 성능 최적화
    - [ ] **메모리 효율성**: 배치 크기 자동 조정
    - [ ] **속도 최적화**: 데이터 로더 및 추론 속도
    - [ ] **디바이스 활용**: GPU/MPS 최적화 설정
    - [ ] **캐싱**: 중복 계산 방지
    
    ### 향후 개선 방향
    
    #### 단기 목표 (1-2주)
    1. **성능 향상**
       - 더 정교한 하이퍼파라미터 튜닝
       - 데이터 증강 기법 적용
       - 앙상블 모델 실험
    
    2. **사용성 개선**
       - 웹 인터페이스 개발
       - API 서버 구축
       - 자동 배포 파이프라인
    
    #### 중기 목표 (1개월)
    1. **고급 기능**
       - 실시간 학습 모니터링
       - A/B 테스트 프레임워크
       - 자동 모델 선택
    
    2. **확장성**
       - 분산 학습 지원
       - 클라우드 배포
       - 스케일링 자동화
    
    #### 장기 목표 (3개월+)
    1. **연구 방향**
       - 새로운 아키텍처 실험
       - 멀티모달 확장
       - 도메인 특화 모델
    
    2. **프로덕션화**
       - 운영 모니터링
       - 성능 최적화
       - 유지보수 자동화
    
    ---
    
    ## 개발 리소스
    
    ### 추천 도구 및 라이브러리
    
    #### 개발 도구
    - **IDE**: VS Code + Python 확장팩
    - **디버깅**: IPython, pdb
    - **프로파일링**: cProfile, memory_profiler
    - **버전 관리**: Git + DVC (데이터 버전 관리)
    
    #### 모니터링 도구
    - **실험 추적**: Weights & Biases, MLflow
    - **성능 모니터링**: TensorBoard, Prometheus
    - **로그 분석**: Elasticsearch, Grafana
    
    #### 배포 도구
    - **컨테이너화**: Docker, Kubernetes
    - **CI/CD**: GitHub Actions, Jenkins
    - **인프라**: AWS, GCP, Azure
    
    ### 학습 자료
    
    #### 공식 문서
    - [Transformers Documentation](https://huggingface.co/docs/transformers)
    - [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
    - [Weights & Biases Guides](https://docs.wandb.ai)
    
    #### 추천 논문
    - BART: Lewis et al., "BART: Denoising Sequence-to-Sequence Pre-training"
    - ROUGE: Lin, "ROUGE: A Package for Automatic Evaluation of Summaries"
    - T5: Raffel et al., "Exploring the Limits of Transfer Learning"
    
    #### 커뮤니티
    - [Hugging Face Community](https://huggingface.co/community)
    - [PyTorch Forums](https://discuss.pytorch.org)
    - [Papers with Code](https://paperswithcode.com)
    
    ---
    
    이 종합 개발 가이드를 통해 NLP 대화 요약 시스템을 효과적으로 개발하고 최적화할 수 있습니다. 각 섹션의 코드와 가이드라인을 따라 단계별로 구현하면서 프로젝트를 성공적으로 완성하시기 바랍니다.
    
    **핵심 원칙을 항상 기억하세요:**
    - 📂 **상대 경로 사용**: 크로스 플랫폼 호환성
    - 🎯 **실험 중심**: 모든 변경사항을 체계적으로 추적
    - ⚡ **성능 최적화**: 메모리와 속도 효율성 고려
    - 🤝 **팀 협업**: 명확한 코딩 표준과 문서화
    - 🔄 **지속적 개선**: 작은 단위의 반복적 개발
    
    # 4. 실제 처리
    # ... 구현 코드 ...
    
    return processed_count
```

#### 클래스 설계 패턴
```python
# ✅ 권장 클래스 구조
class DialogueProcessor:
    \"\"\"대화 처리 클래스\"\"\"
    
    def __init__(self, config_path: Union[str, Path]):
        \"\"\"
        초기화
        
        Args:
            config_path: 설정 파일 경로 (상대 경로)
        \"\"\"
        # 1. 경로 검증
        if isinstance(config_path, str):
            config_path = Path(config_path)
        
        if config_path.is_absolute():
            raise ValueError(f\"상대 경로를 사용하세요: {config_path}\")
        
        # 2. 설정 로딩
        self.config = self._load_config(config_path)
        
        # 3. 디바이스 설정
        self.device = get_optimal_device()
        
        # 4. 내부 상태 초기화
        self._initialize_components()
    
    def _load_config(self, config_path: Path) -> dict:
        \"\"\"설정 파일 로딩 (private 메서드)\"\"\"
        # ... 구현 ...
        pass
    
    def _initialize_components(self):
        \"\"\"컴포넌트 초기화 (private 메서드)\"\"\"
        # ... 구현 ...
        pass
    
    def process(self, input_data: Any) -> Any:
        \"\"\"주요 처리 함수 (public 메서드)\"\"\"
        # ... 구현 ...
        pass
```

### 2. 에러 처리 패턴

#### 포괄적인 에러 처리
```python
def robust_model_training(config_path: str, 
                         train_data_path: str,
                         val_data_path: str) -> dict:
    \"\"\"견고한 모델 학습 함수\"\"\"
    
    try:
        # 1. 입력 검증
        if not all([config_path, train_data_path, val_data_path]):
            raise ValueError(\"모든 경로 파라미터가 필요합니다\")
        
        # 2. 설정 로딩
        config = load_config(config_path)
        
        # 3. 데이터 로딩
        train_data = load_training_data(train_data_path)
        val_data = load_validation_data(val_data_path)
        
        # 4. 모델 초기화
        model = initialize_model(config)
        
        # 5. 학습 실행
        results = train_model(model, train_data, val_data, config)
        
        return {\"status\": \"success\", \"results\": results}
        
    except ValueError as e:
        print(f\"❌ 입력 오류: {e}\")
        return {\"status\": \"error\", \"error_type\": \"ValueError\", \"message\": str(e)}
        
    except FileNotFoundError as e:
        print(f\"❌ 파일 없음: {e}\")
        return {\"status\": \"error\", \"error_type\": \"FileNotFoundError\", \"message\": str(e)}
        
    except torch.cuda.OutOfMemoryError as e:
        print(f\"❌ GPU 메모리 부족: {e}\")
        torch.cuda.empty_cache()
        return {\"status\": \"error\", \"error_type\": \"OutOfMemoryError\", \"message\": \"GPU 메모리 부족\"}
        
    except Exception as e:
        print(f\"❌ 예상치 못한 오류: {e}\")
        import traceback
        traceback.print_exc()
        return {\"status\": \"error\", \"error_type\": \"UnknownError\", \"message\": str(e)}
```

### 3. 로깅 및 모니터링

#### 구조화된 로깅
```python
import logging
import json
from datetime import datetime

def setup_structured_logging(log_file: str = \"logs/development.log\"):
    \"\"\"구조화된 로깅 설정\"\"\"
    
    # 로그 디렉토리 생성
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # 커스텀 포맷터
    class StructuredFormatter(logging.Formatter):
        def format(self, record):
            log_entry = {
                \"timestamp\": datetime.utcnow().isoformat(),
                \"level\": record.levelname,
                \"module\": record.module,
                \"function\": record.funcName,
                \"line\": record.lineno,
                \"message\": record.getMessage()
            }
            
            # 추가 속성이 있으면 포함
            if hasattr(record, 'extra_data'):
                log_entry.update(record.extra_data)
            
            return json.dumps(log_entry, ensure_ascii=False)
    
    # 로거 설정
    logger = logging.getLogger(\"nlp_development\")
    logger.setLevel(logging.INFO)
    
    # 파일 핸들러
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(StructuredFormatter())
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 사용 예시
logger = setup_structured_logging()

def log_experiment_start(experiment_name: str, config: dict):
    \"\"\"실험 시작 로깅\"\"\"
    logger.info(
        f\"실험 시작: {experiment_name}\",
        extra={'extra_data': {
            'experiment_name': experiment_name,
            'config': config,
            'device': get_optimal_device()
        }}
    )
```

---

## 고급 개발 기법

### 1. 커스텀 메트릭 개발

#### ROUGE 확장 메트릭
```python
# utils/custom_metrics.py (고급 메트릭)
class AdvancedRougeCalculator(RougeCalculator):
    \"\"\"확장된 ROUGE 계산기\"\"\"
    
    def __init__(self, use_korean_tokenizer: bool = True):
        super().__init__(use_korean_tokenizer)
        
        # 추가 메트릭 초기화
        self.custom_metrics = {}
    
    def compute_semantic_similarity(self, 
                                  predictions: List[str],
                                  references: List[str]) -> float:
        \"\"\"의미적 유사도 계산 (예: BERT Score 기반)\"\"\"
        
        # 간단한 구현 예시 (실제로는 BERT Score 사용)
        similarities = []
        
        for pred, ref in zip(predictions, references):
            # 단어 겹치는 정도 기반 간단한 유사도
            pred_words = set(pred.split())
            ref_words = set(ref.split())
            
            if len(ref_words) == 0:
                similarity = 0.0
            else:
                intersection = len(pred_words & ref_words)
                similarity = intersection / len(ref_words)
            
            similarities.append(similarity)
        
        return sum(similarities) / len(similarities)
    
    def compute_comprehensive_metrics(self, 
                                    predictions: List[str],
                                    references_list: List[List[str]]) -> dict:
        \"\"\"종합 메트릭 계산\"\"\"
        
        # 기본 ROUGE 계산
        rouge_scores = self.compute_multi_reference_rouge(predictions, references_list)
        
        # 추가 메트릭
        additional_metrics = {}
        
        # 첫 번째 참조로 의미적 유사도 계산
        if references_list:
            first_refs = [refs[0] for refs in references_list if refs]
            semantic_sim = self.compute_semantic_similarity(predictions, first_refs)
            additional_metrics['semantic_similarity'] = semantic_sim
        
        # 요약문 길이 분석
        pred_lengths = [len(p.split()) for p in predictions]
        additional_metrics['avg_summary_length'] = sum(pred_lengths) / len(pred_lengths)
        
        # 결과 결합
        return {
            **rouge_scores,
            \"additional_metrics\": additional_metrics
        }
```

### 2. 동적 하이퍼파라미터 튜닝

#### 베이지안 최적화 기반 튜닝
```python
# utils/hyperparameter_tuning.py (고급 튜닝)
class BayesianOptimizer:
    \"\"\"베이지안 최적화 기반 하이퍼파라미터 튜닝\"\"\"
    
    def __init__(self, parameter_space: dict, objective_metric: str = \"rouge_combined_f1\"):
        \"\"\"
        Args:
            parameter_space: 탐색할 파라미터 공간
            objective_metric: 최적화할 메트릭
        \"\"\"
        self.parameter_space = parameter_space
        self.objective_metric = objective_metric
        self.trials = []
    
    def suggest_parameters(self) -> dict:
        \"\"\"다음 시도할 파라미터 제안\"\"\"
        
        if len(self.trials) < 5:
            # 초기 랜덤 샘플링
            return self._random_sample()
        else:
            # 베이지안 최적화 (간단한 구현)
            return self._bayesian_sample()
    
    def _random_sample(self) -> dict:
        \"\"\"랜덤 파라미터 샘플링\"\"\"
        import random
        
        params = {}
        for param_name, param_config in self.parameter_space.items():
            if param_config[\"type\"] == \"uniform\":
                params[param_name] = random.uniform(
                    param_config[\"min\"], 
                    param_config[\"max\"]
                )
            elif param_config[\"type\"] == \"choice\":
                params[param_name] = random.choice(param_config[\"values\"])
            elif param_config[\"type\"] == \"log_uniform\":
                import math
                log_min = math.log(param_config[\"min\"])
                log_max = math.log(param_config[\"max\"])
                params[param_name] = math.exp(random.uniform(log_min, log_max))
        
        return params
    
    def _bayesian_sample(self) -> dict:
        \"\"\"베이지안 최적화 기반 샘플링 (단순화)\"\"\"
        
        # 가장 좋은 결과 기준으로 주변 탐색
        best_trial = max(self.trials, key=lambda t: t[\"score\"])
        best_params = best_trial[\"parameters\"]
        
        # 베스트 파라미터 주변에서 변형
        params = {}
        for param_name, value in best_params.items():
            param_config = self.parameter_space[param_name]
            
            if param_config[\"type\"] == \"uniform\":
                # 베스트 값 주변 ±20% 범위에서 샘플링
                noise_range = (param_config[\"max\"] - param_config[\"min\"]) * 0.2
                import random
                new_value = value + random.uniform(-noise_range, noise_range)
                params[param_name] = max(param_config[\"min\"], 
                                       min(param_config[\"max\"], new_value))
            else:
                params[param_name] = value
        
        return params
    
    def report_trial(self, parameters: dict, score: float):
        \"\"\"실험 결과 보고\"\"\"
        self.trials.append({
            \"parameters\": parameters,
            \"score\": score,
            \"timestamp\": datetime.now().isoformat()
        })
    
    def get_best_parameters(self) -> dict:
        \"\"\"최적 파라미터 조회\"\"\"
        if not self.trials:
            return None
        
        best_trial = max(self.trials, key=lambda t: t[\"score\"])
        return best_trial[\"parameters\"]

# 사용 예시
parameter_space = {
    \"learning_rate\": {
        \"type\": \"log_uniform\",
        \"min\": 1e-6,
        \"max\": 1e-3
    },
    \"batch_size\": {
        \"type\": \"choice\", 
        \"values\": [8, 16, 32]
    },
    \"warmup_ratio\": {
        \"type\": \"uniform\",
        \"min\": 0.0,
        \"max\": 0.3
    }
}

optimizer = BayesianOptimizer(parameter_space)

# 최적화 루프
for trial in range(20):
    # 다음 파라미터 제안
    params = optimizer.suggest_parameters()
    
    # 모델 학습 및 평가
    score = train_and_evaluate_model(params)
    
    # 결과 보고
    optimizer.report_trial(params, score)
    
    print(f\"Trial {trial+1}: Score {score:.4f}, Params {params}\")

# 최적 파라미터
best_params = optimizer.get_best_parameters()
print(f\"최적 파라미터: {best_params}\")
```

### 3. 앙상블 모델 개발

#### 다중 모델 앙상블
```python
# core/ensemble/ensemble_model.py (앙상블 가이드)
class EnsemblePredictor:
    \"\"\"다중 모델 앙상블 예측기\"\"\"
    
    def __init__(self, model_configs: List[dict]):
        \"\"\"
        Args:
            model_configs
