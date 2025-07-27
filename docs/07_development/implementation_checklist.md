# 개발 완료 체크리스트

## 🎯 개요

이 문서는 **추가 개발 사항들이 올바르게 구현되었는지 검증**하기 위한 종합적인 체크리스트입니다. 각 항목은 구체적인 테스트 방법과 기준을 포함합니다.

**최종 업데이트**: 2025-07-26
**구현 상태**: 🔴 미구현 / 🟡 부분 구현 / 🟢 완료

---

## 📊 현재 구현 상태 요약

### 필수 기능 구현 현황
| 기능 | 상태 | 비고 |
|------|------|------|
| PathManager (상대 경로 시스템) | 🔴 | 미구현 - 절대 경로 사용 중 |
| MPS 디바이스 지원 | 🔴 | 미구현 - CUDA만 지원 |
| 독립적인 추론 엔진 | 🔴 | 미구현 - baseline.ipynb 내장 |
| Multi-reference ROUGE | 🟡 | 부분 구현 - 기본 기능만 |
| 실험 추적 시스템 | 🟢 | 구현 완료 |
| 데이터 처리 시스템 | 🟢 | 구현 완료 |

---

## 🔴 Phase 1: 기반 구조 검증 (Critical - 미구현)

### 1. 크로스 플랫폼 경로 처리 시스템 🔴

#### ⚠️ 현재 상태: **미구현**
- `code/utils/path_utils.py` 파일이 존재하지 않음
- 모든 코드에서 절대 경로 사용 중
- 크로스 플랫폼 호환성 없음

#### ✅ 구현 필요 사항
```python
# code/utils/path_utils.py 생성 필요
from pathlib import Path
import os

class PathManager:
    @staticmethod
    def get_project_root() -> Path:
        """프로젝트 루트 디렉토리 자동 감지"""
        current = Path(__file__).resolve()
        while current != current.parent:
            if (current / 'code').exists() and (current / 'data').exists():
                return current
            current = current.parent
        raise RuntimeError("Project root not found")
    
    @staticmethod
    def resolve_path(relative_path: str) -> Path:
        """상대 경로를 절대 경로로 변환"""
        if Path(relative_path).is_absolute():
            raise ValueError(f"절대 경로는 사용할 수 없습니다: {relative_path}")
        return PathManager.get_project_root() / relative_path
```

#### 🧪 검증 테스트
```bash
# PathManager 구현 후 실행
python -c "from utils.path_utils import PathManager; print(PathManager.get_project_root())"
```

---

### 2. MPS (Mac) 디바이스 최적화 🔴

#### ⚠️ 현재 상태: **미구현**
- trainer.py에서 MPS 지원 코드 없음
- CUDA만 고려된 구현
- Mac 사용자는 CPU로만 실행 가능

#### ✅ 구현 필요 사항
```python
# code/utils/device_utils.py 생성 필요
import torch
import platform

def get_optimal_device() -> str:
    """최적 디바이스 자동 감지"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available() and platform.system() == "Darwin":
        return "mps"
    else:
        return "cpu"

def setup_device_config(config: dict) -> dict:
    """디바이스별 최적화 설정"""
    device = get_optimal_device()
    
    if device == "mps":
        # MPS 최적화
        config["training"]["fp16"] = False  # MPS는 fp16 미지원
        config["training"]["dataloader_num_workers"] = 0  # MPS 호환성
    elif device == "cuda":
        # CUDA 최적화  
        config["training"]["fp16"] = True
        config["training"]["dataloader_num_workers"] = 4
    
    return config
```

---

### 3. 독립적인 추론 파이프라인 🔴

#### ⚠️ 현재 상태: **미구현**
- `code/core/` 디렉토리 자체가 없음
- 추론 코드가 baseline.ipynb에만 존재
- CLI 도구 없음

#### ✅ 구현 필요 사항
```python
# code/core/inference.py 생성 필요
from typing import List, Union
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class InferenceEngine:
    def __init__(self, model_path: str):
        self.device = get_optimal_device()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    def predict_single(self, dialogue: str) -> str:
        """단일 대화 요약"""
        inputs = self.tokenizer(dialogue, return_tensors="pt", 
                               max_length=512, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=100)
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def predict_batch(self, dialogues: List[str], batch_size: int = 8) -> List[str]:
        """배치 예측"""
        predictions = []
        for i in range(0, len(dialogues), batch_size):
            batch = dialogues[i:i+batch_size]
            # 배치 처리 로직
            for dialogue in batch:
                predictions.append(self.predict_single(dialogue))
        return predictions
```

```python
# code/run_inference.py 생성 필요 (CLI)
import argparse
from core.inference import InferenceEngine
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()
    
    # 추론 실행
    engine = InferenceEngine(args.model_path)
    df = pd.read_csv(args.input_file)
    predictions = engine.predict_batch(df['dialogue'].tolist())
    
    # 결과 저장
    submission = pd.DataFrame({
        'fname': df['fname'],
        'summary': predictions
    })
    submission.to_csv(args.output_file, index=False)
    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
```

---

## 🟡 Phase 2: 핵심 기능 검증 (부분 구현)

### 4. Multi-Reference ROUGE 계산 시스템 🟡

#### ⚠️ 현재 상태: **부분 구현**
- `metrics.py`에 기본 RougeCalculator만 존재
- Multi-reference 전용 메서드 없음
- 한국어 토크나이저 통합 부분적

#### ✅ 추가 구현 필요
```python
# utils/metrics.py에 추가 필요
def calculate_multi_reference(self, prediction: str, 
                            references: List[str]) -> EvaluationResult:
    """다중 참조 ROUGE 계산 (대회 특성)"""
    if not references:
        return self._create_zero_score()
    
    # 각 참조와의 점수 계산
    scores = []
    for ref in references:
        score = self.calculate_single_reference(prediction, ref)
        scores.append(score)
    
    # 각 메트릭별로 최고 점수 선택
    best_rouge1_f1 = max(score.rouge1.f1 for score in scores)
    best_rouge2_f1 = max(score.rouge2.f1 for score in scores)  
    best_rougeL_f1 = max(score.rougeL.f1 for score in scores)
    
    # 결합 점수 (대회 최종 점수)
    rouge_combined_f1 = best_rouge1_f1 + best_rouge2_f1 + best_rougeL_f1
    
    return EvaluationResult(...)
```

---

## 🟢 Phase 3: 완료된 기능

### 5. ExperimentUtils 시스템 🟢

#### ✅ 현재 상태: **구현 완료**
- ExperimentTracker 클래스 구현됨
- ModelRegistry 클래스 구현됨
- 실험 추적 기능 정상 작동

### 6. 데이터 처리 시스템 🟢

#### ✅ 현재 상태: **구현 완료**
- DataProcessor 클래스 구현됨
- TextPreprocessor 클래스 구현됨
- 특수 토큰 처리 지원

---

## 📋 구현 우선순위 및 액션 플랜

### 🔴 긴급 구현 필요 (1-2일 내)

1. **PathManager 시스템 구현**
   - `code/utils/path_utils.py` 생성
   - 모든 파일에서 절대 경로를 상대 경로로 변경
   - 예상 작업 시간: 4-6시간

2. **MPS 디바이스 지원**
   - `code/utils/device_utils.py` 생성
   - trainer.py에 디바이스 감지 로직 추가
   - 예상 작업 시간: 2-3시간

3. **독립 추론 엔진**
   - `code/core/` 디렉토리 생성
   - `inference.py` 구현
   - `run_inference.py` CLI 도구 구현
   - 예상 작업 시간: 6-8시간

### 🟡 개선 필요 (3-5일 내)

1. **Multi-reference ROUGE 완성**
   - `calculate_multi_reference()` 메서드 추가
   - 대회 평가 방식과 100% 일치하도록 구현
   - 예상 작업 시간: 3-4시간

2. **에러 처리 시스템 표준화**
   - 통합 로깅 시스템 구현
   - 예외 처리 표준화
   - 예상 작업 시간: 4-5시간

### 🟢 선택적 개선 사항

1. **실험 결과 시각화 도구**
   - matplotlib/seaborn 기반 차트 생성
   - HTML 리포트 자동 생성
   - 예상 작업 시간: 6-8시간

2. **Solar API 통합 개선**
   - 비동기 처리 구현
   - Rate limit 자동 관리
   - 예상 작업 시간: 4-5시간

---

## 🚦 완료 승인 기준

### 필수 완료 항목 (배포 전)
- [ ] **모든 절대 경로 제거**
- [ ] **Mac/Linux/Windows 호환성 확인**
- [ ] **독립 추론 도구 작동 확인**
- [ ] **Multi-reference ROUGE 정확성 검증**

### 권장 완료 항목
- [ ] **MPS 디바이스 최적화**
- [ ] **통합 에러 처리 시스템**
- [ ] **성능 벤치마크 문서화**

---

## 📞 구현 지원 및 참고자료

### 즉시 시작 가능한 작업
1. PathManager 구현 → 모든 경로 처리 수정
2. device_utils.py 구현 → trainer.py 수정
3. inference.py 구현 → CLI 도구 생성

### 참고 문서
- [integration_action_plan.md](../team_progress/integration_action_plan.md) - 통합 가이드
- [project_structure_analysis.md](../project_structure_analysis.md) - 프로젝트 구조
- [baseline_code_analysis.md](../baseline_code_analysis.md) - 베이스라인 분석

---

**마지막 검토**: 2025-07-26  
**다음 업데이트 예정**: 구현 진행 상황에 따라 업데이트
- [ ] `register_model()` 모델 등록 기능
- [ ] `get_best_model()` 최고 모델 조회
- [ ] Trainer와 통합 완료
- [ ] 실험 메타데이터 자동 저장

---

## 🟡 Phase 2: 데이터 처리 검증 (Important)

### 6. Multi-Reference 데이터 처리

#### ✅ 구현 완료 기준
```bash
# DataProcessor 확장
code/utils/data_utils.py 확장 완료
load_multi_reference_data() 메서드 구현
create_inference_dataset() 메서드 구현
export_submission_format() 메서드 구현
```

#### 🧪 검증 테스트
```bash
# 1. Multi-reference 데이터 로딩 테스트
python -c "
from utils.data_utils import DataProcessor
import pandas as pd
import tempfile
import os

processor = DataProcessor()

# 테스트 데이터 생성
test_data = pd.DataFrame({
    'fname': ['test1.txt', 'test2.txt'],
    'dialogue': ['화자1: 안녕\\n화자2: 안녕', '화자1: 좋아\\n화자2: 좋아'],
    'summary1': ['안녕하세요', '좋습니다'],
    'summary2': ['안녕', '좋아요'],
    'summary3': ['안녕하세요', '좋아']
})

# 임시 파일로 저장
with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
    test_data.to_csv(f.name, index=False)
    temp_file = f.name

try:
    # Multi-reference 로딩 테스트
    df = processor.load_multi_reference_data(temp_file)
    print(f'Multi-reference data loaded: {len(df)} samples')
    print(f'Summaries format: {type(df.iloc[0][\"summaries\"])}')
    assert 'summaries' in df.columns, 'summaries column missing'
    assert len(df.iloc[0]['summaries']) == 3, 'Must have 3 summaries'
finally:
    os.unlink(temp_file)
"

# 2. 추론 데이터셋 생성 테스트
python -c "
from utils.data_utils import DataProcessor
import pandas as pd
import tempfile
import os

processor = DataProcessor()

# 테스트 데이터 생성 (추론용 - summary 없음)
test_data = pd.DataFrame({
    'fname': ['test1.txt', 'test2.txt'],
    'dialogue': ['화자1: 안녕\\n화자2: 안녕', '화자1: 좋아\\n화자2: 좋아']
})

with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
    test_data.to_csv(f.name, index=False)
    temp_file = f.name

try:
    df = processor.create_inference_dataset(temp_file)
    print(f'Inference dataset created: {len(df)} samples')
    assert 'dialogue_length' in df.columns, 'dialogue_length column missing'
    assert 'turn_count' in df.columns, 'turn_count column missing'
finally:
    os.unlink(temp_file)
"

# 3. 제출 형식 내보내기 테스트
python -c "
from utils.data_utils import DataProcessor
import tempfile
import os

processor = DataProcessor()

# 테스트 데이터
predictions = ['요약1', '요약2']
fnames = ['test1.txt', 'test2.txt']

with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
    temp_file = f.name

try:
    submission_df = processor.export_submission_format(predictions, fnames, temp_file)
    print(f'Submission format exported: {len(submission_df)} entries')
    
    # 검증
    assert list(submission_df.columns) == ['fname', 'summary'], 'Wrong column format'
    assert len(submission_df) == 2, 'Wrong number of entries'
    
    # 형식 검증
    is_valid = processor.validate_submission_format(temp_file)
    print(f'Submission format valid: {is_valid}')
    assert is_valid, 'Submission format validation failed'
    
finally:
    os.unlink(temp_file)
"
```

#### 📋 체크리스트
- [ ] `load_multi_reference_data()` 정상 작동
- [ ] 3개 정답 요약문 파싱 완료
- [ ] 다양한 구분자 형식 지원
- [ ] `create_inference_dataset()` 추론용 데이터 생성
- [ ] `export_submission_format()` 제출 형식 출력
- [ ] `validate_submission_format()` 형식 검증
- [ ] 대회 요구사항 100% 준수

---

## 🟢 Phase 3: 선택적 기능 검증

### 7. 성능 최적화 도구 (선택적)

#### 🧪 검증 테스트 (구현 시에만)
```bash
# MemoryProfiler 테스트 (구현된 경우)
python -c "
try:
    from utils.profiling import MemoryProfiler
    profiler = MemoryProfiler()
    print('MemoryProfiler imported successfully')
    
    # 기본 기능 테스트
    profiler.take_snapshot(0, 'test_snapshot')
    analysis = profiler.analyze_memory_patterns()
    print(f'Memory analysis completed: {len(analysis)} metrics')
    
except ImportError:
    print('MemoryProfiler not implemented (optional feature)')
"

# AutoTuner 테스트 (구현된 경우)
python -c "
try:
    from utils.auto_tuning import AutoTuner
    tuner = AutoTuner()
    print('AutoTuner imported successfully')
    
    # 배치 크기 제안 테스트
    suggested_batch = tuner.suggest_batch_size('medium', 16)
    print(f'Suggested batch size: {suggested_batch}')
    
except ImportError:
    print('AutoTuner not implemented (optional feature)')
"
```

### 8. 앙상블 시스템 (선택적)

#### 🧪 검증 테스트 (구현 시에만)
```bash
# EnsemblePredictor 테스트 (구현된 경우)
python -c "
try:
    from core.ensemble import EnsemblePredictor
    print('EnsemblePredictor imported successfully')
    
    # 기본 초기화 테스트
    # ensemble = EnsemblePredictor(['model1', 'model2'])
    print('Ensemble system is available')
    
except ImportError:
    print('EnsemblePredictor not implemented (optional feature)')
"
```

---

## 📊 통합 시스템 검증

### 전체 파이프라인 테스트

#### 🧪 End-to-End 테스트
```bash
# 1. 전체 워크플로우 테스트
python -c "
import os
import sys

# 기본 모듈 임포트 테스트
try:
    from utils.path_utils import PathManager
    from utils.config_manager import ConfigManager
    from utils.data_utils import DataProcessor
    from utils.metrics import RougeCalculator
    from utils.experiment_utils import ExperimentTracker, ModelRegistry
    print('✅ All core modules imported successfully')
    
    # 기본 설정 로딩 테스트
    config_manager = ConfigManager('config/base_config.yaml')
    config = config_manager.get_config()
    print('✅ Configuration loaded successfully')
    
    # 프로젝트 루트 확인
    project_root = PathManager.get_project_root()
    print(f'✅ Project root detected: {project_root}')
    
    # 필수 디렉토리 확인
    required_dirs = ['code', 'config', 'docs']
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        assert dir_path.exists(), f'Required directory missing: {dir_name}'
    print('✅ All required directories present')
    
    print('\\n🎉 Basic system integration test PASSED')
    
except Exception as e:
    print(f'❌ System integration test FAILED: {e}')
    sys.exit(1)
"

# 2. 크로스 플랫폼 호환성 최종 확인
python -c "
import platform
from pathlib import Path

print(f'Platform: {platform.system()} {platform.release()}')
print(f'Python: {platform.python_version()}')

# 경로 처리 테스트
test_path = Path('data') / 'test.csv'
print(f'Test path: {test_path}')
print(f'Is POSIX: {hasattr(test_path, \"as_posix\")}')

# 절대 경로 검증
if str(test_path).startswith('/Users/') or str(test_path).startswith('C:\\\\'):
    print('❌ Absolute path detected!')
else:
    print('✅ Relative path confirmed')
"
```

### 성능 기준 검증

#### 📋 성능 체크리스트
```bash
# 메모리 사용량 체크
python -c "
import psutil
import os

# 현재 프로세스 메모리 사용량
process = psutil.Process(os.getpid())
memory_mb = process.memory_info().rss / 1024 / 1024
print(f'Current memory usage: {memory_mb:.1f} MB')

# 기준: 기본 임포트 시 200MB 이하
if memory_mb > 200:
    print('⚠️  High memory usage detected')
else:
    print('✅ Memory usage within acceptable range')
"

# 임포트 속도 체크
python -c "
import time

start_time = time.time()
from trainer import DialogueSummarizationTrainer
end_time = time.time()

import_time = end_time - start_time
print(f'Trainer import time: {import_time:.2f} seconds')

# 기준: 5초 이하
if import_time > 5.0:
    print('⚠️  Slow import detected')
else:
    print('✅ Import time acceptable')
"
```

---

## 📋 최종 완료 체크리스트

### 🔴 필수 기능 (100% 완료 필요)
- [ ] **PathManager 시스템 구현 및 테스트 통과**
- [ ] **모든 절대 경로 제거 완료**
- [ ] **Multi-reference ROUGE 계산 정확성 검증**
- [ ] **완전한 추론 파이프라인 구축**
- [ ] **ExperimentUtils 시스템 완성**
- [ ] **Multi-reference 데이터 처리 구현**
- [ ] **크로스 플랫폼 호환성 검증 (Windows, macOS, Linux)**

### 🟡 중요 기능 (80% 이상 완료 권장)
- [ ] **성능 최적화 (메모리, 속도)**
- [ ] **에러 처리 및 로깅 표준화**
- [ ] **설정 검증 시스템**
- [ ] **문서화 완성**

### 🟢 선택적 기능 (필요에 따라)
- [ ] **성능 프로파일링 도구**
- [ ] **자동 하이퍼파라미터 제안**
- [ ] **앙상블 시스템**
- [ ] **고급 실험 분석 도구**

---

## 🚦 완료 승인 기준

### Green Light 조건 (배포 가능)
1. **모든 🔴 필수 기능 테스트 통과**
2. **크로스 플랫폼 호환성 100% 확인**
3. **메모리 누수 0건**
4. **대회 제출 형식 완벽 지원**
5. **기존 baseline 성능 동등 이상**

### Yellow Light 조건 (추가 작업 필요)
1. **필수 기능 중 일부 미완성**
2. **성능 이슈 존재**
3. **문서화 부족**

### Red Light 조건 (구현 재검토 필요)
1. **핵심 기능 작동 불가**
2. **크로스 플랫폼 호환성 실패**
3. **심각한 버그 존재**

---

## 📞 트러블슈팅 가이드

### 일반적인 문제 해결

#### 문제: PathManager 임포트 오류
```bash
# 해결 방법
1. utils/__init__.py 파일 존재 확인
2. PYTHONPATH 설정 확인
3. 가상환경 활성화 확인
```

#### 문제: 크로스 플랫폼 경로 오류
```bash
# 해결 방법
1. pathlib.Path 사용 확인
2. os.path.join() 사용 금지
3. 하드코딩된 구분자 제거
```

#### 문제: Multi-reference ROUGE 계산 오류
```bash
# 해결 방법
1. 입력 데이터 형식 확인
2. 토크나이저 설정 확인
3. 한국어 인코딩 문제 확인
```

### 에스컬레이션 절차
1. **Level 1**: 문서 참조 및 자체 해결 시도
2. **Level 2**: 팀 내 기술 리뷰 요청
3. **Level 3**: 외부 전문가 자문 요청

---

이 체크리스트를 통해 **체계적이고 완전한 검증**을 수행하여 높은 품질의 시스템을 보장할 수 있습니다.
