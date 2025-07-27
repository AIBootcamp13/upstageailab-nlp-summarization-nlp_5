    # 필수 개발 사항 (Priority 1) - 현재 구현 상태
    
    ## 📊 개요
    
    이 문서는 NLP 대화 요약 프로젝트의 **필수 구현 사항과 현재 상태**를 정리한 문서입니다.
    
    **최종 업데이트**: 2025-07-26
    **문서 상태**: 현재 구현 상태 기준 업데이트
    
    ---
    
    ## 🔴 현재 구현 상태 요약
    
    ### 전체 구현률: ~40%
    
    | 기능 | 상태 | 구현률 | 비고 |
    |------|------|---------|------|
    | 크로스 플랫폼 경로 시스템 | 🔴 | 0% | PathManager 미구현 |
    | MPS 디바이스 지원 | 🔴 | 0% | CUDA만 지원 |
    | 독립적인 추론 파이프라인 | 🔴 | 0% | core/ 디렉토리 없음 |
    | Multi-reference ROUGE | 🟡 | 60% | 기본 기능만 구현 |
    | 실험 추적 시스템 | 🟢 | 100% | 구현 완료 |
    | 데이터 처리 시스템 | 🟢 | 100% | 구현 완료 |
    
    ---
    
    ## 🔴 Phase 1: 크로스 플랫폼 경로 시스템 (0% 구현)
    
    ### 🎯 목표
    모든 파일에서 상대 경로를 사용하여 크로스 플랫폼 호환성 보장
    
    ### 🔍 현재 상태
    - **PathManager 클래스 미구현**
    - **모든 코드에서 절대 경로 하드코딩**
    - **Windows/Mac/Linux 호환성 없음**
    
    ### 📝 구현 필요 사항
    
    #### 1.1 PathManager 클래스 구현
    ```python
    # code/utils/path_utils.py
    from pathlib import Path
    import os
    from typing import Union, Optional
    
    class PathManager:
        """Cross-platform path management system"""
        
        _project_root: Optional[Path] = None
        
        @classmethod
        def get_project_root(cls) -> Path:
            """프로젝트 루트 디렉토리 자동 감지"""
            if cls._project_root is None:
                current = Path(__file__).resolve()
                
                # 프로젝트 루트 특징: code/, data/, config/ 디렉토리 존재
                while current != current.parent:
                    if all((current / d).exists() for d in ['code', 'data', 'config']):
                        cls._project_root = current
                        break
                    current = current.parent
                
                if cls._project_root is None:
                    raise RuntimeError(
                        "Project root not found. Make sure you're running from the project directory."
                    )
            
            return cls._project_root
        
        @staticmethod
        def resolve_path(relative_path: Union[str, Path]) -> Path:
            """상대 경로를 절대 경로로 변환"""
            if isinstance(relative_path, str):
                relative_path = Path(relative_path)
            
            if relative_path.is_absolute():
                raise ValueError(
                    f"절대 경로는 사용할 수 없습니다: {relative_path}\n"
                    f"상대 경로를 사용하세요. 예: 'data/train.csv'"
                )
            
            return PathManager.get_project_root() / relative_path
        
        @staticmethod
        def ensure_dir(directory: Union[str, Path]) -> Path:
            """디렉토리 생성 보장"""
            if isinstance(directory, str):
                directory = Path(directory)
            
            # 상대 경로로 변환
            if not directory.is_absolute():
                directory = PathManager.resolve_path(directory)
            
            directory.mkdir(parents=True, exist_ok=True)
            return directory
    ```
    
    #### 1.2 기존 코드 수정 필요 파일
    - `trainer.py`: 모든 경로 처리를 PathManager로 수정
    - `config_manager.py`: 설정 파일 경로 처리 수정
    - `data_utils.py`: 데이터 파일 경로 처리 수정
    - `sweep_runner.py`: 실험 출력 경로 처리 수정
    
    ### ⚠️ 주의사항
    - 현재 모든 파일에 하드코딩된 절대 경로 존재
    - Windows 경로 구분자 호환성 필요
    
    ---
    
    ## 🔴 Phase 2: MPS 디바이스 최적화 (0% 구현)
    
    ### 🎯 목표
    Mac Apple Silicon (M1/M2) 사용자를 위한 MPS 디바이스 지원
    
    ### 🔍 현재 상태
    - **MPS 감지 코드 없음**
    - **CUDA만 고려된 구현**
    - **Mac 사용자는 CPU로만 실행**
    
    ### 📝 구현 필요 사항
    
    #### 2.1 디바이스 유틸리티 구현
    ```python
    # code/utils/device_utils.py
    import torch
    import platform
    from typing import Dict, Any
    import logging
    
    logger = logging.getLogger(__name__)
    
    def get_optimal_device() -> str:
        """최적 디바이스 자동 감지"""
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"CUDA device detected: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available() and platform.system() == "Darwin":
            device = "mps"
            logger.info("Apple MPS device detected")
        else:
            device = "cpu"
            logger.info("Using CPU device")
        
        return device
    
    def setup_device_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """디바이스별 최적화 설정"""
        device = get_optimal_device()
        
        # 디바이스별 기본 설정
        device_configs = {
            "mps": {
                "fp16": False,  # MPS는 현재 fp16 미지원
                "dataloader_num_workers": 0,  # MPS 호환성
                "gradient_accumulation_steps": 4,  # 메모리 효율
                "per_device_train_batch_size": 4
            },
            "cuda": {
                "fp16": True,
                "dataloader_num_workers": 4,
                "gradient_accumulation_steps": 1,
                "per_device_train_batch_size": 8
            },
            "cpu": {
                "fp16": False,
                "dataloader_num_workers": 2,
                "gradient_accumulation_steps": 8,
                "per_device_train_batch_size": 2
            }
        }
        
        # 디바이스별 설정 적용
        device_specific = device_configs.get(device, device_configs["cpu"])
        
        if "training" not in config:
            config["training"] = {}
        
        config["training"].update(device_specific)
        config["device"] = device
        
        return config
    ```
    
    #### 2.2 Trainer 클래스 수정 필요
    - 디바이스 자동 감지 로직 추가
    - MPS에서 mixed precision training 비활성화
    - 배치 크기 자동 조정
    
    ---
    
    ## 🔴 Phase 3: 독립적인 추론 파이프라인 (0% 구현)
    
    ### 🎯 목표
    baseline.ipynb에서 분리된 독립적인 추론 엔진 및 CLI 도구
    
    ### 🔍 현재 상태
    - **core/ 디렉토리 자체가 없음**
    - **추론 코드가 notebook에만 존재**
    - **CLI 도구 없음**
    - **대회 제출 형식 지원 없음**
    
    ### 📝 구현 필요 사항
    
    #### 3.1 InferenceEngine 클래스
    ```python
    # code/core/inference.py
    from typing import List, Union, Optional, Dict, Any
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from tqdm import tqdm
    import pandas as pd
    from pathlib import Path
    
    from utils.device_utils import get_optimal_device
    from utils.path_utils import PathManager
    
    class InferenceEngine:
        """독립적인 추론 엔진"""
        
        def __init__(self, model_path: Union[str, Path], config: Optional[Dict[str, Any]] = None):
            self.device = get_optimal_device()
            self.model_path = PathManager.resolve_path(model_path)
            
            # 모델 및 토크나이저 로드
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                str(self.model_path)
            ).to(self.device)
            
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            
            # 기본 설정
            self.max_input_length = config.get('max_input_length', 512) if config else 512
            self.max_output_length = config.get('max_output_length', 100) if config else 100
            self.batch_size = config.get('batch_size', 8) if config else 8
            
            # 디바이스별 배치 크기 조정
            if self.device == "mps":
                self.batch_size = min(self.batch_size, 4)
            elif self.device == "cpu":
                self.batch_size = min(self.batch_size, 2)
        
        def predict_single(self, dialogue: str) -> str:
            """단일 대화 요약"""
            # 입력 처리
            inputs = self.tokenizer(
                dialogue,
                return_tensors="pt",
                max_length=self.max_input_length,
                truncation=True,
                padding=True
            )
            
            # 디바이스로 이동
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 추론
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_output_length,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3
                )
            
            # 디코딩
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return summary
        
        def predict_batch(
            self, 
            dialogues: List[str], 
            batch_size: Optional[int] = None,
            show_progress: bool = True
        ) -> List[str]:
            """배치 예측"""
            if batch_size is None:
                batch_size = self.batch_size
            
            predictions = []
            
            # 진행률 표시
            iterator = range(0, len(dialogues), batch_size)
            if show_progress:
                iterator = tqdm(iterator, desc="Generating summaries")
            
            for i in iterator:
                batch = dialogues[i:i+batch_size]
                
                # 배치 토큰화
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    max_length=self.max_input_length,
                    truncation=True,
                    padding=True
                )
                
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # 배치 추론
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=self.max_output_length,
                        num_beams=4,
                        early_stopping=True,
                        no_repeat_ngram_size=3
                    )
                
                # 디코딩
                batch_summaries = [
                    self.tokenizer.decode(output, skip_special_tokens=True)
                    for output in outputs
                ]
                
                predictions.extend(batch_summaries)
            
            return predictions
        
        def predict_from_dataframe(
            self,
            df: pd.DataFrame,
            dialogue_column: str = "dialogue",
            fname_column: str = "fname"
        ) -> pd.DataFrame:
            """데이터프레임에서 직접 예측"""
            dialogues = df[dialogue_column].tolist()
            predictions = self.predict_batch(dialogues)
            
            # 결과 데이터프레임 생성
            result_df = pd.DataFrame({
                fname_column: df[fname_column],
                'summary': predictions
            })
            
            return result_df
    ```
    
    #### 3.2 CLI 도구
    ```python
    # code/run_inference.py
    import argparse
    import pandas as pd
    from pathlib import Path
    import logging
    import sys
    
    # code 디렉토리를 Python 경로에 추가
    sys.path.append(str(Path(__file__).parent))
    
    from core.inference import InferenceEngine
    from utils.path_utils import PathManager
    
    def main():
        parser = argparse.ArgumentParser(
            description="NLP 대화 요약 추론 도구"
        )
        
        parser.add_argument(
            "--model_path",
            type=str,
            required=True,
            help="학습된 모델 경로 (outputs/best_model 등)"
        )
        
        parser.add_argument(
            "--input_file",
            type=str,
            required=True,
            help="입력 CSV 파일 경로 (data/test.csv 등)"
        )
        
        parser.add_argument(
            "--output_file",
            type=str,
            required=True,
            help="출력 CSV 파일 경로 (outputs/submission.csv 등)"
        )
        
        parser.add_argument(
            "--batch_size",
            type=int,
            default=8,
            help="배치 크기 (default: 8)"
        )
        
        args = parser.parse_args()
        
        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        logger = logging.getLogger(__name__)
        
        try:
            # 추론 엔진 초기화
            logger.info(f"Loading model from {args.model_path}")
            engine = InferenceEngine(
                model_path=args.model_path,
                config={'batch_size': args.batch_size}
            )
            
            # 데이터 로드
            logger.info(f"Loading data from {args.input_file}")
            input_path = PathManager.resolve_path(args.input_file)
            df = pd.read_csv(input_path)
            
            # 추론 실행
            logger.info(f"Running inference on {len(df)} samples")
            result_df = engine.predict_from_dataframe(df)
            
            # 결과 저장
            output_path = PathManager.resolve_path(args.output_file)
            PathManager.ensure_dir(output_path.parent)
            result_df.to_csv(output_path, index=False)
            
            logger.info(f"Results saved to {args.output_file}")
            logger.info(f"Total predictions: {len(result_df)}")
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise
    
    if __name__ == "__main__":
        main()
    ```
    
    ---
    
    ## 🟡 Phase 4: Multi-Reference ROUGE 계산 시스템 (60% 구현)
    
    ### 🎯 목표
    3개 정답 요약문에 대한 정확한 ROUGE 계산
    
    ### 🔍 현재 상태
    - **기본 RougeCalculator 클래스 구현됨**
    - **Multi-reference 전용 메서드 없음**
    - **대회 평가 방식과의 일치성 미확인**
    
    ### 📝 추가 구현 필요
    
    #### 4.1 Multi-reference ROUGE 메서드
    ```python
    # utils/metrics.py에 추가
    def calculate_multi_reference(
        self,
        prediction: str,
        references: List[str]
    ) -> EvaluationResult:
        """
        다중 참조 ROUGE 계산 (대회 평가 방식)
        
        Args:
            prediction: 예측된 요약문
            references: 3개의 정답 요약문 리스트
        
        Returns:
            EvaluationResult: 각 메트릭별 최고 점수
        """
        if not references:
            return self._create_zero_score()
        
        # 각 참조와의 ROUGE 점수 계산
        all_scores = []
        for ref in references:
            if ref and ref.strip():  # 비어있는 참조 건너뛰기
                score = self.calculate_single_reference(prediction, ref)
                all_scores.append(score)
        
        if not all_scores:
            return self._create_zero_score()
        
        # 각 메트릭별로 최고 점수 선택 (대회 규칙)
        best_rouge1_f1 = max(score.rouge1.f1 for score in all_scores)
        best_rouge2_f1 = max(score.rouge2.f1 for score in all_scores)
        best_rougeL_f1 = max(score.rougeL.f1 for score in all_scores)
        
        # Precision과 Recall도 최고 점수 기준
        best_rouge1_precision = max(score.rouge1.precision for score in all_scores)
        best_rouge1_recall = max(score.rouge1.recall for score in all_scores)
        
        best_rouge2_precision = max(score.rouge2.precision for score in all_scores)
        best_rouge2_recall = max(score.rouge2.recall for score in all_scores)
        
        best_rougeL_precision = max(score.rougeL.precision for score in all_scores)
        best_rougeL_recall = max(score.rougeL.recall for score in all_scores)
        
        # 결합 점수 (대회 최종 점수)
        rouge_combined_f1 = best_rouge1_f1 + best_rouge2_f1 + best_rougeL_f1
        
        return EvaluationResult(
            rouge1=RougeScore(
                precision=best_rouge1_precision,
                recall=best_rouge1_recall,
                f1=best_rouge1_f1
            ),
            rouge2=RougeScore(
                precision=best_rouge2_precision,
                recall=best_rouge2_recall,
                f1=best_rouge2_f1
            ),
            rougeL=RougeScore(
                precision=best_rougeL_precision,
                recall=best_rougeL_recall,
                f1=best_rougeL_f1
            ),
            rouge_combined_f1=rouge_combined_f1
        )
    
    def compute_metrics_for_trainer(
        self,
        predictions: List[str],
        references_list: List[List[str]]
    ) -> Dict[str, float]:
        """
        Trainer에서 사용할 메트릭 계산 (Multi-reference 지원)
        """
        total_scores = {
            'rouge1_f1': 0,
            'rouge2_f1': 0,
            'rougeL_f1': 0,
            'rouge_combined_f1': 0
        }
        
        for pred, refs in zip(predictions, references_list):
            result = self.calculate_multi_reference(pred, refs)
            
            total_scores['rouge1_f1'] += result.rouge1.f1
            total_scores['rouge2_f1'] += result.rouge2.f1
            total_scores['rougeL_f1'] += result.rougeL.f1
            total_scores['rouge_combined_f1'] += result.rouge_combined_f1
        
        # 평균 계산
        n = len(predictions)
        return {k: v / n for k, v in total_scores.items()}
    ```
    
    ---
    
    ## 🟢 Phase 5: 완료된 기능
    
    ### 5.1 실험 추적 시스템 (100% 구현)
    - `ExperimentTracker` 클래스 구현 완료
    - `ModelRegistry` 클래스 구현 완료
    - 실험 메타데이터 자동 저장
    - WandB와 통합 가능
    
    ### 5.2 데이터 처리 시스템 (100% 구현)
    - `DataProcessor` 클래스 구현 완료
    - `TextPreprocessor` 클래스 구현 완료
    - 특수 토큰 처리 지원
    - DialogSum 데이터셋 특화 처리
    
    ---
    
    ## 📋 개발 우선순위 및 액션 플랜
    
    ### 🔴 긴급 개발 필요 (1-2일)
    
    1. **PathManager 시스템**
       - 파일: `code/utils/path_utils.py`
       - 예상 시간: 4-6시간
       - 영향: 모든 파일 수정 필요
    
    2. **MPS 디바이스 지원**
       - 파일: `code/utils/device_utils.py`
       - 예상 시간: 2-3시간
       - 영향: trainer.py 수정
    
    3. **독립 추론 엔진**
       - 파일: `code/core/inference.py`, `code/run_inference.py`
       - 예상 시간: 6-8시간
       - 영향: 대회 제출 프로세스 개선
    
    ### 🟡 개선 필요 (3-5일)
    
    1. **Multi-reference ROUGE 완성**
       - 파일: `code/utils/metrics.py`
       - 예상 시간: 3-4시간
       - 영향: 평가 정확성 향상
    
    2. **통합 테스트 및 검증**
       - 예상 시간: 4-5시간
       - 모든 기능 통합 테스트
    
    ---
    
    ## 📞 참고자료 및 지원
    
    ### 관련 문서
    - [implementation_checklist.md](implementation_checklist.md) - 상세 체크리스트
    - [integration_action_plan.md](../team_progress/integration_action_plan.md) - 통합 가이드
    - [baseline_code_analysis.md](../baseline_code_analysis.md) - 코드 분석
    
    ### 즉시 시작 가능한 작업
    1. PathManager 구현 → 모든 경로 처리 수정
    2. device_utils.py 구현 → trainer.py 수정
    3. inference.py 구현 → CLI 도구 생성
    
    ---
    
    **마지막 업데이트**: 2025-07-26  
    **다음 단계**: PathManager 구현 후 모든 코드 업데이트

## 📋 Priority 1 구현 체크리스트

### Phase 1: 기반 시스템 (Day 1-3)
- [ ] **PathManager 시스템** (Day 1)
  - [ ] `code/utils/path_utils.py` 구현
  - [ ] 상대 경로 강제 및 절대 경로 금지
  - [ ] 크로스 플랫폼 호환성 확인
  - [ ] `python validate_paths.py` 검증 통과

- [ ] **디바이스 최적화** (Day 1)
  - [ ] `code/utils/device_utils.py` 구현
  - [ ] MPS (Mac) / CUDA (Ubuntu) 자동 감지
  - [ ] 디바이스별 최적화 설정 적용

- [ ] **기존 코드 수정** (Day 2)
  - [ ] `trainer.py` 경로 처리 수정
  - [ ] `config_manager.py` 경로 처리 수정
  - [ ] `sweep_runner.py` 경로 처리 수정
  - [ ] 모든 절대 경로 제거 확인

### Phase 2: 핵심 기능 (Day 4-7)
- [ ] **Multi-Reference ROUGE** (Day 3-4)
  - [ ] `code/utils/metrics.py` 완전 구현
  - [ ] 3개 정답 요약문 최적 점수 계산
  - [ ] 한국어 토크나이저 통합
  - [ ] `python test_rouge_calculator.py` 검증 통과

- [ ] **추론 파이프라인** (Day 5-6)
  - [ ] `code/core/inference.py` 구현
  - [ ] 디바이스별 최적화 추론
  - [ ] CLI 인터페이스 구현
  - [ ] `python test_inference_engine.py` 검증 통과

- [ ] **실험 추적 시스템** (Day 7)
  - [ ] `code/utils/experiment_utils.py` 완전 구현
  - [ ] ExperimentTracker, ModelRegistry 구현
  - [ ] `python test_experiment_utils.py` 검증 통과

### Phase 3: 데이터 처리 (Day 8-9)
- [ ] **Multi-Reference 데이터 처리** (Day 8-9)
  - [ ] `code/utils/data_utils.py` 확장
  - [ ] 3개 정답 요약문 로딩
  - [ ] 대회 제출 형식 완벽 지원
  - [ ] `python test_data_processor.py` 검증 통과

### Phase 4: 통합 및 검증 (Day 10)
- [ ] **전체 시스템 통합**
  - [ ] 모든 모듈 간 연동 확인
  - [ ] 크로스 플랫폼 실행 테스트
  - [ ] 메모리 사용량 확인
  - [ ] 성능 기준 달성 확인

---

## ⚠️ 핵심 주의사항

### 1. 상대 경로 엄격 준수
```python
# ❌ 절대 경로 사용 금지
"/Users/jayden/project/data/train.csv"
"C:\\Users\\project\\data\\train.csv"

# ✅ 상대 경로만 허용
"data/train.csv"
"outputs/model/best_model"
"config/base_config.yaml"
```

### 2. 디바이스 최적화 필수
```python
# 모든 모델 로딩 시 디바이스 자동 감지 적용
device = get_optimal_device()  # "mps", "cuda", "cpu"

# 디바이스별 최적화 설정 적용
if device == "mps":
    model = model.to("mps")
    torch_dtype = torch.float32  # MPS는 float32 권장
elif device == "cuda":
    model = model.to("cuda") 
    torch_dtype = torch.float16  # GPU 메모리 절약
```

### 3. 실행 중심 검증
- **테스트 코드 작성 불필요**
- **즉시 실행 가능한 검증 스크립트 제공**
- **에러 발생 시 바로 수정하는 방식**

---

## 🎯 최종 성공 기준

### 기술적 성공 기준
- [ ] **Mac (MPS) / Ubuntu (CUDA)에서 동일 결과**
- [ ] **모든 경로가 상대 경로 기반**
- [ ] **Multi-reference ROUGE 정확 계산**
- [ ] **대회 제출 형식 100% 준수**
- [ ] **메모리 사용량 기존 대비 120% 이내**

### 품질 관리 기준
- [ ] **모든 검증 스크립트 통과**
- [ ] **에러 처리 완전 구현**
- [ ] **API 문서 완성**
- [ ] **사용자 가이드 업데이트**

### 즉시 실행 검증
```bash
# 최종 통합 검증 (프로젝트 루트에서 실행)
python validate_paths.py          # 경로 시스템 검증
python test_rouge_calculator.py   # ROUGE 계산 검증  
python test_inference_engine.py   # 추론 엔진 검증
python test_experiment_utils.py   # 실험 추적 검증
python test_data_processor.py     # 데이터 처리 검증
```

모든 검증 스크립트가 성공하면 **Priority 1 구현 완료**로 간주합니다.

---

## 📞 구현 지원

### 즉시 시작 방법
1. **프로젝트 루트에서 실행**: 모든 스크립트는 프로젝트 루트에서 실행
2. **단계별 구현**: PathManager → ROUGE → Inference → Experiment → Data 순서
3. **즉시 검증**: 각 단계마다 해당 검증 스크립트 실행
4. **에러 즉시 수정**: 테스트 코드 없이 직접 실행으로 문제 해결

### 구현 우선순위
1. **PathManager (최우선)**: 모든 다른 모듈의 기반
2. **ROUGE Calculator**: 평가 시스템의 핵심
3. **Inference Engine**: 실제 사용의 핵심  
4. **Experiment Utils**: 체계적 관리
5. **Data Processor**: 완전한 데이터 지원

이 순서로 구현하면 **의존성 문제 없이** 체계적인 개발이 가능합니다.
