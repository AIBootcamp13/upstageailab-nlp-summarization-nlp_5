# 추론 시스템 구현 상세 명세서

## 1. 구현 대상 파일

### 1.1 수정할 파일
- `/code/auto_experiment_runner.py` - `_run_test_inference()` 메서드 추가
- `/code/post_training_inference.py` - 전체 리팩토링

### 1.2 새로 생성할 파일
- `/code/utils/baseline_compatible.py` - baseline.py 호환 클래스들
- `/code/utils/model_handler.py` - 모델별 처리 핸들러

## 2. 구현 상세

### 2.1 auto_experiment_runner.py의 _run_test_inference()

```python
def _run_test_inference(self, 
                       experiment_id: str,
                       checkpoint_path: str,
                       config: Dict[str, Any]) -> Dict[str, Any]:
    """
    학습 완료 후 test.csv에 대한 추론 수행
    
    Args:
        experiment_id: 실험 ID
        checkpoint_path: 모델 체크포인트 경로
        config: 실험 설정
        
    Returns:
        Dict containing:
        - submission_path: 생성된 제출 파일 경로
        - experiment_id: 실험 ID
        - model_name: 모델 이름
        - status: 성공/실패 상태
        - metrics: 추론 관련 메트릭 (선택)
    """
    import torch
    from datetime import datetime
    from .post_training_inference import PostTrainingInference
    
    try:
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # PostTrainingInference 초기화
        inference_runner = PostTrainingInference(
            experiment_name=experiment_id,
            model_path=checkpoint_path,
            config=config
        )
        
        # test.csv 경로
        test_file = path_manager.resolve_path("data/test.csv")
        if not test_file.exists():
            raise FileNotFoundError(f"Test file not found: {test_file}")
        
        # 추론 실행
        logger.info(f"Starting inference for experiment: {experiment_id}")
        submission_path = inference_runner.run_test_inference(str(test_file))
        
        # CSV 관리자를 통한 결과 등록
        self.csv_manager.register_submission(
            experiment_name=experiment_id,
            submission_path=submission_path,
            model_info={
                "model_name": config.get('general', {}).get('model_name', 'unknown'),
                "architecture": config.get('model', {}).get('architecture', 'unknown'),
                "checkpoint": checkpoint_path
            }
        )
        
        # experiment_index.csv 업데이트
        self._update_experiment_index(
            experiment_id=experiment_id,
            submission_path=submission_path,
            config=config
        )
        
        logger.info(f"Inference completed successfully: {submission_path}")
        
        return {
            "submission_path": str(submission_path),
            "experiment_id": experiment_id,
            "model_name": config.get('general', {}).get('model_name'),
            "status": "success",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Inference failed for {experiment_id}: {str(e)}", exc_info=True)
        return {
            "experiment_id": experiment_id,
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
```

### 2.2 baseline_compatible.py

```python
"""
baseline.py 호환 클래스 및 함수

baseline.py의 핵심 로직을 정확히 재현하는 클래스들
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import List, Tuple, Dict, Any


class BaselinePreprocess:
    """baseline.py의 Preprocess 클래스 재현"""
    
    def __init__(self, bos_token: str, eos_token: str):
        self.bos_token = bos_token
        self.eos_token = eos_token
    
    @staticmethod
    def make_set_as_df(file_path: str, is_train: bool = True):
        """데이터 로드 및 필요 컬럼 선택"""
        df = pd.read_csv(file_path)
        if is_train:
            return df[['fname', 'dialogue', 'summary']]
        else:
            return df[['fname', 'dialogue']]
    
    def make_input(self, dataset: pd.DataFrame, is_test: bool = False) -> Tuple:
        """BART 모델 입력 형태로 변환"""
        if is_test:
            encoder_input = dataset['dialogue']
            decoder_input = [self.bos_token] * len(dataset['dialogue'])
            return encoder_input.tolist(), list(decoder_input)
        else:
            encoder_input = dataset['dialogue']
            decoder_input = dataset['summary'].apply(lambda x: self.bos_token + str(x))
            decoder_output = dataset['summary'].apply(lambda x: str(x) + self.eos_token)
            return encoder_input.tolist(), decoder_input.tolist(), decoder_output.tolist()


class DatasetForInference(Dataset):
    """baseline.py의 DatasetForInference 클래스 재현"""
    
    def __init__(self, encoder_input, test_id, length):
        self.encoder_input = encoder_input
        self.test_id = test_id
        self.len = length
    
    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()}
        item['ID'] = self.test_id[idx]
        return item
    
    def __len__(self):
        return self.len


def prepare_test_dataset(config: Dict[str, Any], preprocessor: BaselinePreprocess, tokenizer):
    """baseline.py의 prepare_test_dataset 함수 재현"""
    import os
    
    test_file_path = os.path.join(config['general']['data_path'], 'test.csv')
    
    # 데이터 로드
    test_data = preprocessor.make_set_as_df(test_file_path, is_train=False)
    test_id = test_data['fname']
    
    print('-'*150)
    print(f'test_data:\n{test_data["dialogue"][0]}')
    print('-'*150)
    
    # 입력 생성
    encoder_input_test, decoder_input_test = preprocessor.make_input(test_data, is_test=True)
    print('-'*10, 'Load data complete', '-'*10)
    
    # 토큰화
    test_tokenized_encoder_inputs = tokenizer(
        encoder_input_test,
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
        truncation=True,
        max_length=config['tokenizer']['encoder_max_len'],
        return_token_type_ids=False
    )
    
    # Dataset 생성
    test_encoder_inputs_dataset = DatasetForInference(
        test_tokenized_encoder_inputs,
        test_id.tolist(),
        len(encoder_input_test)
    )
    
    print('-'*10, 'Make dataset complete', '-'*10)
    
    return test_data, test_encoder_inputs_dataset


def remove_special_tokens(summaries: List[str], remove_tokens: List[str]) -> List[str]:
    """baseline.py의 특수 토큰 제거 로직 재현"""
    preprocessed_summaries = summaries.copy()
    for token in remove_tokens:
        preprocessed_summaries = [sentence.replace(token, " ") for sentence in preprocessed_summaries]
    return preprocessed_summaries
```

### 2.3 model_handler.py

```python
"""
모델별 처리 핸들러

다양한 모델 아키텍처에 대한 통합 처리
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ModelSpecificHandler:
    """모델별 특성 처리 핸들러"""
    
    # 모델별 설정
    MODEL_CONFIGS = {
        "mt5": {
            "model_class": "T5ForConditionalGeneration",
            "tokenizer_class": "T5Tokenizer",
            "use_prefix": True,
            "default_prefix": "dialogue summarization in korean: ",
            "architecture": "t5"
        },
        "t5": {
            "model_class": "T5ForConditionalGeneration", 
            "tokenizer_class": "T5Tokenizer",
            "use_prefix": True,
            "default_prefix": "dialogue summarization in korean: ",
            "architecture": "t5"
        },
        "bart": {
            "model_class": "BartForConditionalGeneration",
            "tokenizer_class": "BartTokenizer",
            "use_prefix": False,
            "default_prefix": "",
            "architecture": "bart"
        },
        "kobart": {
            "model_class": "BartForConditionalGeneration",
            "tokenizer_class": "PreTrainedTokenizer",
            "use_prefix": False,
            "default_prefix": "",
            "architecture": "bart"
        }
    }
    
    @classmethod
    def detect_model_type(cls, model_name: str) -> str:
        """모델 이름에서 타입 추론"""
        model_name_lower = model_name.lower()
        
        if "mt5" in model_name_lower:
            return "mt5"
        elif "t5" in model_name_lower:
            return "t5"
        elif "kobart" in model_name_lower:
            return "kobart"
        elif "bart" in model_name_lower:
            return "bart"
        else:
            logger.warning(f"Unknown model type for {model_name}, defaulting to bart")
            return "bart"
    
    @classmethod
    def get_model_config(cls, model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """모델별 최적 설정 반환"""
        # 모델 타입 감지
        model_type = cls.detect_model_type(model_name)
        
        # 기본 설정 가져오기
        model_config = cls.MODEL_CONFIGS.get(model_type, cls.MODEL_CONFIGS["bart"]).copy()
        
        # 사용자 설정으로 업데이트
        if "input_prefix" in config:
            model_config["prefix"] = config["input_prefix"]
        else:
            model_config["prefix"] = model_config["default_prefix"]
        
        # 토크나이저 설정
        model_config["encoder_max_len"] = config.get("tokenizer", {}).get("encoder_max_len", 1024)
        model_config["decoder_max_len"] = config.get("tokenizer", {}).get("decoder_max_len", 200)
        
        # 생성 설정
        model_config["num_beams"] = config.get("inference", {}).get("num_beams", 4)
        model_config["no_repeat_ngram_size"] = config.get("inference", {}).get("no_repeat_ngram_size", 2)
        model_config["early_stopping"] = config.get("inference", {}).get("early_stopping", True)
        model_config["max_length"] = config.get("inference", {}).get("generate_max_length", 100)
        
        return model_config
    
    @classmethod
    def load_model_for_inference(cls, model_path: str, model_config: Dict[str, Any], device):
        """모델 로드 (아키텍처별 처리)"""
        import torch
        from transformers import AutoTokenizer
        
        architecture = model_config.get("architecture", "bart")
        
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 모델 로드
        if architecture == "t5":
            from transformers import T5ForConditionalGeneration
            model = T5ForConditionalGeneration.from_pretrained(model_path)
        else:  # bart
            from transformers import BartForConditionalGeneration
            model = BartForConditionalGeneration.from_pretrained(model_path)
        
        # 디바이스 설정
        model = model.to(device)
        model.eval()
        
        return model, tokenizer
    
    @classmethod
    def prepare_input_with_prefix(cls, texts: List[str], model_config: Dict[str, Any]) -> List[str]:
        """모델별 prefix 처리"""
        if model_config.get("use_prefix", False):
            prefix = model_config.get("prefix", "")
            if prefix:
                return [f"{prefix}{text}" for text in texts]
        return texts
```

### 2.4 수정된 post_training_inference.py

```python
#!/usr/bin/env python3
"""
학습 후 자동 추론 스크립트 - baseline.py 완벽 호환 버전

baseline.py의 inference() 함수를 정확히 재현하면서
다중 모델을 지원하도록 확장
"""

import os
import sys
import logging
from pathlib import Path
import pandas as pd
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any, List

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.path_utils import path_manager
from utils.baseline_compatible import (
    BaselinePreprocess, 
    DatasetForInference,
    prepare_test_dataset,
    remove_special_tokens
)
from utils.model_handler import ModelSpecificHandler

logger = logging.getLogger(__name__)


class PostTrainingInference:
    """학습 후 자동 추론 클래스 - baseline.py 완벽 호환"""
    
    def __init__(self, experiment_name: str, model_path: str, config: dict):
        """
        Args:
            experiment_name: 실험명
            model_path: 학습된 모델 체크포인트 경로
            config: 실험 설정
        """
        self.experiment_name = experiment_name
        self.model_path = model_path
        self.config = config
        
        # 모델별 설정 가져오기
        model_name = config.get('general', {}).get('model_name', '')
        self.model_config = ModelSpecificHandler.get_model_config(model_name, config)
        
        logger.info(f"Initializing PostTrainingInference for {experiment_name}")
        logger.info(f"Model: {model_name}, Architecture: {self.model_config.get('architecture')}")
    
    def run_test_inference(self, test_file: str) -> str:
        """
        baseline.py의 inference() 함수를 정확히 재현
        
        Args:
            test_file: 테스트 데이터 파일 경로
            
        Returns:
            생성된 제출 파일 경로
        """
        # 1. 디바이스 설정 (baseline.py와 동일)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('-'*10, f'device : {device}', '-'*10)
        print(torch.__version__)
        
        # 2. 모델과 토크나이저 로드
        print('-'*10, 'Load tokenizer & model', '-'*10)
        model, tokenizer = self._load_model_and_tokenizer(device)
        
        # 3. 특수 토큰 추가 (baseline.py와 동일)
        special_tokens_dict = {
            'additional_special_tokens': self.config['tokenizer']['special_tokens']
        }
        tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))
        
        # 4. 전처리기 초기화 (baseline.py의 Preprocess와 동일)
        preprocessor = BaselinePreprocess(
            self.config['tokenizer']['bos_token'],
            self.config['tokenizer']['eos_token']
        )
        
        # 5. 테스트 데이터 준비 (baseline.py의 prepare_test_dataset와 동일)
        test_data, test_encoder_inputs_dataset = self._prepare_test_data(
            preprocessor, tokenizer, test_file
        )
        
        # 6. DataLoader 생성 (baseline.py와 동일)
        dataloader = DataLoader(
            test_encoder_inputs_dataset,
            batch_size=self.config['inference']['batch_size']
        )
        
        # 7. 추론 실행 (baseline.py와 완전 동일)
        summary = []
        text_ids = []
        
        with torch.no_grad():
            for item in tqdm(dataloader):
                text_ids.extend(item['ID'])
                generated_ids = model.generate(
                    input_ids=item['input_ids'].to(device),
                    no_repeat_ngram_size=self.config['inference']['no_repeat_ngram_size'],
                    early_stopping=self.config['inference']['early_stopping'],
                    max_length=self.config['inference']['generate_max_length'],
                    num_beams=self.config['inference']['num_beams'],
                )
                for ids in generated_ids:
                    result = tokenizer.decode(ids)
                    summary.append(result)
        
        # 8. 특수 토큰 제거 (baseline.py와 동일)
        remove_tokens = self.config['inference']['remove_tokens']
        preprocessed_summary = remove_special_tokens(summary, remove_tokens)
        
        # 9. 결과 저장 (baseline.py와 동일한 형식)
        output = pd.DataFrame({
            "fname": test_data['fname'],
            "summary": preprocessed_summary,
        })
        
        # 10. 파일 저장
        submission_path = self._save_results(output)
        
        return str(submission_path)
    
    def _load_model_and_tokenizer(self, device):
        """모델과 토크나이저 로드 - 다중 모델 지원"""
        from transformers import AutoTokenizer
        
        # 모델명에서 원본 토크나이저 경로 추출
        model_name = self.config['general']['model_name']
        
        # 토크나이저 로드 (원본 모델에서)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 모델 로드 (체크포인트에서)
        model, _ = ModelSpecificHandler.load_model_for_inference(
            self.model_path,
            self.model_config,
            device
        )
        
        return model, tokenizer
    
    def _prepare_test_data(self, preprocessor, tokenizer, test_file):
        """테스트 데이터 준비 - prefix 처리 포함"""
        # 기본 데이터 로드
        test_df = pd.read_csv(test_file)
        test_data = pd.DataFrame({
            'fname': test_df['fname'],
            'dialogue': test_df['dialogue']
        })
        
        # 모델별 prefix 처리
        if self.model_config.get('use_prefix', False):
            prefix = self.model_config.get('prefix', '')
            if prefix:
                test_data['dialogue'] = test_data['dialogue'].apply(
                    lambda x: f"{prefix}{x}"
                )
                print(f"Applied prefix: {prefix[:50]}...")
        
        # baseline.py 스타일로 데이터 준비
        test_id = test_data['fname']
        
        print('-'*150)
        print(f'test_data:\n{test_data["dialogue"][0]}')
        print('-'*150)
        
        # 입력 생성
        encoder_input_test, decoder_input_test = preprocessor.make_input(
            test_data, is_test=True
        )
        print('-'*10, 'Load data complete', '-'*10)
        
        # 토큰화
        test_tokenized_encoder_inputs = tokenizer(
            encoder_input_test,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
            truncation=True,
            max_length=self.config['tokenizer']['encoder_max_len'],
            return_token_type_ids=False
        )
        
        # Dataset 생성
        test_encoder_inputs_dataset = DatasetForInference(
            test_tokenized_encoder_inputs,
            test_id.tolist(),
            len(encoder_input_test)
        )
        
        print('-'*10, 'Make dataset complete', '-'*10)
        
        return test_data, test_encoder_inputs_dataset
    
    def _save_results(self, output_df: pd.DataFrame) -> Path:
        """결과 저장 - baseline.py와 동일한 구조"""
        # 타임스탬프 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 저장 경로 설정
        result_path = path_manager.ensure_dir("prediction")
        
        # 실험별 폴더 생성
        exp_folder = result_path / f"{self.experiment_name}_{timestamp}"
        exp_folder.mkdir(parents=True, exist_ok=True)
        
        # output.csv 저장
        output_path = exp_folder / "output.csv"
        output_df.to_csv(output_path, index=False)
        print(f"Saved submission to: {output_path}")
        
        # latest_output.csv 업데이트
        latest_path = result_path / "latest_output.csv"
        output_df.to_csv(latest_path, index=False)
        print(f"Updated latest submission: {latest_path}")
        
        return output_path


# CLI 지원 (선택사항)
if __name__ == "__main__":
    import argparse
    from utils import load_config
    
    parser = argparse.ArgumentParser(description="학습 후 추론 실행")
    parser.add_argument("--experiment", required=True, help="실험명")
    parser.add_argument("--checkpoint", required=True, help="체크포인트 경로")
    parser.add_argument("--config", required=True, help="설정 파일 경로")
    parser.add_argument("--test-file", default="data/test.csv", help="테스트 파일")
    
    args = parser.parse_args()
    
    # 설정 로드
    config = load_config(args.config)
    
    # 추론 실행
    inference = PostTrainingInference(
        experiment_name=args.experiment,
        model_path=args.checkpoint,
        config=config
    )
    
    submission_path = inference.run_test_inference(args.test_file)
    print(f"Inference completed: {submission_path}")
```

## 3. 테스트 시나리오

### 3.1 단위 테스트
```python
# test_inference_integration.py

def test_baseline_preprocess():
    """BaselinePreprocess 클래스 테스트"""
    preprocessor = BaselinePreprocess("<s>", "</s>")
    # 테스트 데이터로 검증
    
def test_model_handler():
    """ModelSpecificHandler 테스트"""
    # 각 모델 타입별 설정 검증
    
def test_inference_flow():
    """전체 추론 플로우 테스트"""
    # 더미 데이터로 전체 과정 검증
```

### 3.2 통합 테스트
1. KoBART 모델로 테스트
2. mT5 모델로 테스트  
3. T5 모델로 테스트
4. 결과 비교

## 4. 검증 체크리스트

- [ ] baseline.py와 동일한 전처리 과정
- [ ] 동일한 Dataset 구조
- [ ] 동일한 generate 파라미터
- [ ] 동일한 특수 토큰 제거
- [ ] 동일한 출력 형식
- [ ] 모든 모델 타입 지원
- [ ] 에러 처리 및 복구
- [ ] GPU 메모리 관리
