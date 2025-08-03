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
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        # 모델명에서 원본 토크나이저 경로 추출
        model_name = self.config['general']['model_name']
        
        print('-'*10, f'Model Name : {model_name}', '-'*10)
        
        # baseline.py와 동일한 방식으로 로드
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 모델 로드 (체크포인트에서)
        model, _ = ModelSpecificHandler.load_model_for_inference(
            self.model_path,
            self.model_config,
            device
        )
        
        print('-'*10, 'Load tokenizer & model complete', '-'*10)
        
        return model, tokenizer
    
    def _prepare_test_data(self, preprocessor, tokenizer, test_file):
        """테스트 데이터 준비 - prefix 처리 포함"""
        # config 업데이트 (data_path 설정)
        config_copy = self.config.copy()
        test_path = Path(test_file)
        config_copy['general']['data_path'] = str(test_path.parent)
        
        # baseline.py의 prepare_test_dataset 호출
        test_data, test_encoder_inputs_dataset = prepare_test_dataset(
            config_copy, preprocessor, tokenizer
        )
        
        # 모델별 prefix 처리 - 설정 파일의 input_prefix 직접 사용
        input_prefix = self.config.get('input_prefix', '')
        if input_prefix and input_prefix.strip():  # 빈 문자열이 아닌 경우만
            # 이미 토크나이지된 데이터이므로 원본 데이터를 다시 처리
            test_df = pd.read_csv(test_file)
            dialogues = test_df['dialogue'].tolist()
            
            # prefix 추가
            dialogues_with_prefix = [f"{input_prefix}{d}" for d in dialogues]
            
            # 다시 토크나이지
            test_tokenized_encoder_inputs = tokenizer(
                dialogues_with_prefix,
                return_tensors="pt",
                padding=True,
                add_special_tokens=True,
                truncation=True,
                max_length=self.config['tokenizer']['encoder_max_len'],
                return_token_type_ids=False
            )
            
            # Dataset 재생성
            test_encoder_inputs_dataset = DatasetForInference(
                test_tokenized_encoder_inputs,
                test_data['fname'].tolist(),
                len(dialogues)
            )
            
            print(f"Applied prefix: '{input_prefix}' to {len(dialogues)} samples")
        
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
