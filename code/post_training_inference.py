#!/usr/bin/env python3
"""
학습 후 자동 추론 스크립트

학습이 완료된 후 test.csv에 대한 예측을 수행하고 제출 파일을 생성합니다.
"""

import os
import sys
import json
import logging
from pathlib import Path
import pandas as pd
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.inference import InferenceEngine, InferenceConfig
from utils.path_utils import path_manager
from utils import load_config

logger = logging.getLogger(__name__)


class PostTrainingInference:
    """학습 후 자동 추론 클래스"""
    
    def __init__(self, experiment_name: str, model_path: str, config: dict):
        """
        Args:
            experiment_name: 실험명
            model_path: 학습된 모델 경로
            config: 실험 설정
        """
        self.experiment_name = experiment_name
        self.model_path = model_path
        self.config = config
        
    def run_test_inference(self, test_file: str = "data/test.csv") -> str:
        """
        테스트 데이터에 대한 추론 수행
        
        Args:
            test_file: 테스트 데이터 파일 경로
            
        Returns:
            생성된 제출 파일 경로
        """
        logger.info(f"Starting test inference for {self.experiment_name}")
        
        # 테스트 파일 경로 확인
        test_path = path_manager.resolve_path(test_file)
        if not test_path.exists():
            raise FileNotFoundError(f"Test file not found: {test_path}")
        
        # 추론 설정
        inference_config = InferenceConfig(
            model_path=self.model_path,
            batch_size=self.config.get('inference', {}).get('batch_size', 8),
            max_source_length=self.config.get('tokenizer', {}).get('encoder_max_len', 512),
            max_target_length=self.config.get('tokenizer', {}).get('decoder_max_len', 100),
            num_beams=self.config.get('inference', {}).get('num_beams', 4),
            device=self.config.get('device', None),
            fp16=self.config.get('training', {}).get('fp16', False)
        )
        
        # 추론 엔진 생성
        engine = InferenceEngine(inference_config)
        
        # 테스트 데이터 로드
        test_df = pd.read_csv(test_path)
        logger.info(f"Loaded {len(test_df)} test samples")
        
        # 추론 실행
        result_df = engine.predict_from_dataframe(
            test_df,
            dialogue_column='dialogue',
            output_column='summary',
            show_progress=True
        )
        
        # 제출 파일 경로 생성
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = path_manager.ensure_dir("outputs/submissions")
        output_file = output_dir / f"{self.experiment_name}_{timestamp}.csv"
        
        # 제출 형식으로 저장
        submission_df = result_df[['fname', 'summary']].copy()
        submission_df.to_csv(output_file, index=False, encoding='utf-8')
        
        logger.info(f"Submission file saved: {output_file}")
        
        # 샘플 출력
        logger.info("\n--- Sample predictions ---")
        for idx, row in submission_df.head(3).iterrows():
            logger.info(f"[{idx+1}] {row['fname']}: {row['summary'][:100]}...")
        
        return str(output_file)


def generate_submission_after_training(experiment_name: str, model_path: str, 
                                     config_path: str = None, config_dict: dict = None):
    """
    학습 완료 후 제출 파일 생성
    
    Args:
        experiment_name: 실험명
        model_path: 학습된 모델 경로
        config_path: 설정 파일 경로
        config_dict: 설정 딕셔너리 (config_path 대신 사용 가능)
        
    Returns:
        생성된 제출 파일 경로
    """
    # 설정 로드
    if config_dict:
        config = config_dict
    elif config_path:
        config = load_config(config_path)
    else:
        raise ValueError("Either config_path or config_dict must be provided")
    
    # 추론 실행
    inferencer = PostTrainingInference(experiment_name, model_path, config)
    submission_path = inferencer.run_test_inference()
    
    return submission_path


if __name__ == "__main__":
    # 테스트용 메인 함수
    import argparse
    
    parser = argparse.ArgumentParser(description="Run inference after training")
    parser.add_argument("--experiment", required=True, help="Experiment name")
    parser.add_argument("--model", required=True, help="Model path")
    parser.add_argument("--config", required=True, help="Config file path")
    parser.add_argument("--test", default="data/test.csv", help="Test data path")
    
    args = parser.parse_args()
    
    submission_path = generate_submission_after_training(
        experiment_name=args.experiment,
        model_path=args.model,
        config_path=args.config
    )
    
    print(f"Submission file created: {submission_path}")
