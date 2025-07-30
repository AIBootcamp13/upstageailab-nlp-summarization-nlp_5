"""
테스트 예측 및 CSV 생성을 위한 추가 메서드들
"""

import pandas as pd
from datetime import datetime
from tqdm import tqdm
import torch
from pathlib import Path


def generate_test_predictions(self, test_dataset) -> pd.DataFrame:
    """
    테스트 데이터셋에 대한 예측 생성
    
    Args:
        test_dataset: 테스트 데이터셋
        
    Returns:
        예측 결과 DataFrame (fname, summary 커럼)
    """
    self.model.eval()
    predictions = []
    fnames = []
    
    # 테스트 데이터에서 fname 추출
    if hasattr(test_dataset, 'fname'):
        fnames = test_dataset.fname
    elif 'fname' in test_dataset.features:
        fnames = test_dataset['fname']
    else:
        # fname이 없으면 인덱스 사용
        fnames = [f"test_{i:04d}" for i in range(len(test_dataset))]
    
    # 생성 설정
    gen_config = self.config['generation']
    
    # 배치 단위 예측
    batch_size = self.config['training'].get('per_device_eval_batch_size', 8)
    
    with torch.no_grad():
        for i in tqdm(range(0, len(test_dataset), batch_size), desc="Generating predictions"):
            batch = test_dataset[i:i + batch_size]
            
            # 토크나이징
            inputs = self.tokenizer(
                batch['input'],
                max_length=self.config['tokenizer']['encoder_max_len'],
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # 생성
            outputs = self.model.generate(
                **inputs,
                max_length=gen_config['max_length'],
                min_length=gen_config.get('min_length', 30),
                num_beams=gen_config['num_beams'],
                length_penalty=gen_config.get('length_penalty', 1.0),
                no_repeat_ngram_size=gen_config.get('no_repeat_ngram_size', 2),
                early_stopping=gen_config.get('early_stopping', True),
                do_sample=gen_config.get('do_sample', False),
                temperature=gen_config.get('temperature', 1.0) if gen_config.get('do_sample') else None,
                top_k=gen_config.get('top_k', 50) if gen_config.get('do_sample') else None,
                top_p=gen_config.get('top_p', 0.95) if gen_config.get('do_sample') else None
            )
            
            # 디코딩
            batch_predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(batch_predictions)
    
    # DataFrame 생성
    submission_df = pd.DataFrame({
        'fname': fnames[:len(predictions)],
        'summary': predictions
    })
    
    return submission_df


def _save_submission_csv(self, predictions_df: pd.DataFrame) -> Path:
    """
    예측 결과를 CSV 파일로 저장
    
    Args:
        predictions_df: 예측 결과 DataFrame
        
    Returns:
        저장된 파일 경로
    """
    # 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = self._get_model_architecture().replace('/', '_')
    filename = f"submission_{model_name}_{timestamp}.csv"
    
    # 저장 경로
    submission_path = self.output_dir / filename
    
    # CSV 저장
    predictions_df.to_csv(submission_path, index=False, encoding='utf-8')
    
    # 복사본을 최상위 디렉토리에도 저장
    latest_submission_path = self.output_dir.parent / f"submission_latest_{model_name}.csv"
    predictions_df.to_csv(latest_submission_path, index=False, encoding='utf-8')
    
    # 통계 출력
    self.logger.info(f"\n=== Submission Statistics ===")
    self.logger.info(f"Total samples: {len(predictions_df)}")
    self.logger.info(f"Average summary length: {predictions_df['summary'].str.len().mean():.1f}")
    self.logger.info(f"Min summary length: {predictions_df['summary'].str.len().min()}")
    self.logger.info(f"Max summary length: {predictions_df['summary'].str.len().max()}")
    
    return submission_path
