"""
Augmented Data Loader for Dialogue Summarization

This module extends the basic data loading to include data augmentation
"""

import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
import json
import time

from data_augmentation import DialogueAugmenter

logger = logging.getLogger(__name__)


class AugmentedDataLoader:
    """Data loader with augmentation capabilities"""
    
    def __init__(self, data_path: str, config: Dict):
        """
        Initialize the augmented data loader
        
        Args:
            data_path: Path to the data directory
            config: Configuration dictionary
        """
        self.data_path = Path(data_path)
        self.config = config
        self.augmentation_config = config.get('data_augmentation', {})
        
        # 활성화된 경우 증강기 초기화
        if self.augmentation_config.get('enabled', False):
            self.augmenter = DialogueAugmenter(self.augmentation_config)
            logger.info("Data augmentation enabled with ratio: %.2f", 
                       self.augmentation_config.get('augmentation_ratio', 0.2))
        else:
            self.augmenter = None
            logger.info("Data augmentation disabled")
    
    def load_data(self, filename: str) -> pd.DataFrame:
        """Load data from CSV file"""
        file_path = self.data_path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} samples from {filename}")
        
        return df
    
    def augment_data(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Apply data augmentation to the dataset
        
        Args:
            df: Input dataframe with 'dialogue' and 'summary' columns
            is_training: Whether this is training data (only augment training)
            
        Returns:
            Augmented dataframe
        """
        if not is_training or self.augmenter is None:
            return df
        
        logger.info("Applying data augmentation...")
        start_time = time.time()
        
        # 대화와 요약 추출
        dialogues = df['dialogue'].tolist()
        summaries = df['summary'].tolist()
        
        # 증강 적용
        aug_dialogues, aug_summaries = self.augmenter.augment_dataset(dialogues, summaries)
        
        # 통계 계산
        original_size = len(dialogues)
        augmented_size = len(aug_dialogues)
        stats = self.augmenter.get_augmentation_stats(original_size, augmented_size)
        
        # 증강된 데이터프레임 생성
        aug_df = pd.DataFrame({
            'dialogue': aug_dialogues,
            'summary': aug_summaries
        })
        
        # 다른 컴럼이 있는 경우 추가
        if 'fname' in df.columns:
            # 증강된 샘플을 위한 새 파일명 생성
            original_fnames = df['fname'].tolist()
            aug_fnames = original_fnames.copy()
            
            for i in range(original_size, augmented_size):
                base_idx = (i - original_size) % original_size
                aug_fnames.append(f"{original_fnames[base_idx]}_aug_{i-original_size}")
            
            aug_df['fname'] = aug_fnames
        
        # 토픽이 있는 경우 추가
        if 'topic' in df.columns:
            original_topics = df['topic'].tolist()
            aug_topics = original_topics.copy()
            
            # 증강된 샘플을 위한 토픽 복제
            for i in range(original_size, augmented_size):
                base_idx = (i - original_size) % original_size
                aug_topics.append(original_topics[base_idx])
            
            aug_df['topic'] = aug_topics
        
        elapsed_time = time.time() - start_time
        
        # 통계 로깅
        logger.info("Augmentation completed in %.2f seconds", elapsed_time)
        logger.info("Original samples: %d", stats['original_size'])
        logger.info("Augmented samples: %d", stats['augmented_size'])
        logger.info("Augmentation ratio: %.2f%%", stats['augmentation_ratio'] * 100)
        logger.info("Total added: %d", stats['total_added'])
        
        # 증강 통계 저장
        stats_file = Path(self.augmentation_config.get('cache_dir', 'outputs/augmented_data_cache')) / 'augmentation_stats.json'
        stats_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(stats_file, 'w') as f:
            json.dump({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'elapsed_time_seconds': elapsed_time,
                **stats
            }, f, indent=2)
        
        return aug_df
    
    def load_train_data(self) -> pd.DataFrame:
        """Load and augment training data"""
        df = self.load_data(self.config.get('data', {}).get('train_file', 'train.csv'))
        df = self.augment_data(df, is_training=True)
        return df
    
    def load_dev_data(self) -> pd.DataFrame:
        """Load development data (no augmentation)"""
        return self.load_data(self.config.get('data', {}).get('dev_file', 'dev.csv'))
    
    def load_test_data(self) -> pd.DataFrame:
        """Load test data (no augmentation)"""
        return self.load_data(self.config.get('data', {}).get('test_file', 'test.csv'))


def integrate_augmentation_with_trainer(config: Dict) -> Dict:
    """
    Helper function to integrate augmentation with existing trainer
    
    Args:
        config: Original configuration
        
    Returns:
        Modified configuration with augmentation hooks
    """
    if config.get('data_augmentation', {}).get('enabled', False):
        # 증강 전용 설정 추가
        config['training']['run_name'] = config['training'].get('run_name', '') + '_augmented'
        config['training']['output_dir'] = config['training'].get('output_dir', './output') + '_augmented'
        
        # 캐시 디렉토리 존재 확인
        cache_dir = config['data_augmentation'].get('cache_dir', 'outputs/augmented_data_cache')
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("Augmentation integration configured")
    
    return config


# 사용 예제
if __name__ == "__main__":
    # 테스트 설정
    test_config = {
        'data': {
            'train_file': 'train.csv',
            'dev_file': 'dev.csv',
            'test_file': 'test.csv'
        },
        'data_augmentation': {
            'enabled': True,
            'augmentation_ratio': 0.2,
            'seed': 42,
            'synonym_replacement': {
                'enabled': True,
                'num_replacements': 3
            },
            'sentence_reorder': {
                'enabled': True,
                'max_distance': 2
            }
        }
    }
    
    # 데이터 로더 테스트
    loader = AugmentedDataLoader('../data', test_config)
    
    try:
        # 실제 데이터 파일 없이는 실패하지만, 사용법을 보여줌
        train_df = loader.load_train_data()
        print(f"Loaded {len(train_df)} training samples (with augmentation)")
    except FileNotFoundError as e:
        print(f"Expected error in test: {e}")
