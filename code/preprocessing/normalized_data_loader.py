"""
정규화된 데이터 로더 - 텍스트 정규화를 통합한 데이터 로딩

이 모듈은 데이터 로딩 시 자동으로 텍스트 정규화를 적용합니다.
"""

import pandas as pd
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from preprocessing import TextNormalizer

logger = logging.getLogger(__name__)


class NormalizedDataLoader:
    """텍스트 정규화가 통합된 데이터 로더"""
    
    def __init__(self, data_path: str, config: Dict):
        """
        정규화 데이터 로더 초기화
        
        Args:
            data_path: 데이터 디렉토리 경로
            config: 설정 딕셔너리
        """
        self.data_path = Path(data_path)
        self.config = config
        self.normalization_config = config.get('text_normalization', {})
        
        # 텍스트 정규화기 초기화
        if self.normalization_config.get('enabled', False):
            normalize_level = self.normalization_config.get('normalize_level', 'medium')
            self.normalizer = TextNormalizer(normalize_level)
            logger.info(f"텍스트 정규화 활성화: {normalize_level} 레벨")
        else:
            self.normalizer = None
            logger.info("텍스트 정규화 비활성화")
    
    def load_and_normalize_data(self, filename: str, is_training: bool = True) -> pd.DataFrame:
        """
        데이터 로드 및 정규화
        
        Args:
            filename: 파일명
            is_training: 학습 데이터 여부
            
        Returns:
            정규화된 DataFrame
        """
        # 데이터 로드
        file_path = self.data_path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없음: {file_path}")
        
        df = pd.read_csv(file_path)
        logger.info(f"{filename}에서 {len(df)}개 샘플 로드")
        
        # 정규화 적용
        if self.normalizer and is_training:
            df = self._apply_normalization(df)
        
        return df
    
    def _apply_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        DataFrame에 텍스트 정규화 적용
        
        Args:
            df: 입력 DataFrame
            
        Returns:
            정규화된 DataFrame
        """
        logger.info("텍스트 정규화 적용 중...")
        start_time = time.time()
        
        # 대화문 정규화
        original_dialogues = df['dialogue'].tolist()
        normalized_dialogues = []
        
        # 통계 수집
        total_stats = {
            'total_original_tokens': 0,
            'total_normalized_tokens': 0,
            'total_length_reduction': 0,
            'samples_changed': 0
        }
        
        # 샘플 로깅을 위한 리스트
        sample_logs = []
        log_sample_count = self.normalization_config.get('log_sample_count', 10)
        
        for i, dialogue in enumerate(original_dialogues):
            # 정규화 수행
            normalized = self.normalizer.normalize(dialogue)
            normalized_dialogues.append(normalized)
            
            # 통계 수집
            stats = self.normalizer.get_normalization_stats(dialogue, normalized)
            total_stats['total_original_tokens'] += stats['original_tokens']
            total_stats['total_normalized_tokens'] += stats['normalized_tokens']
            total_stats['total_length_reduction'] += stats['length_reduction']
            
            if dialogue != normalized:
                total_stats['samples_changed'] += 1
            
            # 샘플 로깅
            if i < log_sample_count and self.normalization_config.get('log_samples', True):
                sample_logs.append({
                    'index': i,
                    'original': dialogue[:200] + '...' if len(dialogue) > 200 else dialogue,
                    'normalized': normalized[:200] + '...' if len(normalized) > 200 else normalized,
                    'token_reduction': stats['token_reduction'],
                    'reduction_percentage': stats['reduction_percentage']
                })
        
        # 정규화된 데이터로 업데이트
        df['dialogue'] = normalized_dialogues
        
        # 요약문도 간단히 정리 (약한 레벨로)
        if 'summary' in df.columns:
            light_normalizer = TextNormalizer('light')
            df['summary'] = df['summary'].apply(lambda x: light_normalizer.normalize(str(x)))
        
        elapsed_time = time.time() - start_time
        
        # 통계 계산
        total_stats['change_percentage'] = (total_stats['samples_changed'] / len(df)) * 100
        total_stats['avg_token_reduction'] = (
            total_stats['total_original_tokens'] - total_stats['total_normalized_tokens']
        ) / len(df)
        total_stats['processing_time'] = elapsed_time
        
        # 로깅
        logger.info(f"정규화 완료: {elapsed_time:.2f}초")
        logger.info(f"변경된 샘플: {total_stats['samples_changed']}/{len(df)} "
                   f"({total_stats['change_percentage']:.1f}%)")
        logger.info(f"평균 토큰 감소: {total_stats['avg_token_reduction']:.1f} 토큰/샘플")
        
        # 샘플 로그 출력
        if sample_logs:
            logger.info("\n=== 정규화 샘플 ===")
            for log in sample_logs[:3]:  # 처음 3개만 로그에 출력
                logger.info(f"\n샘플 {log['index']}:")
                logger.info(f"원본: {log['original']}")
                logger.info(f"정규화: {log['normalized']}")
                logger.info(f"토큰 감소: {log['token_reduction']} ({log['reduction_percentage']:.1f}%)")
        
        # 정규화된 데이터 저장 (옵션)
        if self.normalization_config.get('save_normalized_data', False):
            self._save_normalized_data(df, total_stats, sample_logs)
        
        return df
    
    def _save_normalized_data(self, df: pd.DataFrame, stats: Dict, samples: List[Dict]):
        """정규화된 데이터 및 통계 저장"""
        output_path = Path(self.normalization_config.get('normalized_data_path', 'outputs/normalized_data'))
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 데이터 저장
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        df.to_csv(output_path / f'normalized_data_{timestamp}.csv', index=False)
        
        # 통계 저장
        stats_data = {
            'timestamp': timestamp,
            'normalize_level': self.normalization_config.get('normalize_level', 'medium'),
            'statistics': stats,
            'sample_logs': samples
        }
        
        with open(output_path / f'normalization_stats_{timestamp}.json', 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"정규화된 데이터 저장: {output_path}")
    
    def load_train_data(self) -> pd.DataFrame:
        """학습 데이터 로드 및 정규화"""
        return self.load_and_normalize_data(
            self.config.get('data', {}).get('train_file', 'train.csv'),
            is_training=True
        )
    
    def load_dev_data(self) -> pd.DataFrame:
        """개발 데이터 로드 (정규화 없음)"""
        return self.load_and_normalize_data(
            self.config.get('data', {}).get('dev_file', 'dev.csv'),
            is_training=False
        )
    
    def load_test_data(self) -> pd.DataFrame:
        """테스트 데이터 로드 (정규화 없음)"""
        return self.load_and_normalize_data(
            self.config.get('data', {}).get('test_file', 'test.csv'),
            is_training=False
        )


# 사용 예제
if __name__ == "__main__":
    # 테스트 설정
    test_config = {
        'data': {
            'train_file': 'train.csv',
            'dev_file': 'dev.csv',
            'test_file': 'test.csv'
        },
        'text_normalization': {
            'enabled': True,
            'normalize_level': 'medium',
            'log_samples': True,
            'log_sample_count': 5,
            'save_normalized_data': True,
            'normalized_data_path': 'outputs/test_normalized'
        }
    }
    
    # 데이터 로더 테스트
    loader = NormalizedDataLoader('../data', test_config)
    
    print("정규화된 데이터 로더 테스트")
    print("="*50)
    
    try:
        # 실제 데이터 파일이 없으면 에러가 발생합니다
        train_df = loader.load_train_data()
        print(f"학습 데이터 로드 완료: {len(train_df)} 샘플")
    except FileNotFoundError as e:
        print(f"예상된 에러: {e}")
