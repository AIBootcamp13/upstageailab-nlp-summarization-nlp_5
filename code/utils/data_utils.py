"""
데이터 처리 유틸리티

NLP 대화 요약 프로젝트를 위한 데이터 전처리, 후처리, 변환 기능을 제공합니다.
기존 baseline.ipynb의 데이터 처리 로직을 모듈화하고 확장했습니다.
"""

import re
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass
from transformers import PreTrainedTokenizer
import torch
from torch.utils.data import Dataset
from .path_utils import PathManager, path_manager


@dataclass
class DataSample:
    """데이터 샘플 클래스"""
    dialogue: str
    summary: str
    fname: str
    dialogue_length: int = 0
    summary_length: int = 0
    
    def __post_init__(self) -> None:
        self.dialogue_length = len(self.dialogue)
        self.summary_length = len(self.summary)


class TextPreprocessor:
    """
    텍스트 전처리기
    
    한국어 대화 텍스트의 정규화, 정제, 특수 토큰 처리 등을 담당합니다.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        TextPreprocessor 초기화
        
        Args:
            config: 전처리 설정
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 특수 토큰 설정
        self.special_tokens = [
            '#Person1#', '#Person2#', '#Person3#', '#Person4#', 
            '#Person5#', '#Person6#', '#Person7#',
            '#PhoneNumber#', '#Address#', '#PassportNumber#'
        ]
        
        # 정규 표현식 패턴
        self._compile_patterns()
    
    def _compile_patterns(self) -> None:
        """정규 표현식 패턴 컴파일"""
        # 개행 문자 변형 패턴
        self.newline_pattern = re.compile(r'\\n')
        
        # HTML 태그 패턴
        self.html_pattern = re.compile(r'<[^>]+>')
        
        # 연속 공백 패턴
        self.whitespace_pattern = re.compile(r'\s+')
        
        # 특수 문자 정규화 패턴
        self.quote_pattern = re.compile(r'["""]')
        self.dash_pattern = re.compile(r'[―—–-]')
        
        # 화자 구분 패턴
        self.speaker_pattern = re.compile(r'#Person(\d+)#\s*:\s*')
    
    def preprocess_text(self, text: str, 
                       normalize_quotes: bool = True,
                       normalize_whitespace: bool = True,
                       remove_html: bool = True) -> str:
        """
        텍스트 전처리
        
        Args:
            text: 입력 텍스트
            normalize_quotes: 따옴표 정규화 여부
            normalize_whitespace: 공백 정규화 여부
            remove_html: HTML 태그 제거 여부
            
        Returns:
            전처리된 텍스트
        """
        if not text or not isinstance(text, str):
            return ""
        
        # 개행 문자 변형 처리
        text = self.newline_pattern.sub('\n', text)
        
        # HTML 태그 제거
        if remove_html:
            text = self.html_pattern.sub('', text)
        
        # 따옴표 정규화
        if normalize_quotes:
            text = self.quote_pattern.sub('"', text)
        
        # 대시 정규화
        text = self.dash_pattern.sub('-', text)
        
        # 공백 정규화
        if normalize_whitespace:
            text = self.whitespace_pattern.sub(' ', text)
        
        # 앞뒤 공백 제거
        text = text.strip()
        
        return text
    
    def extract_speakers(self, dialogue: str) -> List[str]:
        """
        대화에서 화자 목록 추출
        
        Args:
            dialogue: 대화 텍스트
            
        Returns:
            화자 목록
        """
        speakers = self.speaker_pattern.findall(dialogue)
        return [f"#Person{speaker}#" for speaker in sorted(set(speakers))]
    
    def count_turns(self, dialogue: str) -> int:
        """
        대화 턴 수 계산
        
        Args:
            dialogue: 대화 텍스트
            
        Returns:
            턴 수
        """
        return len(self.speaker_pattern.findall(dialogue))
    
    def clean_dialogue(self, dialogue: str) -> str:
        """
        대화 텍스트 정제
        
        Args:
            dialogue: 원본 대화 텍스트
            
        Returns:
            정제된 대화 텍스트
        """
        # 기본 전처리
        dialogue = self.preprocess_text(dialogue)
        
        # 화자 구분 형식 표준화
        dialogue = self.speaker_pattern.sub(r'#Person\1#: ', dialogue)
        
        return dialogue
    
    def clean_summary(self, summary: str) -> str:
        """
        요약문 정제
        
        Args:
            summary: 원본 요약문
            
        Returns:
            정제된 요약문
        """
        # 기본 전처리
        summary = self.preprocess_text(summary)
        
        # 요약문에는 화자 구분 불필요하므로 제거
        summary = self.speaker_pattern.sub('', summary)
        
        return summary


class DataProcessor:
    """
    데이터 프로세서
    
    CSV 파일 로딩, 데이터 필터링, 분할, 통계 분석 등을 담당합니다.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        DataProcessor 초기화
        
        Args:
            config: 데이터 처리 설정
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.preprocessor = TextPreprocessor(config)
        
        # 데이터 필터 설정
        self.min_dialogue_length = self.config.get('min_source_length', 10)
        self.max_dialogue_length = self.config.get('max_source_length', 1024)
        self.min_summary_length = self.config.get('min_target_length', 5)
        self.max_summary_length = self.config.get('max_target_length', 256)
    def load_dataset(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        데이터셋 로딩
        
        Args:
            file_path: CSV 파일 경로 (상대 경로 또는 절대 경로)
            
        Returns:
            로딩된 데이터프레임
        """
        # 경로 관리자를 통해 경로 해결
        file_path = path_manager.resolve_path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            self.logger.info(f"Loaded dataset: {len(df)} samples from {file_path}")
            
            # 필수 컬럼 확인
            required_columns = ['fname', 'dialogue', 'summary']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            raise
    
    def preprocess_dataset(self, df: pd.DataFrame, 
                          clean_text: bool = True,
                          filter_data: bool = True) -> pd.DataFrame:
        """
        데이터셋 전처리
        
        Args:
            df: 원본 데이터프레임
            clean_text: 텍스트 정제 여부
            filter_data: 데이터 필터링 여부
            
        Returns:
            전처리된 데이터프레임
        """
        df = df.copy()
        
        # 결측값 제거
        initial_count = len(df)
        df = df.dropna(subset=['dialogue', 'summary'])
        after_na_count = len(df)
        
        if initial_count > after_na_count:
            self.logger.info(f"Removed {initial_count - after_na_count} samples with missing values")
        
        # 텍스트 정제
        if clean_text:
            self.logger.info("Cleaning text data...")
            df['dialogue'] = df['dialogue'].apply(self.preprocessor.clean_dialogue)
            df['summary'] = df['summary'].apply(self.preprocessor.clean_summary)
        
        # 길이 계산
        df['dialogue_length'] = df['dialogue'].str.len()
        df['summary_length'] = df['summary'].str.len()
        df['turn_count'] = df['dialogue'].apply(self.preprocessor.count_turns)
        
        # 데이터 필터링
        if filter_data:
            df = self._filter_by_length(df)
        
        # 통계 정보 로깅
        self._log_dataset_stats(df)
        
        return df
    
    def _filter_by_length(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        길이 기반 데이터 필터링
        
        Args:
            df: 데이터프레임
            
        Returns:
            필터링된 데이터프레임
        """
        initial_count = len(df)
        
        # 대화 길이 필터링
        df = df[
            (df['dialogue_length'] >= self.min_dialogue_length) &
            (df['dialogue_length'] <= self.max_dialogue_length)
        ]
        
        # 요약문 길이 필터링
        df = df[
            (df['summary_length'] >= self.min_summary_length) &
            (df['summary_length'] <= self.max_summary_length)
        ]
        
        # 빈 텍스트 제거
        df = df[df['dialogue'].str.strip().astype(bool)]
        df = df[df['summary'].str.strip().astype(bool)]
        
        final_count = len(df)
        filtered_count = initial_count - final_count
        
        if filtered_count > 0:
            self.logger.info(f"Filtered out {filtered_count} samples based on length constraints")
            self.logger.info(f"Remaining samples: {final_count}")
        
        return df
    
    def _log_dataset_stats(self, df: pd.DataFrame):
        """
        데이터셋 통계 로깅
        
        Args:
            df: 데이터프레임
        """
        stats = {
            'total_samples': len(df),
            'dialogue_length': {
                'mean': df['dialogue_length'].mean(),
                'std': df['dialogue_length'].std(),
                'min': df['dialogue_length'].min(),
                'max': df['dialogue_length'].max()
            },
            'summary_length': {
                'mean': df['summary_length'].mean(),
                'std': df['summary_length'].std(),
                'min': df['summary_length'].min(),
                'max': df['summary_length'].max()
            },
            'turn_count': {
                'mean': df['turn_count'].mean(),
                'std': df['turn_count'].std(),
                'min': df['turn_count'].min(),
                'max': df['turn_count'].max()
            }
        }
        
        self.logger.info("Dataset Statistics:")
        self.logger.info(f"  Total samples: {stats['total_samples']}")
        self.logger.info(f"  Dialogue length - Mean: {stats['dialogue_length']['mean']:.1f}, "
                        f"Std: {stats['dialogue_length']['std']:.1f}, "
                        f"Range: [{stats['dialogue_length']['min']}, {stats['dialogue_length']['max']}]")
        self.logger.info(f"  Summary length - Mean: {stats['summary_length']['mean']:.1f}, "
                        f"Std: {stats['summary_length']['std']:.1f}, "
                        f"Range: [{stats['summary_length']['min']}, {stats['summary_length']['max']}]")
        self.logger.info(f"  Turn count - Mean: {stats['turn_count']['mean']:.1f}, "
                        f"Std: {stats['turn_count']['std']:.1f}, "
                        f"Range: [{stats['turn_count']['min']}, {stats['turn_count']['max']}]")
    
    def create_data_samples(self, df: pd.DataFrame) -> List[DataSample]:
        """
        데이터프레임을 DataSample 객체 리스트로 변환
        
        Args:
            df: 데이터프레임
            
        Returns:
            DataSample 객체 리스트
        """
        samples = []
        
        for _, row in df.iterrows():
            sample = DataSample(
                dialogue=row['dialogue'],
                summary=row['summary'],
                fname=row['fname'],
                dialogue_length=row.get('dialogue_length', len(row['dialogue'])),
                summary_length=row.get('summary_length', len(row['summary']))
            )
            samples.append(sample)
        
        return samples
    
    def split_dataset(self, df: pd.DataFrame, 
                     train_ratio: float = 0.8,
                     val_ratio: float = 0.1,
                     test_ratio: float = 0.1,
                     random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        데이터셋 분할
        
        Args:
            df: 데이터프레임
            train_ratio: 훈련 데이터 비율
            val_ratio: 검증 데이터 비율
            test_ratio: 테스트 데이터 비율
            random_state: 랜덤 시드
            
        Returns:
            (train_df, val_df, test_df) 튜플
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        # 데이터 셔플
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        n_total = len(df)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_df = df[:n_train]
        val_df = df[n_train:n_train + n_val]
        test_df = df[n_train + n_val:]
        
        self.logger.info(f"Dataset split - Train: {len(train_df)}, "
                        f"Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df


class DialogueSummarizationDataset(Dataset):
    """
    대화 요약 데이터셋 클래스
    
    PyTorch Dataset을 상속하여 배치 단위 데이터 로딩을 지원합니다.
    """
    
    def __init__(self, data_samples: List[DataSample], 
                 tokenizer: PreTrainedTokenizer,
                 max_source_length: int = 512,
                 max_target_length: int = 128,
                 prefix: str = ""):
        """
        DialogueSummarizationDataset 초기화
        
        Args:
            data_samples: DataSample 객체 리스트
            tokenizer: 토크나이저
            max_source_length: 최대 입력 길이
            max_target_length: 최대 출력 길이
            prefix: 입력 프리픽스 (T5 등에서 사용)
        """
        self.data_samples = data_samples
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.prefix = prefix
        
        # 특수 토큰 추가
        self._add_special_tokens()
    
    def _add_special_tokens(self):
        """특수 토큰을 토크나이저에 추가"""
        special_tokens = [
            '#Person1#', '#Person2#', '#Person3#', '#Person4#',
            '#Person5#', '#Person6#', '#Person7#',
            '#PhoneNumber#', '#Address#', '#PassportNumber#'
        ]
        
        # 기존에 없는 토큰만 추가
        new_tokens = [token for token in special_tokens 
                     if token not in self.tokenizer.get_vocab()]
        
        if new_tokens:
            self.tokenizer.add_tokens(new_tokens)
    
    def __len__(self) -> int:
        """데이터셋 크기 반환"""
        return len(self.data_samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        인덱스에 해당하는 데이터 샘플 반환
        
        Args:
            idx: 데이터 인덱스
            
        Returns:
            토크나이징된 데이터 딕셔너리
        """
        sample = self.data_samples[idx]
        
        # 입력 텍스트 준비
        source_text = self.prefix + sample.dialogue
        target_text = sample.summary
        
        # 토크나이징
        source_encoding = self.tokenizer(
            source_text,
            max_length=self.max_source_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 레이블 준비 (패딩 토큰은 -100으로)
        labels = target_encoding['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': source_encoding['input_ids'].squeeze(),
            'attention_mask': source_encoding['attention_mask'].squeeze(),
            'labels': labels.squeeze(),
            'fname': sample.fname
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    배치 데이터 정리 함수
    
    Args:
        batch: 배치 데이터 리스트
        
    Returns:
        정리된 배치 딕셔너리
    """
    # 텐서 데이터들 스택
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    # 문자열 데이터들 리스트로 유지
    fnames = [item['fname'] for item in batch]
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'fnames': fnames
    }
