"""
데이터 처리 유틸리티

NLP 대화 요약 프로젝트를 위한 데이터 전처리, 후처리, 변환 기능을 제공합니다.
기존 baseline.ipynb의 데이터 처리 로직을 모듈화하고 확장했습니다.
"""

import re
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from pathlib import Path
import logging
from dataclasses import dataclass
from transformers import PreTrainedTokenizer
import torch
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset
from .path_utils import PathManager, path_manager


logger = logging.getLogger(__name__)


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
    
    CSV/JSON 파일 로딩, 데이터 필터링, 토크나이징, HuggingFace Dataset 변환 등을 담당합니다.
    """
    
    def __init__(self, 
                 tokenizer: PreTrainedTokenizer, 
                 config: Optional[Dict[str, Any]] = None,
                 preprocessor: Optional[Callable] = None,
                 model: Optional[Any] = None):  # 모델 추가
        """
        DataProcessor 초기화
        
        Args:
            tokenizer: 사전 학습된 토크나이저
            config: 데이터 처리 설정
            preprocessor: 모델별 전처리 함수 (optional)
            model: 모델 객체 (embedding 크기 조정을 위해)
        """
        self.tokenizer = tokenizer
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.text_preprocessor = TextPreprocessor(config)
        self.model_preprocessor = preprocessor  # 모델별 전처리 함수
        self.model = model  # 모델 객체 저장
        
        # 토크나이저 설정
        self.encoder_max_len = self.config.get('tokenizer', {}).get('encoder_max_len', 512)
        self.decoder_max_len = self.config.get('tokenizer', {}).get('decoder_max_len', 128)
        # 데이터 필터 설정
        self.min_dialogue_length = self.config.get('data', {}).get('min_source_length', 10)
        self.max_dialogue_length = self.config.get('data', {}).get('max_source_length', 1024)
        self.min_summary_length = self.config.get('data', {}).get('min_target_length', 5)
        self.max_summary_length = self.config.get('data', {}).get('max_target_length', 256)
        
        # 특수 토큰 추가
        self._add_special_tokens()
        
        def _add_special_tokens(self):
        """특수 토큰을 토크나이저에 추가"""
        special_tokens = self.text_preprocessor.special_tokens
        
        # 모델 이름 확인
        model_name = self.config.get('general', {}).get('model_name', '')
        
        # KoBART 모델의 경우 특별 처리
        if "kobart" in model_name.lower() or "bart" in model_name.lower():
            logger.info("🔍 KoBART/BART 모델 감지: 특수 토큰 처리를 조정합니다.")
            
            # 모델이 있으면 안전한 토큰 추가 사용
            if self.model is not None:
                try:
                    from utils.model_utils import safe_add_special_tokens
                    self.tokenizer, self.model = safe_add_special_tokens(
                        self.tokenizer, self.model, special_tokens, model_name
                    )
                    return
                except ImportError:
                    logger.warning("⚠️ model_utils를 찾을 수 없습니다. 기본 처리를 사용합니다.")
            else:
                # 모델이 없으면 최소한의 토큰만 추가
                logger.warning("모델 객체가 없어 안전 모드로 특수 토큰을 추가합니다.")
                # 기본 PII 토큰만 추가
                safe_tokens = ['#PhoneNumber#', '#Address#', '#PassportNumber#']
                new_tokens = [token for token in safe_tokens if token not in self.tokenizer.get_vocab()]
                if new_tokens:
                    self.tokenizer.add_tokens(new_tokens)
                    logger.info(f"🔒 {len(new_tokens)}개의 기본 PII 토큰만 추가했습니다.")
                return
        
        # eenzeenee 모델의 경우 특별 처리
        if "eenzeenee" in model_name.lower():
            try:
                from utils.eenzeenee_utils import check_and_fix_special_tokens
                self.tokenizer = check_and_fix_special_tokens(
                    self.tokenizer, special_tokens, model_name
                )
                return
            except ImportError:
                logger.warning("⚠️ eenzeenee_utils를 찾을 수 없습니다. 기본 처리를 사용합니다.")
        
        # 기존에 없는 토큰만 추가
        new_tokens = [token for token in special_tokens 
                     if token not in self.tokenizer.get_vocab()]
        
        if new_tokens:
            self.tokenizer.add_tokens(new_tokens)
            logger.info(f"Added {len(new_tokens)} special tokens to tokenizer")
            logger.info(f"Added {len(new_tokens)} special tokens to tokenizer")
    
    def load_data(self, file_path: Union[str, Path], is_test: bool = False) -> pd.DataFrame:
        """
        데이터 파일 로딩 (CSV 또는 JSON 지원)
        # 토크나이징
        dataset = dataset.map(
            self._tokenize_function,
            batched=True,
            remove_columns=['input', 'target', 'fname']  # fname도 제거하여 DataCollator 에러 방지
        )
        Returns:
            로딩된 데이터프레임
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        try:
            if file_path.suffix == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix == '.json':
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            logger.info(f"Loaded {len(df)} samples from {file_path}")
            
            # 필수 컬럼 확인
            if is_test:
                required_columns = ['fname', 'dialogue']
            else:
                required_columns = ['fname', 'dialogue', 'summary']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def process_data(self, df: pd.DataFrame, is_training: bool = True, is_test: bool = False) -> HFDataset:
        """
        데이터프레임을 HuggingFace Dataset으로 변환
        
        Args:
            df: 원본 데이터프레임
            is_training: 학습 데이터 여부
            is_test: 테스트 데이터 여부 (summary 없음)
            
        Returns:
            HuggingFace Dataset 객체
        """
        # 텍스트 정제
        df = df.copy()
        df['dialogue'] = df['dialogue'].apply(self.text_preprocessor.clean_dialogue)
        
        # test 데이터가 아닌 경우에만 summary 처리
        if not is_test:
            df['summary'] = df['summary'].apply(self.text_preprocessor.clean_summary)
        
        # 길이 필터링 (학습 데이터만)
        if is_training:
            df = self._filter_by_length(df)
        
        # 데이터 딕셔너리 생성
        data_dict = {
            'input': df['dialogue'].tolist(),
            'fname': df['fname'].tolist()
        }
        # test 데이터가 아닌 경우에만 target 추가
        if not is_test:
            data_dict['target'] = df['summary'].tolist()
        else:
            # test 데이터의 경우 빈 target 생성
            data_dict['target'] = [''] * len(df)
        
        # 모델별 전처리 적용
        if self.model_preprocessor:
            data_dict = self.model_preprocessor(data_dict)
        
        # HuggingFace Dataset 생성
        dataset = HFDataset.from_dict(data_dict)
        
        # 토크나이징
        dataset = dataset.map(
            self._tokenize_function,
            batched=True,
            remove_columns=['input', 'target', 'fname']  # fname도 제거하여 DataCollator 에러 방지
        )
        
        return dataset
    
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
        df['dialogue_length'] = df['dialogue'].str.len()
        df = df[
            (df['dialogue_length'] >= self.min_dialogue_length) &
            (df['dialogue_length'] <= self.max_dialogue_length)
        ]
        
        # 요약문 길이 필터링
        df['summary_length'] = df['summary'].str.len()
        df = df[
            (df['summary_length'] >= self.min_summary_length) &
            (df['summary_length'] <= self.max_summary_length)
        ]
        
        final_count = len(df)
        if initial_count > final_count:
            logger.info(f"Filtered {initial_count - final_count} samples by length")
        
        return df
    
    def _preprocess_for_model(self, text: str, model_type: str = None) -> str:
        """
        송규헌님 요청사항: 모델별 전처리
        
        각 모델 아키텍처에 맞는 입력 형식으로 변환하여 성능을 최적화합니다.
        trainer.py의 동일 함수와 일관된 처리를 수행합니다.
        
        Args:
            text: 입력 텍스트
            model_type: 모델 타입 ('t5', 'gpt', 'bart', 'default'). None인 경우 자동 추론
            
        Returns:
            전처리된 텍스트
        """
        if model_type is None:
            # 토크나이저를 통해 모델 타입 추론
            model_name = getattr(self.tokenizer, 'name_or_path', '').lower()
            if any(keyword in model_name for keyword in ['t5', 'flan-t5', 'mt5']):
                model_type = 't5'
            elif any(keyword in model_name for keyword in ['gpt', 'kogpt']):
                model_type = 'gpt'
            elif 'bart' in model_name:
                model_type = 'bart'
            else:
                model_type = 'default'
        
        # 입력 텍스트 검증
        if not text or not isinstance(text, str):
            return str(text) if text else ""
        
        # 모델별 전처리
        text = text.strip()
        
        if model_type == 't5':
            # T5 모델들에 대한 prefix 처리
            if not text.startswith('summarize:'):
                text = f'summarize: {text}'
        
        elif model_type == 'gpt':
            # GPT 모델들에 대한 TL;DR 처리
            if not text.endswith(' TL;DR:') and not text.endswith('TL;DR:'):
                text = f'{text} TL;DR:'
        
        # BART 및 기타 모델은 변경사항 없음
        
        return text
    
    def _tokenize_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        """
        배치 토크나이징 함수
        
        Args:
            examples: 배치 데이터 딕셔너리
            
        Returns:
            토크나이징된 데이터 딕셔너리
        """
        # 모델 이름 확인
        model_name = self.config.get('general', {}).get('model_name', '')
        
        # eenzeenee 모델의 경우 길이 제한
        if "eenzeenee" in model_name.lower():
            try:
                from utils.eenzeenee_utils import get_safe_max_length
                safe_lengths = get_safe_max_length(model_name, self.config)
                encoder_max_len = safe_lengths['encoder_max_len']
                decoder_max_len = safe_lengths['decoder_max_len']
            except ImportError:
                encoder_max_len = min(self.encoder_max_len, 256)
                decoder_max_len = min(self.decoder_max_len, 64)
        else:
            encoder_max_len = self.encoder_max_len
            decoder_max_len = self.decoder_max_len
        
        # 입력 토크나이징
        model_inputs = self.tokenizer(
            examples['input'],
            max_length=encoder_max_len,
            padding='max_length',
            truncation=True
        )
        
        # 타겟 토크나이징
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                examples['target'],
                max_length=decoder_max_len,
                padding='max_length',
                truncation=True
            )
        
        # 패딩 토큰을 -100으로 변경 (loss 계산에서 무시)
        labels['input_ids'] = [
            [(label if label != self.tokenizer.pad_token_id else -100) for label in label_ids]
            for label_ids in labels['input_ids']
        ]
        
        model_inputs['labels'] = labels['input_ids']
        # model_inputs['fname'] = examples['fname']  # DataCollator 에러 방지를 위해 제거
        
        return model_inputs
    
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
            'labels': labels.squeeze()
            # fname 제거: DataCollator 호환성을 위해
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
