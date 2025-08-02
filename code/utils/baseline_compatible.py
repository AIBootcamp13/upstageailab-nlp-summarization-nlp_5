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
