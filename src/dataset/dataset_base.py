import os
from torch.utils.data import Dataset

# Train에 사용되는 Dataset 클래스를 정의합니다.
class DatasetForTrain(Dataset):
    # 클래스 초기화 메서드
    def __init__(self, encoder_input, decoder_input, labels, len):
        self.encoder_input = encoder_input # 토큰화된 인코더 입력을 인스턴스 변수에 저장
        self.decoder_input = decoder_input # 토큰화된 디코더 입력을 인스턴스 변수에 저장
        self.labels = labels # 토큰화된 레이블(디코더 출력)을 인스턴스 변수에 저장
        self.len = len # 데이터셋의 길이를 인스턴스 변수에 저장

    # 특정 인덱스(idx)의 데이터를 가져오는 메서드
    def __getitem__(self, idx):
        # 인코더 입력에서 해당 인덱스의 데이터를 복사하여 item 딕셔너리 생성
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()} # item은 'input_ids'와 'attention_mask'를 키로 가짐
        # 디코더 입력에서 해당 인덱스의 데이터를 복사하여 item2 딕셔너리 생성
        item2 = {key: val[idx].clone().detach() for key, val in self.decoder_input.items()} # item2는 'input_ids'와 'attention_mask'를 키로 가짐
        # 모델의 디코더 입력 키 이름('decoder_input_ids')에 맞게 'input_ids'를 변경
        item2['decoder_input_ids'] = item2['input_ids']
        # 모델의 디코더 어텐션 마스크 키 이름('decoder_attention_mask')에 맞게 'attention_mask'를 변경
        item2['decoder_attention_mask'] = item2['attention_mask']
        # 기존의 'input_ids' 키를 item2에서 제거
        item2.pop('input_ids')
        # 기존의 'attention_mask' 키를 item2에서 제거
        item2.pop('attention_mask')
        # item 딕셔너리에 item2 딕셔너리의 내용을 추가
        item.update(item2) # 이제 item은 'input_ids', 'attention_mask', 'decoder_input_ids', 'decoder_attention_mask'를 키로 가짐
        # 최종적으로 'labels' 키에 해당 인덱스의 레이블(토큰화된 디코더 출력)을 추가
        item['labels'] = self.labels['input_ids'][idx] # 이제 item은 'input_ids', 'attention_mask', 'decoder_input_ids', 'decoder_attention_mask', 'labels'를 키로 가짐
        return item # 모델의 입력으로 사용될 딕셔너리 반환

    # 데이터셋의 전체 길이를 반환하는 메서드
    def __len__(self):
        return self.len # __init__에서 저장한 길이 반환

# Validation에 사용되는 Dataset 클래스를 정의합니다.
class DatasetForVal(Dataset):
    # 클래스 초기화 메서드
    def __init__(self, encoder_input, decoder_input, labels, len):
        self.encoder_input = encoder_input # 토큰화된 인코더 입력을 인스턴스 변수에 저장
        self.decoder_input = decoder_input # 토큰화된 디코더 입력을 인스턴스 변수에 저장
        self.labels = labels # 토큰화된 레이블(디코더 출력)을 인스턴스 변수에 저장
        self.len = len # 데이터셋의 길이를 인스턴스 변수에 저장

    # 특정 인덱스(idx)의 데이터를 가져오는 메서드
    def __getitem__(self, idx):
        # 인코더 입력에서 해당 인덱스의 데이터를 복사하여 item 딕셔너리 생성
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()} # item은 'input_ids'와 'attention_mask'를 키로 가짐
        # 디코더 입력에서 해당 인덱스의 데이터를 복사하여 item2 딕셔너리 생성
        item2 = {key: val[idx].clone().detach() for key, val in self.decoder_input.items()} # item2는 'input_ids'와 'attention_mask'를 키로 가짐
        # 모델의 디코더 입력 키 이름('decoder_input_ids')에 맞게 'input_ids'를 변경
        item2['decoder_input_ids'] = item2['input_ids']
        # 모델의 디코더 어텐션 마스크 키 이름('decoder_attention_mask')에 맞게 'attention_mask'를 변경
        item2['decoder_attention_mask'] = item2['attention_mask']
        # 기존의 'input_ids' 키를 item2에서 제거
        item2.pop('input_ids')
        # 기존의 'attention_mask' 키를 item2에서 제거
        item2.pop('attention_mask')
        # item 딕셔너리에 item2 딕셔너리의 내용을 추가
        item.update(item2) # 이제 item은 'input_ids', 'attention_mask', 'decoder_input_ids', 'decoder_attention_mask'를 키로 가짐
        # 최종적으로 'labels' 키에 해당 인덱스의 레이블(토큰화된 디코더 출력)을 추가
        item['labels'] = self.labels['input_ids'][idx] # 이제 item은 'input_ids', 'attention_mask', 'decoder_input_ids', 'decoder_attention_mask', 'labels'를 키로 가짐
        return item # 모델의 입력으로 사용될 딕셔너리 반환

    # 데이터셋의 전체 길이를 반환하는 메서드
    def __len__(self):
        return self.len # __init__에서 저장한 길이 반환

# Test에 사용되는 Dataset 클래스를 정의합니다.
class DatasetForInference(Dataset):
    # 클래스 초기화 메서드
    def __init__(self, encoder_input, test_id, len):
        self.encoder_input = encoder_input # 토큰화된 인코더 입력을 인스턴스 변수에 저장
        self.test_id = test_id # 테스트 데이터의 ID를 인스턴스 변수에 저장
        self.len = len # 데이터셋의 길이를 인스턴스 변수에 저장

    # 특정 인덱스(idx)의 데이터를 가져오는 메서드
    def __getitem__(self, idx):
        # 인코더 입력에서 해당 인덱스의 데이터를 복사하여 item 딕셔너리 생성
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()}
        # item 딕셔너리에 'ID' 키와 해당 인덱스의 테스트 ID 값을 추가
        item['ID'] = self.test_id[idx]
        return item # 모델의 입력 및 ID를 포함하는 딕셔너리 반환

    # 데이터셋의 전체 길이를 반환하는 메서드
    def __len__(self):
        return self.len # __init__에서 저장한 길이 반환

# tokenization 과정까지 진행된 최종적으로 모델에 입력될 데이터를 출력합니다.
def prepare_train_dataset(config, preprocessor, data_dir, tokenizer):
    """_summary_

    :param dictionary config: 각종 설정 딕셔너리
    :param Preprocess preprocessor: Preprocess 객체
    :param str data_dir: data 디렉토리 경로
    :param AutoTokenizer tokenizer: tokenizer 객체
    :return torch.utils.data.Dataset: Custom Dataset for train, val
    """
    train_file_path = os.path.join(data_dir,config['general'].get('train_data','train_japan_backtranslation_filtered.csv'))
    val_file_path = os.path.join(data_dir,'dev.csv')

    # train, validation에 대해 각각 데이터프레임을 구축합니다.
    train_data = preprocessor.make_set_as_df(train_file_path, config)
    val_data = preprocessor.make_set_as_df(val_file_path, config)

    print('-'*150)
    print("train dataframe columns:",train_data.columns)
    print("val dataframe columns:",val_data.columns)

    print('-'*150)
    print(f'train_data:\n {train_data["dialogue"][0]}')
    print(f'train_label:\n {train_data["summary"][0]}')

    print('-'*150)
    print(f'val_data:\n {val_data["dialogue"][0]}')
    print(f'val_label:\n {val_data["summary"][0]}')

    # Enc-Dec 구조의 모델인 경우 Encoder input과 Decoder input을 구분.
    encoder_input_train , decoder_input_train, decoder_output_train = preprocessor.make_input(train_data)
    encoder_input_val , decoder_input_val, decoder_output_val = preprocessor.make_input(val_data)
    print('-'*10, 'Load data complete', '-'*10,)

    tokenized_encoder_inputs = tokenizer(
        encoder_input_train, 
        return_tensors="pt", 
        padding=True,
        add_special_tokens=True, 
        truncation=True, 
        max_length=config['tokenizer']['encoder_max_len'], 
        return_token_type_ids=False
    )
    tokenized_decoder_inputs = tokenizer(
        decoder_input_train, 
        return_tensors="pt", 
        padding=True,
        add_special_tokens=True, 
        truncation=True, 
        max_length=config['tokenizer']['decoder_max_len'], 
        return_token_type_ids=False
    )
    tokenized_decoder_ouputs = tokenizer(
        decoder_output_train, 
        return_tensors="pt", 
        padding=True,
        add_special_tokens=True, 
        truncation=True, 
        max_length=config['tokenizer']['decoder_max_len'], 
        return_token_type_ids=False
    )

    train_inputs_dataset = DatasetForTrain(tokenized_encoder_inputs, tokenized_decoder_inputs, tokenized_decoder_ouputs,len(encoder_input_train))

    val_tokenized_encoder_inputs = tokenizer(
        encoder_input_val, 
        return_tensors="pt", 
        padding=True,
        add_special_tokens=True, 
        truncation=True, 
        max_length=config['tokenizer']['encoder_max_len'], 
        return_token_type_ids=False
    )
    val_tokenized_decoder_inputs = tokenizer(
        decoder_input_val, 
        return_tensors="pt", 
        padding=True,
        add_special_tokens=True, 
        truncation=True, 
        max_length=config['tokenizer']['decoder_max_len'], 
        return_token_type_ids=False
    )
    val_tokenized_decoder_ouputs = tokenizer(
        decoder_output_val, 
        return_tensors="pt", 
        padding=True,
        add_special_tokens=True, 
        truncation=True, 
        max_length=config['tokenizer']['decoder_max_len'], 
        return_token_type_ids=False
    )

    val_inputs_dataset = DatasetForVal(val_tokenized_encoder_inputs, val_tokenized_decoder_inputs, val_tokenized_decoder_ouputs,len(encoder_input_val))

    print('-'*10, 'Make dataset complete', '-'*10,)
    return train_inputs_dataset, val_inputs_dataset
    