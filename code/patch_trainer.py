import pandas as pd
from datasets import Dataset, DatasetDict

def create_prepare_data_patch():
    """prepare_data 메서드를 baseline.py 방식으로 교체하는 패치"""
    
    new_method = '''
    def prepare_data(self, train_path: Optional[str] = None, 
                    val_path: Optional[str] = None,
                    test_path: Optional[str] = None) -> DatasetDict:
        """
        데이터 준비 - baseline.py 방식으로 수정
        """
        # 경로 결정
        data_paths = self.config.get('data', self.config.get('general', {}))
        train_path = train_path or data_paths.get('train_path')
        val_path = val_path or data_paths.get('val_path')
        test_path = test_path or data_paths.get('test_path')
        
        logger.info("Loading and processing datasets (baseline style)...")
        
        datasets = {}
        
        # Train 데이터 처리
        if train_path:
            logger.info(f"Loading train data from: {train_path}")
            
            # pandas로 CSV 읽기
            train_df = pd.read_csv(train_path)
            train_df = train_df[['fname', 'dialogue', 'summary']]
            
            # baseline의 make_input 로직
            encoder_inputs = []
            decoder_inputs = []
            decoder_outputs = []
            
            bos_token = self.tokenizer.bos_token
            eos_token = self.tokenizer.eos_token
            
            for dialogue, summary in zip(train_df['dialogue'], train_df['summary']):
                encoder_input = dialogue
                decoder_input = f"{bos_token} {summary}"
                decoder_output = f"{summary} {eos_token}"
                
                encoder_inputs.append(encoder_input)
                decoder_inputs.append(decoder_input)
                decoder_outputs.append(decoder_output)
            
            # 토크나이징
            tokenized_encoder = self.tokenizer(
                encoder_inputs, 
                return_tensors="pt", 
                padding=True,
                truncation=True, 
                max_length=self.config['tokenizer']['encoder_max_len'],
                return_token_type_ids=False
            )
            
            tokenized_decoder_inputs = self.tokenizer(
                decoder_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config['tokenizer']['decoder_max_len'],
                return_token_type_ids=False
            )
            
            tokenized_decoder_outputs = self.tokenizer(
                decoder_outputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config['tokenizer']['decoder_max_len'],
                return_token_type_ids=False
            )
            
            # HuggingFace Dataset 형식으로 변환
            train_dataset = Dataset.from_dict({
                'input_ids': tokenized_encoder['input_ids'],
                'attention_mask': tokenized_encoder['attention_mask'],
                'decoder_input_ids': tokenized_decoder_inputs['input_ids'],
                'labels': tokenized_decoder_outputs['input_ids']
            })
            
            datasets['train'] = train_dataset
            logger.info(f"Train dataset size: {len(train_dataset)}")
        
        # Validation 데이터 처리
        if val_path:
            logger.info(f"Loading validation data from: {val_path}")
            
            # pandas로 CSV 읽기
            val_df = pd.read_csv(val_path)
            val_df = val_df[['fname', 'dialogue', 'summary']]
            
            # baseline의 make_input 로직
            encoder_inputs = []
            decoder_inputs = []
            decoder_outputs = []
            
            bos_token = self.tokenizer.bos_token
            eos_token = self.tokenizer.eos_token
            
            for dialogue, summary in zip(val_df['dialogue'], val_df['summary']):
                encoder_input = dialogue
                decoder_input = f"{bos_token} {summary}"
                decoder_output = f"{summary} {eos_token}"
                
                encoder_inputs.append(encoder_input)
                decoder_inputs.append(decoder_input)
                decoder_outputs.append(decoder_output)
            
            # 토크나이징
            tokenized_encoder = self.tokenizer(
                encoder_inputs, 
                return_tensors="pt", 
                padding=True,
                truncation=True, 
                max_length=self.config['tokenizer']['encoder_max_len'],
                return_token_type_ids=False
            )
            
            tokenized_decoder_inputs = self.tokenizer(
                decoder_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config['tokenizer']['decoder_max_len'],
                return_token_type_ids=False
            )
            
            tokenized_decoder_outputs = self.tokenizer(
                decoder_outputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config['tokenizer']['decoder_max_len'],
                return_token_type_ids=False
            )
            
            # HuggingFace Dataset 형식으로 변환
            val_dataset = Dataset.from_dict({
                'input_ids': tokenized_encoder['input_ids'],
                'attention_mask': tokenized_encoder['attention_mask'],
                'decoder_input_ids': tokenized_decoder_inputs['input_ids'],
                'labels': tokenized_decoder_outputs['input_ids']
            })
            
            datasets['validation'] = val_dataset
            logger.info(f"Validation dataset size: {len(val_dataset)}")
        
        # Test 데이터 처리
        if test_path:
            logger.info(f"Loading test data from: {test_path}")
            
            # pandas로 CSV 읽기
            test_df = pd.read_csv(test_path)
            test_df = test_df[['fname', 'dialogue']]  # test는 summary 없음
            
            # 토크나이징 (encoder만)
            tokenized_encoder = self.tokenizer(
                test_df['dialogue'].tolist(), 
                return_tensors="pt", 
                padding=True,
                truncation=True, 
                max_length=self.config['tokenizer']['encoder_max_len'],
                return_token_type_ids=False
            )
            
            # HuggingFace Dataset 형식으로 변환
            test_dataset = Dataset.from_dict({
                'input_ids': tokenized_encoder['input_ids'],
                'attention_mask': tokenized_encoder['attention_mask']
            })
            
            datasets['test'] = test_dataset
            logger.info(f"Test dataset size: {len(test_dataset)}")
        
        return DatasetDict(datasets)
'''
    
    return new_method

# 파일 읽기
with open('trainer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# prepare_data 메서드 찾기
import re

# 기존 prepare_data 메서드를 찾아서 교체
pattern = r'def prepare_data\(self,.*?(?=\n    def|\nclass|\Z)'
new_method = create_prepare_data_patch()

# pandas와 datasets import 추가
if 'import pandas as pd' not in content:
    # import 섹션 찾기
    import_section_end = content.find('\nclass')
    if import_section_end > 0:
        content = content[:import_section_end] + '\nimport pandas as pd\nfrom datasets import Dataset, DatasetDict\n' + content[import_section_end:]

# prepare_data 메서드 교체
content = re.sub(pattern, new_method.strip() + '\n', content, flags=re.DOTALL)

# 파일 저장
with open('trainer.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("prepare_data method has been replaced with baseline.py style")
