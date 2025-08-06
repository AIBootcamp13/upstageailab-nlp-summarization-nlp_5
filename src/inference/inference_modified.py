import os
import re
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

import sys
sys.path.append(
    "/data/ephemeral/home/nlp-5/lyj"
)
from dataset.dataset_base import *
from dataset.preprocess import *
from models.BART import *

# tokenization 과정까지 진행된 최종적으로 모델에 입력될 데이터를 출력합니다.
def prepare_test_dataset(config, preprocessor, tokenizer):

    test_file_path = os.path.join(config['general']['data_path'],'test.csv')

    test_data = preprocessor.make_set_as_df(test_file_path,is_train=False)
    test_id = test_data['fname']

    print('-'*150)
    print(f'test_data:\n{test_data["dialogue"][0]}')
    print('-'*150)

    encoder_input_test , decoder_input_test = preprocessor.make_input(test_data,is_test=True)
    print('-'*10, 'Load data complete', '-'*10,)

    test_tokenized_encoder_inputs = tokenizer(encoder_input_test, return_tensors="pt", padding=True,
                    add_special_tokens=True, truncation=True, max_length=config['tokenizer']['encoder_max_len'], return_token_type_ids=False,)
    test_tokenized_decoder_inputs = tokenizer(decoder_input_test, return_tensors="pt", padding=True,
                    add_special_tokens=True, truncation=True, max_length=config['tokenizer']['decoder_max_len'], return_token_type_ids=False,)

    test_encoder_inputs_dataset = DatasetForInference(test_tokenized_encoder_inputs, test_id, len(encoder_input_test))
    print('-'*10, 'Make dataset complete', '-'*10,)

    return test_data, test_encoder_inputs_dataset


def enhance_summary(summary):
    """요약문 품질 향상을 위한 후처리"""
    # 1. 문장 완결성 체크
    if not summary.strip().endswith('.'):
        summary += '.'
    
    # 2. Person 태그 정규화
    summary = re.sub(r'#Person(\d+)#', r'#Person\1#', summary)
    
    # 3. 연속된 공백 제거
    summary = re.sub(r'\s+', ' ', summary)
    
    # 4. 문장 시작/끝 공백 제거
    summary = summary.strip()
    
    # 5. 중복 문구 제거 (간단한 경우만)
    sentences = summary.split('.')
    if len(sentences) > 1:
        unique_sentences = []
        for sent in sentences:
            sent = sent.strip()
            if sent and sent not in unique_sentences:
                unique_sentences.append(sent)
        if unique_sentences:
            summary = '. '.join(unique_sentences)
            if not summary.endswith('.'):
                summary += '.'
    
    return summary


# 학습된 모델이 생성한 요약문의 출력 결과를 보여줍니다.
def inference(config, generate_model, tokenizer):
    device = torch.device('cuda:0' if torch.cuda.is_available()  else 'cpu')
    print('-'*10, f'device : {device}', '-'*10,)
    print(torch.__version__)

    # generate_model , tokenizer = load_tokenizer_and_model_for_test(config,device)

    data_path = config['general']['data_path']
    preprocessor = Preprocess(config['tokenizer']['bos_token'], config['tokenizer']['eos_token'])

    test_data, test_encoder_inputs_dataset = prepare_test_dataset(config,preprocessor, tokenizer)
    dataloader = DataLoader(test_encoder_inputs_dataset, batch_size=config['inference']['batch_size'])

    summary = []
    text_ids = []
    
    # 🚀 최적화된 generation 파라미터 준비
    generation_kwargs = {
        'no_repeat_ngram_size': config['inference']['no_repeat_ngram_size'],
        'early_stopping': config['inference']['early_stopping'],
        'max_length': config['inference']['generate_max_length'],
        'num_beams': config['inference']['num_beams'],
    }
    
    # 추가 파라미터들 (config에 있으면 사용)
    if 'length_penalty' in config['inference']:
        generation_kwargs['length_penalty'] = config['inference']['length_penalty']
    if 'repetition_penalty' in config['inference']:
        generation_kwargs['repetition_penalty'] = config['inference']['repetition_penalty']
    if 'do_sample' in config['inference'] and config['inference']['do_sample']:
        generation_kwargs['do_sample'] = True
        if 'temperature' in config['inference']:
            generation_kwargs['temperature'] = config['inference']['temperature']
        if 'top_k' in config['inference']:
            generation_kwargs['top_k'] = config['inference']['top_k']
        if 'top_p' in config['inference']:
            generation_kwargs['top_p'] = config['inference']['top_p']
    
    print("Generation parameters:", generation_kwargs)
    
    with torch.no_grad():
        for item in tqdm(dataloader):
            text_ids.extend(item['ID'])
            generated_ids = generate_model.generate(
                input_ids=item['input_ids'].to(device),
                **generation_kwargs
            )
            for ids in generated_ids:
                result = tokenizer.decode(ids)
                summary.append(result)

    # 정확한 평가를 위하여 노이즈에 해당되는 스페셜 토큰을 제거합니다.
    remove_tokens = config['inference']['remove_tokens']
    preprocessed_summary = summary.copy()
    for token in remove_tokens:
        preprocessed_summary = [sentence.replace(token," ") for sentence in preprocessed_summary]

    # 🎯 요약문 품질 향상 적용
    enhanced_summary = [enhance_summary(s) for s in preprocessed_summary]
    
    output = pd.DataFrame(
        {
            "fname": test_data['fname'],
            "summary" : enhanced_summary,
        }
    )
    result_path = config['inference']['result_path'] # submission 파일 경로
    if not os.path.exists(result_path):
        os.makedirs(result_path, exist_ok=True)
    output.to_csv(os.path.join(result_path, 'result1'), index=False)

    print(f"✅ Inference complete! Results saved to {result_path}/result1")
    print(f"📊 Generated {len(enhanced_summary)} summaries")
    
    return output
