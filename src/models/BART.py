from transformers import AutoTokenizer, BartForConditionalGeneration, BartConfig
import os

# 학습을 위한 tokenizer와 사전 학습된 모델을 불러옵니다.
def load_tokenizer_and_model_for_train(config, device):
    print('-'*10, 'Load tokenizer & model', '-'*10,)
    print('-'*10, f'Model Name : {config["general"]["model_name"]}', '-'*10,)
    model_name = config['general']['model_name']
    
    # HF_TOKEN 환경변수 가져오기
    hf_token = os.getenv('HF_TOKEN')
    
    bart_config = BartConfig().from_pretrained(model_name, token=hf_token)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    generate_model = BartForConditionalGeneration.from_pretrained(config['general']['model_name'], config=bart_config, token=hf_token)

    special_tokens_dict={'additional_special_tokens':config['tokenizer']['special_tokens']}
    tokenizer.add_special_tokens(special_tokens_dict)

    generate_model.resize_token_embeddings(len(tokenizer)) # 사전에 special token을 추가했으므로 재구성 해줍니다.
    generate_model.to(device)
    print(generate_model.config)

    print('-'*10, 'Load tokenizer & model complete', '-'*10,)
    return generate_model , tokenizer

def load_tokenizer_and_model_for_inference(config, device):
    tokenizer = AutoTokenizer.from_pretrained(
        config['inference']['ckt_dir']
    )
    model = BartForConditionalGeneration.from_pretrained(
        config['inference']['ckt_dir']
    ).to(device)
    return tokenizer, model
