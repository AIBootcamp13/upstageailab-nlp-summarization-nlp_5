import argparse
import pandas as pd
import os
import re
import json
import yaml
from glob import glob
from tqdm import tqdm
from pprint import pprint
import torch
import pytorch_lightning as pl
from rouge import Rouge # 모델의 성능을 평가하기 위한 라이브러리입니다.

from torch.utils.data import Dataset , DataLoader
from transformers import AutoTokenizer, BartForConditionalGeneration, BartConfig
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback

import wandb # 모델 학습 과정을 손쉽게 Tracking하고, 시각화할 수 있는 라이브러리입니다.

project_dir = "/data/ephemeral/home/nlp-5/pyeon/upstageailab-nlp-summarization-nlp_5"

import sys
sys.path.append(
    project_dir
)
from dataset.dataset_base import *
from dataset.preprocess import *
from models.BART import *
from trainer.trainer_base_modified import *
from inference.inference_modified import *

def main(config):
    try:
        # 사용할 device를 정의합니다.~
        device = torch.device('cuda:0' if torch.cuda.is_available()  else 'cpu')
        print('-'*10, f'device : {device}', '-'*10,)
        print(torch.__version__)

        # 사용할 모델과 tokenizer를 불러옵니다.
        generate_model , tokenizer = load_tokenizer_and_model_for_train(config,device)
        print('-'*10,"tokenizer special tokens : ",tokenizer.special_tokens_map,'-'*10)

        # 학습에 사용할 데이터셋을 불러옵니다.
        preprocessor = Preprocess(config['tokenizer']['bos_token'], config['tokenizer']['eos_token']) # decoder_start_token: str, eos_token: str
        data_path = config['general']['data_path']
        train_inputs_dataset, val_inputs_dataset = prepare_train_dataset(config,preprocessor, data_path, tokenizer)

        # Trainer 클래스를 불러옵니다.`
        trainer = load_trainer_for_train(config, generate_model,tokenizer,train_inputs_dataset,val_inputs_dataset)
        trainer.train()   # 모델 학습을 시작합니다.

        # best 모델과 토크나이저 저장
        trainer.model.save_pretrained(config['inference']['ckt_dir'])
        tokenizer.save_pretrained(config['inference']['ckt_dir'])

        # inference 후 submission 파일 저장.
        _ = inference(config, trainer.model, tokenizer)
    finally:
        # (선택) 모델 학습이 완료된 후 wandb를 종료합니다.
        wandb.finish()

if __name__ == "__main__":
    os.chdir(project_dir) ## project directory로 현재 실행 위치를 옮긴다.

    parser = argparse.ArgumentParser(description="Run deep learning training with specified configuration.")
    parser.add_argument(
        '--config',
        type=str,
        default='config_base.yaml', # 기본값 설정
        help='Name of the configuration YAML file (e.g., config.yaml, experiment_A.yaml)'
    )
    parser.add_argument(
        '--inference',
        type=bool,
        default=False,
        help='executing this file as inference mode'
    )

    args = parser.parse_args()

    # Load Configuration file
    config_path = os.path.join(
        project_dir,'src','configs',
        args.config # config 파일 이름을 설정
    )
    with open(config_path, "r") as file:
        loaded_config = yaml.safe_load(file)

    if not args.inference:
        main(loaded_config)
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        tokenizer, generate_model = load_tokenizer_and_model_for_inference(loaded_config, device)
        inference(loaded_config, generate_model, tokenizer)
