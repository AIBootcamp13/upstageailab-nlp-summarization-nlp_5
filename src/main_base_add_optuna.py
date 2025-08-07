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
from trainer.trainer_base2 import *
from inference.inference import *

import optuna
import csv
import copy

def main(config, do_save_and_infer=False):
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

        # # validation metric(예: rouge score) 반환하도록 수정 필요
        # # 예시: trainer의 best metric 반환
        # if hasattr(trainer, 'state') and hasattr(trainer.state, 'best_metric'):
        #     metric =  trainer.state.best_metric
        # else:
        #     metric = 0.0

        # metric 추출: log_history에서 마지막 eval metric을 가져옴
        eval_metrics = {}
        for log in reversed(trainer.state.log_history):
            if "eval_rouge-l" in log:
                eval_metrics = log
                break
        rouge_l = eval_metrics.get("eval_rouge-l", 0.0)
        rouge_mean = eval_metrics.get("eval_rouge-mean", 0.0)

        # 저장/추론은 플래그가 True일 때만 실행
        if do_save_and_infer:
            # best 모델과 토크나이저 저장
            trainer.model.save_pretrained(config['inference']['ckt_dir'])
            tokenizer.save_pretrained(config['inference']['ckt_dir'])

            # inference 후 submission 파일 저장.
            _ = inference(config, trainer.model, tokenizer)

        return rouge_l, rouge_mean
    finally:
        # (선택) 모델 학습이 완료된 후 wandb를 종료합니다.
        wandb.finish()

trial_results = []  # 각 trial의 결과를 저장할 리스트

def objective(trial, loaded_config):
    # 하이퍼파라미터를 trial에서 받아 config에 반영
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 5e-4)
    weight_decay = trial.suggest_uniform('weight_decay', 0.0, 0.1)
    batch_size = trial.suggest_categorical('per_device_train_batch_size', [8, 16, 32, 64])
    num_train_epochs = trial.suggest_categorical('num_train_epochs', [3, 5, 10, 20])
    warmup_steps = trial.suggest_categorical('warmup_steps', [0, 10, 30, 50, 100])
    grad_accum_steps = trial.suggest_categorical('gradient_accumulation_steps', [1, 2, 4, 8])

    loaded_config['training']['learning_rate'] = learning_rate
    loaded_config['training']['weight_decay'] = weight_decay
    loaded_config['training']['per_device_train_batch_size'] = batch_size
    loaded_config['training']['num_train_epochs'] = num_train_epochs
    loaded_config['training']['warmup_steps'] = warmup_steps
    loaded_config['training']['gradient_accumulation_steps'] = grad_accum_steps

    # trial 정보 출력
    print('-'*10, 'Trial hyperparameters', '-'*10)
    print(f"[Trial {trial.number}] Params: \n"
          f"learning_rate={learning_rate}, \n"
          f"weight_decay={weight_decay}, \n"
          f"batch_size={batch_size}, \n"
          f"num_train_epochs={num_train_epochs}, \n"
          f"warmup_steps={warmup_steps}, \n"
          f"gradient_accumulation_steps={grad_accum_steps}")

    # main 함수 실행 및 metric 반환
    rouge_l, rouge_mean = main(loaded_config, do_save_and_infer=False)

    # trial 결과 저장
    trial_results.append({
        "trial_number": trial.number,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "num_train_epochs": num_train_epochs,
        "warmup_steps": warmup_steps,
        "gradient_accumulation_steps": grad_accum_steps,
        "eval_rouge-l": rouge_l,
        "rouge-mean": rouge_mean,
    })

    return rouge_mean

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
        #main(loaded_config)
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, loaded_config), n_trials=2)

        # trial 결과 csv로 저장
        results_path = os.path.join(project_dir, 'src', 'trial', 'optuna_trial_results.csv')
        with open(results_path, 'w', newline='') as csvfile:
            fieldnames = ["trial_number", "learning_rate", "weight_decay", "batch_size", "num_train_epochs", "warmup_steps", "gradient_accumulation_steps", "eval_rouge-l", "rouge-mean"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in trial_results:
                writer.writerow(row)
        print(f"All trial results saved to {results_path}")

        # 1. 최적의 하이퍼파라미터 출력
        print("-"*10, 'Best hyperparameters found:', '-'*10)
        print("Best trial:")
        print(study.best_trial)
        print("Best params:", study.best_params)
        print("Best value:", study.best_value)

        # 2. 최적의 하이퍼파라미터를 새로운 yaml 파일로 저장
        best_config = copy.deepcopy(loaded_config)
        for k, v in study.best_params.items():
            best_config['training'][k] = v
        best_config_path = os.path.join(project_dir, 'src', 'trial', 'best_config.yaml')
        with open(best_config_path, 'w') as f:
            yaml.dump(best_config, f, allow_unicode=True)
        print(f"Best config saved to {best_config_path}")

        # 3. 최적 파라미터로 모델 재학습/저장/추론(submission 생성)
        print("Retraining with best hyperparameters and generating submission...")
        main(best_config, do_save_and_infer=True)

    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        tokenizer, generate_model = load_tokenizer_and_model_for_inference(loaded_config, device)
        inference(loaded_config, generate_model, tokenizer)
