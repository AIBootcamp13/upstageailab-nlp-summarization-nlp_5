import argparse
import pandas as pd
import os
import yaml
import torch
import wandb

project_dir = "/data/ephemeral/home/nlp-5/eunbyul/ah"

import sys
sys.path.append(project_dir)
from dataset.dataset_base import *
from dataset.preprocess import *
from models.BART import *
from trainer.trainer_base_modified import *
from inference.inference_modified import *

def main(config, config_path):
    try:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('-'*10, f'device : {device}', '-'*10)
        print(torch.__version__)

        generate_model, tokenizer = load_tokenizer_and_model_for_train(config, device)
        print('-'*10, "tokenizer special tokens : ", tokenizer.special_tokens_map, '-'*10)

        preprocessor = Preprocess(config['tokenizer']['bos_token'], config['tokenizer']['eos_token'])
        data_path = config['general']['data_path']
        train_inputs_dataset, val_inputs_dataset = prepare_train_dataset(config, preprocessor, data_path, tokenizer)

        trainer = load_trainer_for_train(config, generate_model, tokenizer, train_inputs_dataset, val_inputs_dataset)
        trainer.train()

        trainer.model.save_pretrained(config['inference']['ckt_dir'])
        tokenizer.save_pretrained(config['inference']['ckt_dir'])

        _ = inference(config, trainer.model, tokenizer, config_path)
    finally:
        wandb.finish()

if __name__ == "__main__":
    os.chdir(project_dir)

    parser = argparse.ArgumentParser(description="Run deep learning training with specified configuration.")
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Name of the configuration YAML file (e.g., config.yaml, experiment_A.yaml)'
    )
    parser.add_argument(
        '--inference',
        type=bool,
        default=False,
        help='Executing this file as inference mode'
    )

    args = parser.parse_args()

    config_path = args.config

    with open(config_path, "r") as file:
        loaded_config = yaml.safe_load(file)

    if not args.inference:
        main(loaded_config, config_path)
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        tokenizer, generate_model = load_tokenizer_and_model_for_inference(loaded_config, device)
        inference(loaded_config, generate_model, tokenizer, config_path)