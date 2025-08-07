from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import EarlyStoppingCallback
import torch
from transformers.optimization import get_cosine_with_hard_restarts_schedule_with_warmup
import wandb
import os
from rouge import Rouge # 모델의 성능을 평가하기 위한 라이브러리입니다.
# from transformers import AutoTokenizer

# 모델 성능에 대한 평가 지표를 정의합니다. 본 대회에서는 ROUGE 점수를 통해 모델의 성능을 평가합니다.
def compute_metrics(config, tokenizer, pred): #, eval_tokenizer=None):
    rouge = Rouge()
    predictions = pred.predictions
    labels = pred.label_ids

    predictions[predictions == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id

    # if eval_tokenizer is not None:
    #     decoded_preds = eval_tokenizer.batch_decode(predictions, clean_up_tokenization_spaces=True)
    #     labels = eval_tokenizer.batch_decode(labels, clean_up_tokenization_spaces=True)

    decoded_preds = tokenizer.batch_decode(predictions, clean_up_tokenization_spaces=True)
    labels = tokenizer.batch_decode(labels, clean_up_tokenization_spaces=True)

    # 정확한 평가를 위해 미리 정의된 불필요한 생성토큰들을 제거합니다.
    replaced_predictions = decoded_preds.copy()
    replaced_labels = labels.copy()
    remove_tokens = config['inference']['remove_tokens']
    for token in remove_tokens:
        replaced_predictions = [sentence.replace(token," ") for sentence in replaced_predictions]
        replaced_labels = [sentence.replace(token," ") for sentence in replaced_labels]

    print('-'*150)
    print(f"PRED: {replaced_predictions[0]}")
    print(f"GOLD: {replaced_labels[0]}")
    print('-'*150)
    print(f"PRED: {replaced_predictions[1]}")
    print(f"GOLD: {replaced_labels[1]}")
    print('-'*150)
    print(f"PRED: {replaced_predictions[2]}")
    print(f"GOLD: {replaced_labels[2]}")

    # 최종적인 ROUGE 점수를 계산합니다.
    results = rouge.get_scores(replaced_predictions, replaced_labels,avg=True)

    # ROUGE 점수 중 F-1 score를 통해 평가합니다.
    result = {key: value["f"] for key, value in results.items()}
    return result

# 학습을 위한 trainer 클래스와 매개변수를 정의합니다.
def load_trainer_for_train(config,generate_model,tokenizer,train_inputs_dataset,val_inputs_dataset):
    print('-'*10, 'Make training arguments', '-'*10,)

    training_args = Seq2SeqTrainingArguments(
        seed=config['training']["seed"],
        output_dir=config['training']["output_dir"],
        overwrite_output_dir=config['training']["overwrite_output_dir"],

        save_total_limit=config['training']["save_total_limit"],
        load_best_model_at_end=config['training']["load_best_model_at_end"],
        save_steps=config['training']["save_steps"],

        logging_steps=config['training']["logging_steps"],

        num_train_epochs=config['training']["num_train_epochs"],
        per_device_train_batch_size=config['training']["per_device_train_batch_size"],
        remove_unused_columns=config['training']["remove_unused_columns"],
        fp16=config['training']["fp16"],
        bf16=config['training']["bf16"],
        dataloader_drop_last=config['training']["dataloader_drop_last"],
        group_by_length=config['training']["group_by_length"],
        
        gradient_checkpointing=config['training']["gradient_checkpointing"],
        gradient_checkpointing_kwargs=config['training']["gradient_checkpointing_kwargs"],
        gradient_accumulation_steps=config['training']["gradient_accumulation_steps"],
        torch_empty_cache_steps=config['training']["torch_empty_cache_steps"],
        dataloader_num_workers=config['training']["dataloader_num_workers"],

        per_device_eval_batch_size=config['training']["per_device_eval_batch_size"],
        eval_strategy=config['training']["evaluation_strategy"],
        eval_steps=config['training']["eval_steps"],
        
        predict_with_generate=config['training']["predict_with_generate"],
        generation_max_length=config['training']["generation_max_length"],
        report_to=config['training']["report_to"],
    )

    if config['training']["report_to"] in ['all', 'wandb']:
        # (선택) 모델의 학습 과정을 추적하는 wandb를 사용하기 위해 초기화 해줍니다.
        wandb.init(
            entity=config['wandb']['entity'],
            project=config['wandb']['project'],
            name=config['wandb']['name'],
        )

    # (선택) 모델 checkpoint를 wandb에 저장하도록 환경 변수를 설정합니다.
    os.environ["WANDB_LOG_MODEL"]="false" # wandb에 가장 validation 점수가 좋은 checkpoint만 업로드하여 storage 절약.
    os.environ["WANDB_WATCH"]="false"
    # Hugging Face의 tokenizers 라이브러리가 병렬 처리(parallelism) 기능을 사용할지 여부
    os.environ['TOKENIZERS_PARALLELISM'] = "true"

    # Validation loss가 더 이상 개선되지 않을 때 학습을 중단시키는 EarlyStopping 기능을 사용합니다.
    MyCallback = EarlyStoppingCallback(
        early_stopping_patience=config['training']['early_stopping_patience'],
        early_stopping_threshold=config['training']['early_stopping_threshold']
    )
    print('-'*10, 'Make training arguments complete', '-'*10,)
    print('-'*10, 'Make trainer', '-'*10,)

    optimizer = torch.optim.AdamW(
        generate_model.parameters(),
        lr=config['training']['learning_rate'],
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=config['training']['weight_decay'],
        amsgrad=False
    )

    ### evaluation 용 tokenizer가 설정되어 있다면 해당 토크나이저로 validation 점수 계산
    # eval_tokenizer = None
    # if config['general'].get("eval_tokenizer", False) and len(config['general']['eval_tokenizer']) > 1:
    #     eval_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=config['general']['eval_tokenizer'])
    #     special_tokens_dict={'additional_special_tokens':config['tokenizer']['special_tokens']}
    #     eval_tokenizer.add_special_tokens(special_tokens_dict)

    # Trainer 클래스를 정의합니다.
    trainer = Seq2SeqTrainer(
        model=generate_model, # 사용자가 사전 학습하기 위해 사용할 모델을 입력합니다.
        args=training_args,
        train_dataset=train_inputs_dataset,
        eval_dataset=val_inputs_dataset,
        compute_metrics = lambda pred: compute_metrics(config, tokenizer, pred), #, eval_tokenizer),
        callbacks = [MyCallback],
        optimizers=(
            optimizer,
            get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=config['training']['warmup_steps'],
                num_training_steps=len(train_inputs_dataset) * config['training']['num_train_epochs'],
                num_cycles=config['training'].get("num_cycles", 1)
            )
        )
    )
    print('-'*10, 'Make trainer complete', '-'*10,)

    return trainer
    