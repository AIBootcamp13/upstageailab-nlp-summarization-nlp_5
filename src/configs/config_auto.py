import yaml
import os
from copy import deepcopy

base_dir = os.path.dirname(os.path.abspath(__file__))
base_config_path = os.path.join(base_dir, 'config.yaml')
save_dir = base_dir  # configs 폴더에 저장

with open(base_config_path) as f:
    base_config = yaml.safe_load(f)

experiment_settings = [
    # 실험명,     train_batch, eval_batch, lr,   patience, encoder_max, decoder_max, num_beams, ngram, wd,  epochs, fp16, len_pen, warmup
    ("batch96",      96,         96,        1e-5, 1,        512,        200,         4,         2,     0.01, 20,    True,  1.0,     10),
    ("batch128",     128,        128,       1e-5, 1,        512,        200,         4,         2,     0.01, 20,    True,  1.0,     10),
    ("lr_2e-5",      64,         48,        2e-5, 1,        512,        200,         4,         2,     0.01, 20,    True,  1.0,     10),
    ("patience2",    64,         48,        1e-5, 2,        512,        200,         4,         2,     0.01, 20,    True,  1.0,     10),
    ("encmax1026",   64,         48,        1e-5, 1,       1026,        200,         4,         2,     0.01, 20,    True,  1.0,     10),
    ("decmax180",    64,         48,        1e-5, 1,        512,        180,         4,         2,     0.01, 20,    True,  1.0,     10),
    ("decmax220",    64,         48,        1e-5, 1,        512,        220,         4,         2,     0.01, 20,    True,  1.0,     10),
    ("beam3",        64,         48,        1e-5, 1,        512,        200,         3,         2,     0.01, 20,    True,  1.0,     10),
    ("ngram3",       64,         48,        1e-5, 1,        512,        200,         4,         3,     0.01, 20,    True,  1.0,     10),
    ("wd_0001",      64,         48,        1e-5, 1,        512,        200,         4,         2,     0.001,20,    True,  1.0,     10),
    ("epoch30",      64,         48,        1e-5, 1,        512,        200,         4,         2,     0.01, 30,    True,  1.0,     10),
    ("fp16off",      64,         48,        1e-5, 1,        512,        200,         4,         2,     0.01, 20,    False, 1.0,     10),
    ("lenpen1.2",    64,         48,        1e-5, 1,        512,        200,         4,         2,     0.01, 20,    True,  1.2,     10),
    ("lenpen1.5",    64,         48,        1e-5, 1,        512,        200,         4,         2,     0.01, 20,    True,  1.5,     10),
    ("warmup50",     64,         48,        1e-5, 1,        512,        200,         4,         2,     0.01, 20,    True,  1.0,     50),
]

for exp in experiment_settings:
    (exp_name, train_batch, eval_batch, lr, patience, encoder_max, decoder_max, num_beams, ngram, wd, epochs, fp16, len_pen, warmup) = exp
    config = deepcopy(base_config)

    config['training']['per_device_train_batch_size'] = train_batch
    config['training']['per_device_eval_batch_size'] = eval_batch
    config['training']['learning_rate'] = lr
    config['training']['early_stopping_patience'] = patience
    config['training']['num_train_epochs'] = epochs
    config['training']['fp16'] = fp16
    config['training']['weight_decay'] = wd
    config['training']['warmup_steps'] = warmup

    config['tokenizer']['encoder_max_len'] = encoder_max
    config['tokenizer']['decoder_max_len'] = decoder_max
    config['inference']['generate_max_length'] = decoder_max
    config['inference']['num_beams'] = num_beams
    config['inference']['no_repeat_ngram_size'] = ngram
    config['inference']['length_penalty'] = len_pen

    config['wandb']['name'] = exp_name
    
    config['inference']['remove_tokens'] = [
        "<usr>", "<s>", "</s>", "<pad>"
    ]

    # 저장 (src/configs/폴더에 실험별 yaml 파일)
    with open(os.path.join(save_dir, f"{exp_name}.yaml"), "w") as f:
        yaml.dump(config, f, allow_unicode=True)

print("✅ 모든 실험 config 파일 생성 완료 (src/configs/*.yaml)")
