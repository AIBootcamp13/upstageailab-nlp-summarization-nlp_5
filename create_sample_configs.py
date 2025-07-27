#!/usr/bin/env python3
"""샘플 실험 설정 파일 생성 스크립트"""

import os
import yaml
from pathlib import Path

def create_sample_configs():
    """샘플 실험 설정 파일들을 생성합니다."""
    
    # config/experiments 디렉토리 확인
    config_dir = Path("config/experiments")
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # 샘플 설정들
    sample_configs = {
        "01_baseline.yaml": {
            "experiment_name": "baseline",
            "description": "Baseline KoBART model experiment",
            "model": {
                "name": "digit82/kobart-summarization"
            },
            "training": {
                "learning_rate": 1e-5,
                "per_device_train_batch_size": 8,
                "num_train_epochs": 5
            },
            "wandb": {
                "name": "01_baseline_experiment"
            }
        },
        "02_high_lr.yaml": {
            "experiment_name": "high_learning_rate",
            "description": "Experiment with higher learning rate",
            "model": {
                "name": "digit82/kobart-summarization"
            },
            "training": {
                "learning_rate": 5e-5,
                "per_device_train_batch_size": 8,
                "num_train_epochs": 5
            },
            "wandb": {
                "name": "02_high_lr_experiment"
            }
        },
        "03_large_batch.yaml": {
            "experiment_name": "large_batch_size",
            "description": "Experiment with larger batch size",
            "model": {
                "name": "digit82/kobart-summarization"
            },
            "training": {
                "learning_rate": 1e-5,
                "per_device_train_batch_size": 16,
                "num_train_epochs": 5
            },
            "wandb": {
                "name": "03_large_batch_experiment"
            }
        },
        "04_longer_training.yaml": {
            "experiment_name": "longer_training",
            "description": "Experiment with more epochs",
            "model": {
                "name": "digit82/kobart-summarization"
            },
            "training": {
                "learning_rate": 1e-5,
                "per_device_train_batch_size": 8,
                "num_train_epochs": 10
            },
            "wandb": {
                "name": "04_longer_training_experiment"
            }
        }
    }
    
    # 설정 파일 생성
    for filename, config in sample_configs.items():
        file_path = config_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        print(f"✅ 샘플 설정 생성: {file_path}")
    
    print(f"\n📁 총 {len(sample_configs)}개 샘플 설정 파일 생성 완료")
    print(f"🚀 실행 방법: bash run_auto_experiments.sh")

if __name__ == "__main__":
    create_sample_configs()
