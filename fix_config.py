#!/usr/bin/env python3
"""
실험 config 파일에 필수 섹션 추가 스크립트
trainer.py가 기대하는 config 구조를 보장합니다.
"""

import yaml
import sys
from pathlib import Path

def add_missing_sections(config_path):
    """config 파일에 누락된 필수 섹션 추가"""
    
    # config 파일 읽기
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 필수 섹션 기본값
    defaults = {
        'model': {
            'architecture': 'seq2seq',
            'checkpoint': config.get('general', {}).get('model_name', 'google/mt5-base')
        },
        'generation': {
            'max_length': config.get('inference', {}).get('generate_max_length', 100),
            'num_beams': config.get('inference', {}).get('num_beams', 4),
            'no_repeat_ngram_size': config.get('inference', {}).get('no_repeat_ngram_size', 2),
            'early_stopping': config.get('inference', {}).get('early_stopping', True)
        },
        'data': {
            'train_path': '../data/train.csv',
            'val_path': '../data/dev.csv',
            'test_path': '../data/test.csv'
        },
        'evaluation': {
            'rouge_use_stemmer': True,
            'rouge_tokenize_korean': True
        }
    }
    
    # 누락된 섹션 추가
    modified = False
    for section, values in defaults.items():
        if section not in config:
            config[section] = values
            modified = True
            print(f"✅ '{section}' 섹션 추가됨")
    
    # general 섹션에서 seed가 없으면 추가
    if 'general' not in config:
        config['general'] = {}
    if 'seed' not in config['general']:
        config['general']['seed'] = 42
        modified = True
        print("✅ 'general.seed' 추가됨")
    
    # 수정된 경우 백업 후 저장
    if modified:
        # 백업
        backup_path = Path(config_path).with_suffix('.yaml.backup')
        with open(backup_path, 'w', encoding='utf-8') as f:
            with open(config_path, 'r', encoding='utf-8') as orig:
                f.write(orig.read())
        print(f"📄 원본 백업: {backup_path}")
        
        # 수정된 config 저장
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, sort_keys=False)
        print(f"💾 수정된 config 저장: {config_path}")
    else:
        print("ℹ️  모든 필수 섹션이 이미 존재합니다.")
    
    return config

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python fix_config.py <config_file.yaml>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    if not Path(config_path).exists():
        print(f"❌ 파일을 찾을 수 없습니다: {config_path}")
        sys.exit(1)
    
    add_missing_sections(config_path)
