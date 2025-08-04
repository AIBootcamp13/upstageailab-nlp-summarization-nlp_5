#!/usr/bin/env python3
import yaml
import sys

try:
    with open('/Users/jayden/Developer/Projects/nlp-5/nlp-sum-lyj/config/experiments/mt5_xlsum_ultimate_korean_qlora.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 기본 구조 검증
    required_sections = ['experiment_name', 'general', 'model', 'tokenizer', 'training', 'qlora', 'wandb', 'generation', 'inference']
    
    print("=== YAML 구조 검증 ===")
    for section in required_sections:
        if section in config:
            print(f"✅ {section}: 존재")
        else:
            print(f"❌ {section}: 누락")
    
    # 중복 확인
    print("\n=== 중복 검사 ===")
    
    # target_modules 중복 확인
    if 'qlora' in config and 'target_modules' in config['qlora']:
        print(f"✅ target_modules: {config['qlora']['target_modules']}")
    
    # inference 섹션 중복 확인
    if 'inference' in config:
        inference_keys = list(config['inference'].keys())
        duplicates = [key for key in set(inference_keys) if inference_keys.count(key) > 1]
        if duplicates:
            print(f"❌ inference 중복 키: {duplicates}")
        else:
            print("✅ inference: 중복 없음")
    
    # remove_tokens 중복 확인 (특별 처리)
    if 'inference' in config and 'remove_tokens' in config['inference']:
        print(f"✅ remove_tokens: {config['inference']['remove_tokens']}")
    
    print("\n✅ YAML 파일 검증 완료!")
    
except yaml.YAMLError as e:
    print(f"❌ YAML 문법 오류: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ 기타 오류: {e}")
    sys.exit(1)
