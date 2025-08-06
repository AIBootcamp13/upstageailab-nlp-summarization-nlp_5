#!/usr/bin/env python
"""
⚡ Quick Inference Script (WandB 없이)
기존 모델로 빠른 추론 실행
"""

import os
import torch
import yaml
import sys
import pandas as pd

# 프로젝트 루트 추가
sys.path.append("/data/ephemeral/home/nlp-5/lyj")

from models.BART import load_tokenizer_and_model_for_inference
from inference.inference_modified import inference
from dataset.preprocess import Preprocess

def quick_inference(config_name="config_quick_boost.yaml"):
    """빠른 추론 실행"""
    
    # Config 로드
    config_path = f"/data/ephemeral/home/nlp-5/lyj/src/configs/{config_name}"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    print(f"⚡ Quick Inference Started!")
    print(f"📁 Config: {config_name}")
    print(f"🎯 Using model: {config['inference']['ckt_dir']}")
    
    # 디바이스 설정
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    try:
        # 모델 및 토크나이저 로드
        print("📦 Loading model and tokenizer...")
        model, tokenizer = load_tokenizer_and_model_for_inference(config, device)
        
        # 추론 실행
        print("🔮 Starting inference...")
        result = inference(config, model, tokenizer)
        
        print(f"✅ Inference completed!")
        print(f"📁 Results saved to: {config['inference']['result_path']}/result1")
        print(f"📊 Generated {len(result)} summaries")
        
        return result
        
    except Exception as e:
        print(f"❌ Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config_quick_boost.yaml')
    args = parser.parse_args()
    
    quick_inference(args.config)
