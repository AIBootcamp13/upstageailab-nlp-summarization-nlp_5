#!/usr/bin/env python
"""
🚀 Final Boost Inference Script
기존 best 모델을 활용한 최적화된 추론 실행
"""

import argparse
import yaml
import torch
import sys
import os

# 프로젝트 루트 경로 설정
project_dir = "/data/ephemeral/home/nlp-5/lyj"
sys.path.append(project_dir)

from models.BART import load_tokenizer_and_model_for_inference
from inference.inference_modified import inference
from dataset.preprocess import Preprocess

def main():
    parser = argparse.ArgumentParser(description='Final Boost Inference')
    parser.add_argument('--config', type=str, default='config_final_boost.yaml',
                       help='Config file name in src/configs/')
    args = parser.parse_args()
    
    # Config 파일 로드
    config_path = os.path.join(project_dir, 'src/configs', args.config)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    print("🚀 Final Boost Inference Started!")
    print(f"📁 Config: {args.config}")
    print(f"🎯 Model checkpoint: {config['inference']['ckt_dir']}")
    print(f"📊 Target score: {config.get('experiment', {}).get('target_score', 'N/A')}")
    
    # 디바이스 설정
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    # 모델 및 토크나이저 로드
    print("📦 Loading model and tokenizer...")
    model, tokenizer = load_tokenizer_and_model_for_inference(config, device)
    
    # 추론 실행
    print("🔮 Starting inference...")
    result = inference(config, model, tokenizer)
    
    print(f"✅ Inference completed!")
    print(f"📁 Results saved to: {config['inference']['result_path']}/result1")
    print(f"📊 Generated {len(result)} summaries")
    
    # 샘플 결과 출력
    print("\n🔍 Sample results:")
    for i in range(min(3, len(result))):
        print(f"  {result.iloc[i]['fname']}: {result.iloc[i]['summary'][:100]}...")

if __name__ == "__main__":
    main()
