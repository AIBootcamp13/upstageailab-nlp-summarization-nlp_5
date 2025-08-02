#!/bin/bash
# 모델 자동 다운로드 및 캐시 최적화 스크립트
# Models Auto Download and Cache Optimization Script

echo "🚀 HuggingFace 모델 캐시 최적화 시작..."

# 환경 설정
export HUGGINGFACE_HUB_CACHE="/data/ephemeral/home/upstageailab-nlp-summarization-nlp_5/nlp-sum-lyj/.hf_cache"
export TRANSFORMERS_CACHE="/data/ephemeral/home/upstageailab-nlp-summarization-nlp_5/nlp-sum-lyj/.hf_cache"
export HF_HOME="/data/ephemeral/home/upstageailab-nlp-summarization-nlp_5/nlp-sum-lyj/.hf_cache"

# 캐시 디렉토리 생성
mkdir -p "$HUGGINGFACE_HUB_CACHE"

echo "📂 캐시 디렉토리 설정: $HUGGINGFACE_HUB_CACHE"

# Python을 사용한 모델 사전 다운로드
/opt/conda/bin/python3 -c "
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

print('🔗 환경 변수 확인...')
print(f'HF_HOME: {os.environ.get(\"HF_HOME\", \"Not Set\")}')
print(f'TRANSFORMERS_CACHE: {os.environ.get(\"TRANSFORMERS_CACHE\", \"Not Set\")}')

models = [
    'csebuetnlp/mT5_multilingual_XLSum',
    'eenzeenee/t5-base-korean-summarization'
]

for model_name in models:
    print(f'📥 {model_name} 다운로드 중...')
    try:
        # 토크나이저 다운로드
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=os.environ.get('HUGGINGFACE_HUB_CACHE')
        )
        print(f'✅ {model_name} 토크나이저 다운로드 완료')
        
        # 모델 다운로드 (메모리 효율을 위해 torch_dtype 지정)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            cache_dir=os.environ.get('HUGGINGFACE_HUB_CACHE'),
            torch_dtype=torch.float16,
            device_map='auto'
        )
        print(f'✅ {model_name} 모델 다운로드 완료')
        
        # 메모리 정리
        del model
        del tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        print(f'❌ {model_name} 다운로드 실패: {e}')
        continue

print('🎉 모든 모델 다운로드 완료!')
"

# 캐시 정보 출력
echo "📊 캐시 상태 확인..."
du -sh "$HUGGINGFACE_HUB_CACHE" 2>/dev/null || echo "캐시 디렉토리 크기 확인 실패"
ls -la "$HUGGINGFACE_HUB_CACHE" 2>/dev/null || echo "캐시 디렉토리가 아직 생성되지 않음"

echo "✨ 모델 자동 다운로드 설정 완료!"
echo "이제 실험 시 모델이 인터넷에서 자동으로 다운로드됩니다."
