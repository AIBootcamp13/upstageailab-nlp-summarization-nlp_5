# 🚀 모델 자동 다운로드 설정 가이드

## 📋 개요

이 프로젝트는 `pytorch_model.bin` 파일 없이도 HuggingFace Hub에서 자동으로 모델을 다운로드하여 실험을 진행할 수 있도록 설계되었습니다.

## 🔧 설정 방법

### 1. 환경 변수 설정

```bash
# .env.template을 .env로 복사
cp .env.template .env

# .env 파일을 편집하여 실제 토큰 값 입력
vim .env
```

### 2. 필수 토큰 설정

```bash
# HuggingFace Hub Token (모델 다운로드 최적화)
HUGGINGFACE_HUB_TOKEN=hf_your_actual_token_here

# WandB API Key (실험 추적용)
WANDB_API_KEY=your_wandb_api_key_here

# Upstage API Key  
UPSTAGE_API_KEY=your_upstage_api_key_here
```

### 3. 모델 사전 다운로드 (선택사항)

실험 전에 미리 모델을 다운로드하여 실행 시간을 단축할 수 있습니다:

```bash
# 모델 사전 다운로드 실행
./setup_model_cache.sh
```

## 🎯 지원 모델

1. **csebuetnlp/mT5_multilingual_XLSum**
   - 다국어 텍스트 요약 모델
   - XL-Sum 데이터셋으로 학습된 mT5 기반 모델

2. **eenzeenee/t5-base-korean-summarization**
   - 한국어 텍스트 요약 특화 모델
   - T5 베이스 모델을 한국어로 파인튜닝

## 🔄 자동 다운로드 프로세스

1. **실험 시작 시**: 모델 파일 로컬 캐시 확인
2. **캐시 없음**: HuggingFace Hub에서 자동 다운로드
3. **캐시 저장**: 다음 실험 시 재사용
4. **실험 진행**: 정상적인 학습/추론 수행

## 📊 로그 예시

```
🌐 네트워크에서 csebuetnlp/mT5_multilingual_XLSum 다운로드 시도 (1/3)
✅ 네트워크에서 csebuetnlp/mT5_multilingual_XLSum 다운로드 성공
```

## 🛠️ 문제 해결

### 1. 네트워크 연결 실패
- `model_loading_utils.py`의 재시도 로직이 자동으로 처리
- 오프라인 캐시 활용으로 연속성 보장

### 2. 토큰 권한 오류
- HuggingFace 토큰의 권한 확인
- 필요 시 새 토큰 발급

### 3. 디스크 용량 부족
- 캐시 디렉토리 정리: `rm -rf ~/.cache/huggingface/`
- 불필요한 모델 삭제

## 📝 주의사항

- `.env` 파일은 절대 Git에 커밋하지 마세요 (보안 위험)
- HuggingFace 토큰은 개인별로 발급받아 사용하세요
- 모델 다운로드 시 인터넷 연결이 필요합니다

## 🎉 장점

✅ **간편성**: pytorch_model.bin 파일 관리 불필요  
✅ **자동화**: 실험 시 필요한 모델 자동 다운로드  
✅ **효율성**: 로컬 캐시로 재다운로드 방지  
✅ **안정성**: 네트워크 오류 시 자동 재시도  
✅ **확장성**: 새로운 HuggingFace 모델 쉽게 추가 가능
