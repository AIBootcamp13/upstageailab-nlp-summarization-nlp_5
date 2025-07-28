# 팀원 요청사항 구현 문서

이 디렉토리는 팀원들의 요청사항에 대한 구현 내용과 가이드를 포함합니다.

## 문서 목록

### 1. [송규헌님 요청사항 구현](송규헌님_요청사항_구현.md)
- 코드 오류 수정 내역
- 다양한 모델 지원 구현
- unsloth 라이브러리 적용 방법

### 2. [다양한 모델 실험 가이드](다양한_모델_실험_가이드.md)
- 지원 모델 목록 및 특징
- 모델별 최적 설정값
- 실험 실행 방법
- 트러블슈팅 가이드

## 빠른 시작

### 1. 기본 실험 실행
```bash
# BART 모델로 실험
python code/trainer.py --config config/model_configs/bart_base.yaml

# T5 모델로 실험
python code/trainer.py --config config/model_configs/t5_base.yaml
```

### 2. 다중 모델 실험
```bash
# 모든 모델 순차 실행
./run_multi_model_experiments.sh
```

### 3. unsloth 최적화 실험
```bash
# unsloth 설치
./install_unsloth.sh

# KoBART + unsloth 실행
python code/trainer.py --config config/model_configs/kobart_unsloth.yaml
```

## 주요 개선사항

1. **확장성 향상**
   - `AutoModelForSeq2SeqLM`과 `AutoModelForCausalLM`을 통한 다양한 모델 지원
   - 모델별 전처리 함수 분리

2. **메모리 효율성**
   - QLoRA 지원으로 40% 메모리 사용량 감소
   - unsloth 지원으로 75% 메모리 사용량 감소

3. **실험 자동화**
   - 다중 모델 실험 스크립트
   - 결과 자동 수집 및 비교

## 추가 요청사항

새로운 요청사항이 있으시면 다음 형식으로 Issue를 생성해주세요:

```markdown
## 요청 내용
- 구체적인 기능 요청
- 예상 사용 시나리오
- 우선순위

## 기대 효과
- 성능 향상
- 사용성 개선
- 기타 효과
```

## 연락처

- 이영준: [이메일]
- Slack: #nlp-summarization 채널
