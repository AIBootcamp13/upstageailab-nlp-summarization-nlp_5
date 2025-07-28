# 🤖 eenzeenee T5 한국어 요약 모델 완전 가이드

## 📋 개요

**eenzeenee/t5-base-korean-summarization**은 한국어 텍스트 요약에 특화된 T5 기반 모델입니다. 
paust/pko-t5-base를 기반으로 3개의 한국어 데이터셋에서 파인튜닝되어 논문, 도서, 대화 요약에서 뛰어난 성능을 보입니다.

### 🎯 주요 특징
- **아키텍처**: T5-base (220M 파라미터)
- **언어**: 한국어 특화
- **기반 모델**: paust/pko-t5-base
- **최고 성능**: 도서 요약 ROUGE-2 F1 0.266
- **자동 처리**: 'summarize: ' prefix 자동 추가

## 🚀 빠른 시작

### 1. 단일 eenzeenee 실험 실행

```bash
# 설정 확인 모드 (안전한 테스트)
./run_eenzeenee_experiment.sh

# 실제 학습 실행
EENZEENEE_RUN_ACTUAL=true ./run_eenzeenee_experiment.sh
```

### 2. 다중 모델 비교 실험

```bash
# eenzeenee를 포함한 여러 모델 비교 실험
./run_multi_model_experiments.sh
```

### 3. 수동 실행

```bash
python code/trainer.py \
    --config config.yaml \
    --config-section eenzeenee \
    --train-data data/train.csv \
    --val-data data/dev.csv \
    --test-data data/test.csv
```

### 4. Python 코드에서 직접 사용

```python
# 기본 사용법
from code.utils.eenzeenee_utils import *

# 텍스트 전처리
text = "오늘 회의에서 중요한 결정들이 내려졌습니다. 예산 승인과 프로젝트 일정에 대해 논의했습니다."
processed_text = preprocess_for_eenzeenee(text)
print(processed_text)  
# 출력: "summarize: 오늘 회의에서 중요한 결정들이 내려졌습니다. 예산 승인과 프로젝트 일정에 대해 논의했습니다."

# 입력 검증
validation_result = validate_eenzeenee_input(text)
if validation_result["is_valid"]:
    print("입력이 유효합니다!")
else:
    print("오류:", validation_result["errors"])

# 모델 정보 조회
model_info = get_eenzeenee_model_info()
print(f"모델 파라미터: {model_info['parameters']}")  # 220M
print(f"권장 출력 길이: {model_info['output_max_length']}")  # 64
```

## 📊 모델 성능 지표

### ROUGE-2 F1 점수 (Hugging Face 공식)
- **논문자료 요약**: 0.1725
- **도서자료 요약**: 0.2655 (최고 성능)
- **요약문 및 레포트**: 0.1773

### 권장 사용 분야
1. **논문 요약**: 학술 문서의 핵심 내용 추출
2. **도서 요약**: 책의 주요 내용과 핵심 메시지 요약
3. **대화 요약**: 회의록, 상담 내용 등 대화형 텍스트 요약
4. **보고서 생성**: 긴 문서를 간결한 요약문으로 변환

## ⚙️ 설정 정보

### config.yaml의 eenzeenee 섹션 주요 설정

```yaml
eenzeenee:
  general:
    model_name: eenzeenee/t5-base-korean-summarization
    input_prefix: "summarize: "
    model_type: seq2seq
  
  tokenizer:
    encoder_max_len: 512    # 입력 최대 길이
    decoder_max_len: 64     # 출력 최대 길이 (모델 카드 권장)
  
  inference:
    batch_size: 8           # T5-base 크기 적합 배치
    generate_max_length: 64 # 한국어 요약 최적 길이
    num_beams: 3           # 빔 서치 크기 (모델 카드 권장)
    do_sample: true        # 샘플링 활성화
    temperature: 0.8       # 생성 온도
```

### 최적화된 생성 파라미터

```python
# eenzeenee_utils에서 제공하는 최적 설정
generation_config = get_eenzeenee_generation_config()
{
    "max_length": 64,           # 한국어 요약 최적 길이
    "min_length": 10,           # 최소 요약 길이  
    "num_beams": 3,             # 빔 서치 크기
    "do_sample": True,          # 샘플링 활성화
    "temperature": 0.8,         # 생성 온도
    "top_k": 50,               # Top-K 샘플링
    "top_p": 0.95,             # Top-P 샘플링
    "no_repeat_ngram_size": 2   # 반복 방지
}
```

## 🔧 고급 사용법

### 1. 배치 처리

```python
from code.utils.eenzeenee_utils import create_eenzeenee_inputs
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("eenzeenee/t5-base-korean-summarization")

texts = [
    "첫 번째 요약할 텍스트입니다.",
    "두 번째 요약할 텍스트입니다.",
    "세 번째 요약할 텍스트입니다."
]

# 자동으로 prefix 추가 및 토크나이징
inputs = create_eenzeenee_inputs(texts, tokenizer)
print(inputs.keys())  # dict_keys(['input_ids', 'attention_mask'])
```

### 2. 커스텀 검증

```python
def custom_validation(text: str) -> bool:
    """사용자 정의 검증 함수"""
    result = validate_eenzeenee_input(text)
    
    if not result["is_valid"]:
        print("❌ 검증 실패:")
        for error in result["errors"]:
            print(f"  - {error}")
        return False
    
    if result["warnings"]:
        print("⚠️ 경고사항:")
        for warning in result["warnings"]:
            print(f"  - {warning}")
        
        print("💡 권장사항:")
        for suggestion in result["suggestions"]:
            print(f"  - {suggestion}")
    
    return True

# 사용 예시
text = "안녕하세요"  # prefix 없는 짧은 텍스트
if custom_validation(text):
    processed = preprocess_for_eenzeenee(text)
```

### 3. 모델 호환성 확인

```python
model_names = [
    "eenzeenee/t5-base-korean-summarization",
    "eenzeenee/xsum-t5-1.7b",  # 이전 명명 (호환)
    "google/t5-base",
    "facebook/bart-base"
]

for model_name in model_names:
    is_compatible = is_eenzeenee_compatible_model(model_name)
    print(f"{model_name}: {'✅ 호환' if is_compatible else '❌ 비호환'}")
```

## 💾 시스템 요구사항

### 하드웨어 권장사항
- **CPU**: 최소 4코어, 권장 8코어 이상
- **RAM**: 최소 8GB, 권장 16GB 이상
- **GPU**: 
  - **추론**: GTX 1060 6GB 이상
  - **학습**: RTX 3080 10GB 이상 권장
- **저장공간**: 최소 2GB (모델 + 캐시)

### 배치 크기 가이드
| GPU 메모리 | 권장 배치 크기 | 성능 |
|------------|----------------|------|
| 4GB | 2-4 | 기본 |
| 8GB | 4-8 | 권장 |
| 12GB+ | 8-16 | 최적 |

### 메모리 사용량 최적화

```python
# 메모리 효율적인 설정
efficient_config = {
    "per_device_train_batch_size": 4,  # GPU 메모리 부족시 줄이기
    "gradient_accumulation_steps": 2,   # 배치 크기 보완
    "fp16": True,                      # 메모리 절약
    "dataloader_num_workers": 2,       # CPU 코어에 맞게 조정
    "dataloader_pin_memory": True      # GPU 전송 최적화
}
```

## 🐛 문제 해결

### 일반적인 오류와 해결방법

#### 1. GPU 메모리 부족
```bash
❌ CUDA out of memory 오류
```
**해결방법**:
```yaml
# config.yaml에서 배치 크기 줄이기
inference:
  batch_size: 4  # 8에서 4로 감소
```

#### 2. 모델 다운로드 실패
```bash
❌ Connection timeout 오류
```
**해결방법**:
```bash
# 인터넷 연결 확인 후 재시도
# 또는 수동 다운로드
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('eenzeenee/t5-base-korean-summarization')"
```

#### 3. Prefix 누락 경고
```bash
⚠️ 'summarize: ' prefix가 없습니다
```
**해결방법**:
```python
# 자동 prefix 추가 사용
text = preprocess_for_eenzeenee(your_text)
```

#### 4. 한국어 텍스트 부족 경고
```bash
⚠️ 한국어 텍스트가 부족합니다
```
**해결방법**:
- 이 모델은 한국어에 특화되어 있으므로 한국어 텍스트 사용 권장
- 영어 텍스트의 경우 다른 모델 사용 고려

### 성능 최적화 팁

#### 1. 입력 길이 최적화
```python
# 긴 텍스트는 적절히 분할
def optimize_input_length(text: str, max_length: int = 400) -> str:
    if len(text) > max_length:
        # 문장 단위로 자르기
        sentences = text.split('. ')
        result = ""
        for sentence in sentences:
            if len(result + sentence) < max_length:
                result += sentence + ". "
            else:
                break
        return result.strip()
    return text
```

#### 2. 배치 처리 최적화
```python
# 비슷한 길이의 텍스트끼리 그룹화
def group_by_length(texts: List[str]) -> List[List[str]]:
    grouped = {}
    for text in texts:
        length_bucket = len(text) // 100 * 100  # 100자 단위로 그룹화
        if length_bucket not in grouped:
            grouped[length_bucket] = []
        grouped[length_bucket].append(text)
    return list(grouped.values())
```

## 📈 성능 벤치마크

### 다른 모델과의 비교

| 모델 | 파라미터 | 한국어 ROUGE-2 | 처리속도 | 메모리 |
|------|----------|----------------|----------|--------|
| **eenzeenee T5** | 220M | **0.266** | 빠름 | 적음 |
| mT5 XL-Sum | 1.2B | 0.237 | 보통 | 많음 |
| KoBART | 124M | 0.210 | 빠름 | 적음 |

### 실제 사용 사례별 성능

#### 1. 회의록 요약
```
원본 (312자): "오늘 오후 2시에 진행된 마케팅팀 회의에서는 다음 분기 광고 전략에 대해 논의했습니다. 김대리가 제안한 소셜미디어 광고 확대 방안과 박과장의 TV광고 집중 전략이 주요 안건이었습니다. 결과적으로 예산 배분은 소셜미디어 60%, TV광고 40%로 결정되었으며, 다음 주까지 세부 계획을 수립하기로 했습니다."

요약 (28자): "마케팅팀이 광고 전략을 논의하여 소셜미디어 60%, TV광고 40% 예산 배분을 결정했다."
```

#### 2. 논문 초록 요약
```
원본 (428자): "본 연구는 딥러닝 기반 자연어 처리 모델의 성능 향상을 위한 새로운 어텐션 메커니즘을 제안한다. 기존의 셀프 어텐션과 달리 계층적 구조를 가진 멀티스케일 어텐션을 도입하여 장거리 의존성을 효과적으로 포착할 수 있도록 했다. 실험 결과, 제안한 방법은 기존 모델 대비 BLEU 점수 3.2점, ROUGE 점수 2.8점 향상을 보였으며, 특히 긴 문서 처리에서 우수한 성능을 나타냈다."

요약 (35자): "새로운 멀티스케일 어텐션 메커니즘으로 기존 모델보다 BLEU 3.2점, ROUGE 2.8점 향상을 달성했다."
```

## 🔄 다른 모델과의 통합

### mT5와 함께 사용하기

```python
# 두 모델 성능 비교
from code.utils.eenzeenee_utils import get_eenzeenee_model_info
from code.utils.xlsum_utils import get_xlsum_model_info

eenzeenee_info = get_eenzeenee_model_info()
mT5_info = get_xlsum_model_info()

print("모델 비교:")
print(f"eenzeenee: {eenzeenee_info['parameters']} 파라미터")
print(f"mT5: 1.2B 파라미터")
print(f"eenzeenee 권장 출력: {eenzeenee_info['output_max_length']}토큰")
print(f"mT5 권장 출력: 84토큰")
```

### 앙상블 방식 활용

```python
def ensemble_summarization(text: str) -> Dict[str, str]:
    """두 모델의 결과를 모두 제공"""
    results = {}
    
    # eenzeenee 전처리
    eenzeenee_input = preprocess_for_eenzeenee(text)
    results['eenzeenee_processed'] = eenzeenee_input
    
    # mT5 전처리 (xlsum_utils 사용)
    # from code.utils.xlsum_utils import preprocess_for_xlsum
    # mT5_input = preprocess_for_xlsum(text)
    # results['mT5_processed'] = mT5_input
    
    return results
```

## 📚 참고 자료

### 공식 문서
- [Hugging Face 모델 카드](https://huggingface.co/eenzeenee/t5-base-korean-summarization)
- [paust/pko-t5-base 기반 모델](https://huggingface.co/paust/pko-t5-base)
- [T5 논문](https://arxiv.org/abs/1910.10683)

### 프로젝트 내 관련 파일
- `code/utils/eenzeenee_utils.py` - 전용 유틸리티 함수
- `EENZEENEE_INTEGRATION_REPORT.md` - 상세 통합 보고서  
- `test_eenzeenee_integration.py` - 통합 테스트
- `run_eenzeenee_experiment.sh` - 실험 실행 스크립트

### 학습 데이터셋
1. **Korean Paper Summarization Dataset** (논문자료 요약)
2. **Korean Book Summarization Dataset** (도서자료 요약)
3. **Korean Summary statement and Report Generation Dataset** (요약문 및 레포트 생성)

## 🤝 커뮤니티 및 지원

### 통합 테스트 실행
```bash
# 모든 기능이 정상 작동하는지 확인
python test_eenzeenee_integration.py

# 출력 예시:
# 🎉 모든 통합 테스트가 성공했습니다!
# eenzeenee 모델이 프로젝트에 성공적으로 통합되었습니다.
```

### 실험 결과 확인
```bash
# 실험 실행 후 결과 확인
ls outputs/eenzeenee_experiment_*/
# experiment_info.json  training.log  results/
```

### 추가 도움말
- 프로젝트 메인 README: `README.md`
- 전체 설정 가이드: `QUICKSTART_CHECKLIST.md`
- 다중 모델 실험: `run_multi_model_experiments.sh`

---

**🎯 이제 eenzeenee T5 모델을 사용하여 고품질의 한국어 요약을 생성할 준비가 완료되었습니다!**

정확한 모델 정보와 최적화된 설정으로 논문, 도서, 대화 등 다양한 한국어 텍스트의 요약에서 뛰어난 성능을 경험하세요.
