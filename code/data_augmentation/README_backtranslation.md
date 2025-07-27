# 백트랜슬레이션 데이터 증강

## 개요
백트랜슬레이션(Back-translation)을 통해 의역된 데이터를 생성하여 학습 데이터의 다양성을 높입니다. 한국어→영어→한국어 번역 과정에서 자연스럽게 표현이 바뀌면서도 의미는 유지되는 증강 데이터를 얻을 수 있습니다.

## 주요 특징

### 1. 다중 번역 엔진 지원
- **Google Translate**: 고품질 번역 (API 할당량 제한)
- **MarianMT**: 로컬 실행 가능한 오픈소스 모델
- **Both**: 두 엔진을 번갈아 사용하여 다양성 극대화

### 2. 다중 언어 백트랜슬레이션
- 단일 중간 언어 (기본: 영어)
- 다중 중간 언어 (영어, 일본어, 중국어, 스페인어 등)
- 각 언어의 특성을 활용한 다양한 의역 생성

### 3. 품질 관리
- **의미 유사도 검증**: 원문과 너무 다르면 제외
- **특수 토큰 보존**: PII 토큰과 화자 구분 토큰 완벽 보존
- **길이 제약**: 원문 대비 0.5~2.0배 범위 유지
- **문법성 검사**: 기본적인 문법 오류 필터링

### 4. 성능 최적화
- **캐싱 시스템**: 동일한 텍스트 재번역 방지
- **배치 처리**: 효율적인 대량 처리
- **Rate Limiting**: API 할당량 관리

## 사용 방법

### 1. 의존성 설치
```bash
# Google Translate 사용 시
pip install googletrans==4.0.0-rc1

# MarianMT 사용 시
pip install transformers torch

# 품질 평가용
pip install sentence-transformers scikit-learn
```

### 2. 단일 언어 백트랜슬레이션
```bash
python scripts/run_backtranslation.py \
    --config config/experiments/09_backtranslation.yaml \
    --data-dir ./data \
    --output-dir ./data/augmented \
    --evaluate
```

### 3. 다중 언어 백트랜슬레이션
```bash
python scripts/run_backtranslation.py \
    --config config/experiments/09_multilingual_backtranslation.yaml \
    --data-dir ./data \
    --output-dir ./data/augmented \
    --evaluate
```

### 4. 코드에서 직접 사용
```python
from code.data_augmentation.backtranslation import BackTranslationAugmenter

# 증강기 생성
augmenter = BackTranslationAugmenter(
    method="google",
    source_lang="ko",
    intermediate_lang="en",
    quality_threshold=0.3,
    max_similarity=0.9
)

# 텍스트 증강
original = "#Person1#: 오늘 회의는 3시에 시작합니다."
augmented = augmenter.augment(original, num_augmentations=3)

# DataFrame 증강
augmented_df = augmenter.augment_dataframe(
    df=train_df,
    text_column='dialogue',
    num_augmentations=2
)
```

## 설정 옵션

### 기본 설정
```yaml
backtranslation:
  method: "google"  # "google", "marian", "both"
  source_lang: "ko"
  intermediate_lang: "en"
  quality_threshold: 0.3  # 최소 차이
  max_similarity: 0.9     # 최대 유사도
  augmentation_ratio: 0.5 # 원본의 50% 증강
```

### 다중 언어 설정
```yaml
backtranslation:
  augmenter_type: "multilingual"
  intermediate_langs: ["en", "ja", "zh-cn", "es", "fr"]
  language_weights:
    en: 0.3
    ja: 0.25
    zh-cn: 0.2
    es: 0.15
    fr: 0.1
```

## 품질 평가 메트릭

### 1. 의미 유사도
- Sentence-BERT 임베딩 기반 코사인 유사도
- 목표 범위: 0.65 ~ 0.90

### 2. 어휘 다양성
- 원본과 다른 단어의 비율
- 목표: 20% 이상

### 3. 특수 토큰 보존율
- PII 토큰과 화자 구분 토큰의 완전한 보존
- 목표: 100%

### 4. 문법성 점수
- 문장 부호, 괄호 짝, 공백 등 기본 문법 체크
- 목표: 0.8 이상

## 예시

### 입력 (원본)
```
#Person1#: 안녕하세요, 오늘 회의는 몇 시에 시작하나요?
#Person2#: 오후 3시에 시작할 예정입니다. 회의실은 #Address# 5층입니다.
```

### 출력 (백트랜슬레이션)
```
#Person1#: 안녕하세요, 오늘 미팅이 언제 시작되나요?
#Person2#: 오후 3시에 시작될 예정입니다. 회의실은 #Address# 5층에 있습니다.
```

주요 변화:
- "회의" → "미팅"
- "몇 시에 시작하나요?" → "언제 시작되나요?"
- "5층입니다" → "5층에 있습니다"

## 주의사항

### 1. API 제한
- Google Translate: 무료 할당량 제한 (시간당 요청 수)
- 대량 처리 시 캐싱과 rate limiting 필수

### 2. 품질 변동
- 중간 언어에 따라 번역 품질 차이
- 기술 용어나 은어는 왜곡 가능성

### 3. 특수 토큰 처리
- 번역 과정에서 특수 토큰 손실 방지를 위해 placeholder 사용
- 번역 후 반드시 복원 과정 필요

## 성능 향상 팁

### 1. 최적 중간 언어 선택
- **영어**: 가장 일반적, 안정적
- **일본어**: 한국어와 문법 구조 유사
- **중국어**: 간결한 표현으로 압축 효과

### 2. 품질 임계값 조정
- 너무 엄격하면 증강 데이터 부족
- 너무 느슨하면 품질 저하
- 실험을 통해 최적값 탐색

### 3. 조합 전략
- 백트랜슬레이션 + 동의어 치환
- 백트랜슬레이션 + 문장 재배열
- 다단계 증강으로 다양성 극대화

## 문제 해결

### ImportError: googletrans
```bash
pip uninstall googletrans
pip install googletrans==4.0.0-rc1
```

### API 할당량 초과
- Rate limiting 지연 시간 증가
- 캐시 활용도 높이기
- MarianMT로 전환

### 특수 토큰 손실
- placeholder 매핑 확인
- 정규표현식 패턴 검증
- 복원 로직 디버깅

## 예상 효과

- **데이터 다양성**: 30-50% 향상
- **ROUGE-F1**: +2-3% 향상
- **과적합 감소**: 검증 손실 안정화
- **일반화 성능**: 새로운 표현에 대한 강건성 향상

## 참고 자료

- [Back-translation for NLP](https://arxiv.org/abs/1808.09381)
- [Google Translate API](https://cloud.google.com/translate)
- [MarianMT Models](https://huggingface.co/Helsinki-NLP)
- [Sentence-BERT](https://www.sbert.net/)
