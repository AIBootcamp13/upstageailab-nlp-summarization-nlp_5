# DialogSum 데이터셋 상세 분석 가이드

## 목차
1. [데이터셋 개요](#1-데이터셋-개요)
2. [원본과 대회 데이터 비교](#2-원본과-대회-데이터-비교)
3. [데이터 품질 분석](#3-데이터-품질-분석)
4. [Special Token 및 약어 처리](#4-special-token-및-약어-처리)
5. [적용 방안](#5-적용-방안)

---

## 1. 데이터셋 개요

### 1.1 DialogSum 소개
- **출처**: DialogSum 논문의 공식 데이터셋
- **목적**: 일상 대화 요약 (2인 이상 대화)
- **특징**: 
  - 영어 원본을 한글로 번역
  - 다양한 일상 주제 포함
  - 발화자별 구분 (#Person1#, #Person2# 등)

### 1.2 데이터 구성
```python
# 원본 데이터 다운로드
import pandas as pd

splits = {'train': 'train.csv', 'validation': 'validation.csv', 'test': 'test.csv'}
train_df = pd.read_csv("hf://datasets/knkarthick/dialogsum/" + splits["train"])
val_df = pd.read_csv("hf://datasets/knkarthick/dialogsum/" + splits["validation"])
test_df = pd.read_csv("hf://datasets/knkarthick/dialogsum/" + splits["test"])
```

## 2. 원본과 대회 데이터 비교

### 2.1 데이터 크기 차이

| 구분 | 원본 (영어) | 대회 (한글) | 차이 |
|------|------------|------------|------|
| Train | 12,460 | 12,457 | -3 |
| Validation | 500 | 499 | -1 |
| Test | 1,500 | 250 | -1,250 |

### 2.2 제외된 데이터 분석

#### 2.2.1 Train 데이터에서 제외
```python
# 제외된 3개 데이터
- train_10933: topic = "after-sales service"
- train_10972: topic = "after-sales service"  
- train_11473: topic = "after-sales service"
```

#### 2.2.2 Validation 데이터에서 제외
```python
# 제외된 1개 데이터
- dev_475: topic = "buy furniture"
```

**분석 결과**:
- after-sales service 관련 대화가 주로 제외됨
- 특별한 패턴이나 이유는 명확하지 않음
- 가구 구매 관련 대화도 일부 제외

### 2.3 Test 데이터 구조 차이
- **원본**: 1,500개, summary 1개
- **대회**: 250개 (public) + 249개 (hidden)
- **특징**: 대회는 summary 3개로 평가 (다양성 고려)

## 3. 데이터 품질 분석

### 3.1 요약문 품질 이슈

#### 예시: train_5000
```python
dialogue = """
#Person1#: 너 동물 좋아해? 난 진짜 개 좋아해.
#Person2#: 나도 그래. 근데 난 고양이는 별로야.
#Person1#: 왜? 난 고양이도 괜찮던데.
#Person2#: 고양이 근처에는 못 있겠어. 걔네들도 나를 별로 안 좋아하는 것 같아.
#Person1#: 난 야생동물을 좋아해. 거미랑 뱀은 싫어. 왠지 좀 무섭잖아.
#Person2#: 난 뱀이 좋아. 참 멋있는 것 같아. 거미에 대해선 동감이야. 다리가 너무 많아서 싫어.
#Person1#: 난 곰이 멋진 것 같아. 판다는 정말 환상적이야. 털 때문에 죽이는 사람들 이해 안 돼.
#Person2#: 동감이야. 난 쥐가 너무 귀여운 것 같아.
#Person1#: 정말? 난 그 매력을 모르겠어. 쥐가 좀 무서워.
"""

summary = """
#Person1#는 개와 야생동물을 좋아하지만 거미와 뱀은 무서워합니다. 
#Person2#는 고양이는 안 좋아하지만 뱀과 쥐를 좋아합니다.
"""

# 누락된 정보:
# - Person1: 곰과 판다 좋아함, 쥐가 무서움
# - Person2: 거미 싫어함
```

### 3.2 주요 문제점

1. **정보 누락**
   - 중요한 선호/혐오 정보 누락
   - 감정 표현 누락

2. **표현 불일치**
   - 무섭다/싫다 혼용
   - 좋다/멋있다 혼용

3. **Topic 일관성**
   - 영어 → 한글 번역 시 일관성 부족
   - 3단어 제한으로 인한 정보 손실

## 4. Special Token 및 약어 처리

### 4.1 토크나이저 문제점

```python
from transformers import AutoTokenizer

# KoBART tokenizer 문제
tokenizer = AutoTokenizer.from_pretrained("digit82/kobart-summarization")
print(tokenizer.tokenize("ATM"))  # ['_A', 'T', 'M'] - 부적절한 분리
print(tokenizer.tokenize("A/S"))  # ['_A', '/', 'S']
print(tokenizer.tokenize("EDD"))  # ['_E', 'D', 'D']

# mT5 tokenizer (더 나은 처리)
tokenizer_mt5 = AutoTokenizer.from_pretrained("csebuetnlp/mT5_multilingual_XLSum")
print(tokenizer_mt5.tokenize("ATM"))  # ['▁ATM']
```

### 4.2 발견된 약어 및 특수 표현

#### 4.2.1 약어
```python
abbreviations = [
    'ATM', 'A/S', 'AS', 'BBC', 'CEO', 'CPR', 
    'EDD', 'ETV', 'Kts', 'LCD', 'GPS', 'USB'
]
```

#### 4.2.2 인물명
```python
person_names = [
    '알버트 아인슈타인', '에이브러햄 링컨', '스티브 잡스',
    '빌 게이츠', '마크 저커버그', '일론 머스크'
]
```

### 4.3 Topic Dictionary 생성

```python
# topic_dict_cleaned.yaml 구조
topic_dict = {
    'get a check-up': '건강검진 받기',
    'vaccines': '백신',
    'find keys': '열쇠 찾기',
    'have a girlfriend': '여자친구가 있다',
    'dance': '댄스',
    'after-sales service': 'A/S 서비스',
    'buy furniture': '가구 구입',
    # ... 총 6,526개 매핑
}
```

## 5. 적용 방안

### 5.1 즉시 적용 가능한 전처리

```python
class DialoguePreprocessor:
    def __init__(self):
        self.load_resources()
        
    def load_resources(self):
        """리소스 로드"""
        with open('data/topic_dict_cleaned.yaml', 'r', encoding='utf-8') as f:
            self.topic_dict = yaml.safe_load(f)
        
        # 약어 및 인물명 로드
        self.abbreviations = self.extract_abbreviations()
        self.person_names = self.extract_person_names()
    
    def preprocess(self, text):
        """통합 전처리"""
        # 1. 약어 처리
        text = self.handle_abbreviations(text)
        
        # 2. 인물명 처리
        text = self.handle_person_names(text)
        
        # 3. 구어체 정제
        text = self.clean_colloquial(text)
        
        # 4. 특수문자 정규화
        text = self.normalize_special_chars(text)
        
        return text
    
    def handle_abbreviations(self, text):
        """약어를 special token으로 변환"""
        for abbr in self.abbreviations:
            # A/S -> AS 통일
            if abbr == 'A/S':
                text = text.replace('A/S', 'AS')
            # 약어를 special token으로
            text = text.replace(abbr, f'#{abbr}#')
        return text
    
    def clean_colloquial(self, text):
        """구어체 표현 정제"""
        replacements = {
            r'ㅋ+': '웃음',
            r'ㅎ+': '웃음',
            r'ㅠ+': '슬픔',
            r'ㅜ+': '슬픔',
            'ㅇㅇ': '응',
            'ㄴㄴ': '아니'
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        
        return text
```

### 5.2 토크나이저 설정

```python
def setup_tokenizer(model_name="gogamza/kobart-base-v2"):
    """토크나이저 설정 및 special token 추가"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Special tokens 정의
    special_tokens = []
    
    # 1. 발화자 토큰
    special_tokens.extend([f'#Person{i}#' for i in range(1, 8)])
    
    # 2. PII 마스킹 토큰
    special_tokens.extend([
        '#PhoneNumber#', '#Address#', '#DateOfBirth#',
        '#PassportNumber#', '#SSN#', '#CardNumber#',
        '#CarNumber#', '#Email#'
    ])
    
    # 3. 약어 토큰
    special_tokens.extend([f'#{abbr}#' for abbr in abbreviations])
    
    # 4. 인물명 토큰 (선택적)
    # special_tokens.extend([f'#NAME_{name.replace(" ", "_")}#' for name in person_names])
    
    # 토크나이저에 추가
    special_tokens_dict = {'additional_special_tokens': special_tokens}
    num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    
    print(f"추가된 special token 수: {num_added_tokens}")
    
    return tokenizer
```

### 5.3 Solar API 활용 자동화

```python
import asyncio
from upstage import AsyncUpstage

class SolarEnhancer:
    def __init__(self, api_key):
        self.client = AsyncUpstage(api_key=api_key)
    
    async def detect_abbreviations(self, text):
        """텍스트에서 약어 자동 감지"""
        prompt = f"""
        다음 텍스트에서 약어(abbreviation)를 모두 찾아주세요:
        
        텍스트: {text}
        
        약어만 리스트로 반환하세요.
        """
        
        response = await self.client.chat.completions.create(
            model="solar-1-mini-chat",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return self.parse_abbreviations(response)
    
    async def enhance_summary(self, dialogue, summary):
        """누락된 정보를 찾아 요약 개선"""
        prompt = f"""
        대화문과 요약문을 비교하여 누락된 중요 정보를 찾아주세요:
        
        대화문: {dialogue}
        요약문: {summary}
        
        누락된 정보만 간단히 나열하세요.
        """
        
        response = await self.client.chat.completions.create(
            model="solar-1-mini-chat",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response
```

### 5.4 통합 파이프라인

```python
class DialogueSummarizationPipeline:
    def __init__(self, config):
        self.preprocessor = DialoguePreprocessor()
        self.tokenizer = setup_tokenizer(config['model_name'])
        self.solar_enhancer = SolarEnhancer(config['solar_api_key'])
    
    def process_data(self, df):
        """데이터 전처리 파이프라인"""
        # 1. 기본 전처리
        df['dialogue_clean'] = df['dialogue'].apply(self.preprocessor.preprocess)
        df['summary_clean'] = df['summary'].apply(self.preprocessor.preprocess)
        
        # 2. Topic 정규화
        if 'topic' in df.columns:
            df['topic_normalized'] = df['topic'].map(self.preprocessor.topic_dict)
        
        # 3. 데이터 분석
        df['dialogue_length'] = df['dialogue_clean'].apply(
            lambda x: len(self.tokenizer.encode(x))
        )
        df['summary_length'] = df['summary_clean'].apply(
            lambda x: len(self.tokenizer.encode(x))
        )
        
        return df
    
    async def enhance_data(self, df, sample_size=100):
        """Solar API를 활용한 데이터 개선"""
        # 샘플링하여 처리 (API 제한 고려)
        sample_df = df.sample(min(sample_size, len(df)))
        
        enhanced_summaries = []
        for idx, row in sample_df.iterrows():
            enhanced = await self.solar_enhancer.enhance_summary(
                row['dialogue'], row['summary']
            )
            enhanced_summaries.append(enhanced)
        
        return enhanced_summaries
```

## 결론

DialogSum 데이터셋 분석을 통해 다음과 같은 주요 이슈를 발견했습니다:

1. **번역 품질**: 영어→한글 번역 시 일관성 부족
2. **정보 누락**: 요약문에서 중요 정보 누락
3. **토크나이저 이슈**: 약어 및 특수 표현 처리 문제

이를 해결하기 위한 방안:
1. **즉시**: Special token 추가, 구어체 정제
2. **단기**: Topic dictionary 활용, Solar API 자동화
3. **중기**: 데이터 증강, 요약문 품질 개선

체계적인 전처리와 자동화를 통해 모델 성능을 향상시킬 수 있을 것으로 기대됩니다.
