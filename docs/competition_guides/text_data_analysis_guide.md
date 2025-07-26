# 텍스트 데이터 분석 및 전처리 가이드

## 목차
1. [텍스트 데이터 특성 이해](#1-텍스트-데이터-특성-이해)
2. [대화 데이터 구조 분석](#2-대화-데이터-구조-분석)
3. [텍스트 전처리 기법](#3-텍스트-전처리-기법)
4. [워드클라우드와 시각화](#4-워드클라우드와-시각화)
5. [개인정보 마스킹 처리](#5-개인정보-마스킹-처리)
6. [프로젝트 적용 코드](#6-프로젝트-적용-코드)

---

## 1. 텍스트 데이터 특성 이해

### 1.1 대화 요약 데이터셋 구조
- **dialogue**: 여러 사람의 대화 전문
- **summary**: 대화 요약문
- **fname**: 각 대화의 고유 ID

### 1.2 데이터 특징
- **구어체 대화문** → **문어체 요약문** 변환
- 최소 2명, 최대 7명의 대화 참여자
- 발화자는 `#Person1#`, `#Person2#` 등으로 구분
- 대화 턴은 `\n`으로 구분

## 2. 대화 데이터 구조 분석

### 2.1 데이터 로드 및 기본 탐색
```python
import pandas as pd
import numpy as np

# 데이터 로드
train = pd.read_csv('data/train.csv')
dev = pd.read_csv('data/dev.csv')
test = pd.read_csv('data/test.csv')

print(f"학습 데이터: {len(train)}개")
print(f"검증 데이터: {len(dev)}개")
print(f"테스트 데이터: {len(test)}개")

# 데이터 구조 확인
print("\n=== 학습 데이터 구조 ===")
print(train.head())
```

### 2.2 대화/요약문 길이 분석
```python
# 음절 기반 길이 분석
train['dialogue_length'] = train['dialogue'].apply(len)
train['summary_length'] = train['summary'].apply(len)

print("대화 길이 통계:")
print(train['dialogue_length'].describe())
print("\n요약문 길이 통계:")
print(train['summary_length'].describe())

# 길이 분포 시각화
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.hist(train['dialogue_length'], bins=50, alpha=0.7)
ax1.set_title('대화 길이 분포')
ax1.set_xlabel('길이 (문자 수)')

ax2.hist(train['summary_length'], bins=50, alpha=0.7, color='orange')
ax2.set_title('요약문 길이 분포')
ax2.set_xlabel('길이 (문자 수)')

plt.tight_layout()
plt.show()
```

### 2.3 토큰 기반 길이 분석
```python
from transformers import AutoTokenizer

# Tokenizer 로드
tokenizer = AutoTokenizer.from_pretrained("gogamza/kobart-base-v2")

# 토큰화 후 길이 확인
def get_token_length(text, tokenizer):
    return len(tokenizer.encode(text))

train['dialogue_tokens'] = train['dialogue'].apply(
    lambda x: get_token_length(x, tokenizer)
)
train['summary_tokens'] = train['summary'].apply(
    lambda x: get_token_length(x, tokenizer)
)

print("대화 토큰 수 통계:")
print(train['dialogue_tokens'].describe())
print("\n요약문 토큰 수 통계:")
print(train['summary_tokens'].describe())

# 최적 max_length 설정을 위한 분위수 확인
print(f"\n대화 토큰 95% 분위수: {train['dialogue_tokens'].quantile(0.95)}")
print(f"대화 토큰 99% 분위수: {train['dialogue_tokens'].quantile(0.99)}")
print(f"요약문 토큰 95% 분위수: {train['summary_tokens'].quantile(0.95)}")
print(f"요약문 토큰 99% 분위수: {train['summary_tokens'].quantile(0.99)}")
```

## 3. 텍스트 전처리 기법

### 3.1 구어체 표현 정제
```python
import re

def clean_dialogue(text):
    """구어체 표현을 정제하는 함수"""
    
    # 자음/모음 반복 처리
    replacements = {
        'ㅋㅋ+': '웃음',
        'ㅎㅎ+': '웃음',
        'ㅠㅠ+': '슬픔',
        'ㅜㅜ+': '슬픔',
        'ㅇㅇ': '응',
        'ㄴㄴ': '아니'
    }
    
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    
    # 중복 공백 제거
    text = re.sub(r'\s+', ' ', text)
    
    # 특수문자 정규화
    text = re.sub(r'[…]+', '...', text)
    text = re.sub(r'[~]+', '~', text)
    
    return text.strip()

# 전처리 적용
train['dialogue_clean'] = train['dialogue'].apply(clean_dialogue)
```

### 3.2 문장 분리 및 턴 분석
```python
def analyze_dialogue_turns(dialogue):
    """대화 턴 분석"""
    turns = dialogue.split('\n')
    
    # 발화자별 통계
    speaker_stats = {}
    for turn in turns:
        # 발화자 추출 (#Person1#: ... 형식)
        match = re.match(r'(#Person\d+#):', turn)
        if match:
            speaker = match.group(1)
            if speaker not in speaker_stats:
                speaker_stats[speaker] = 0
            speaker_stats[speaker] += 1
    
    return {
        'total_turns': len(turns),
        'num_speakers': len(speaker_stats),
        'speaker_stats': speaker_stats
    }

# 대화 분석
train['dialogue_analysis'] = train['dialogue'].apply(analyze_dialogue_turns)

# 통계 출력
print("평균 대화 턴 수:", 
      np.mean([d['total_turns'] for d in train['dialogue_analysis']]))
print("평균 발화자 수:", 
      np.mean([d['num_speakers'] for d in train['dialogue_analysis']]))
```

## 4. 워드클라우드와 시각화

### 4.1 명사 추출 및 빈도 분석
```python
from konlpy.tag import Okt
from collections import Counter
from wordcloud import WordCloud

# 형태소 분석기 초기화
okt = Okt()

def extract_nouns(text):
    """명사 추출 함수"""
    nouns = okt.nouns(text)
    # 2글자 이상만 필터링
    nouns = [n for n in nouns if len(n) >= 2]
    return nouns

# 전체 대화에서 명사 추출
all_nouns = []
for dialogue in train['dialogue'].sample(1000):  # 샘플링으로 속도 개선
    all_nouns.extend(extract_nouns(dialogue))

# 빈도 계산
noun_counts = Counter(all_nouns)
top_nouns = noun_counts.most_common(100)

print("상위 20개 명사:")
for noun, count in top_nouns[:20]:
    print(f"{noun}: {count}")
```

### 4.2 워드클라우드 생성
```python
# 한글 폰트 설정 (AIStages 환경)
font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'

# 워드클라우드 생성
wc = WordCloud(
    font_path=font_path,
    background_color='white',
    width=800,
    height=600,
    max_words=100,
    relative_scaling=0.5,
    min_font_size=10
).generate_from_frequencies(dict(top_nouns))

# 시각화
plt.figure(figsize=(12, 8))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title('대화 데이터 워드클라우드', fontsize=20)
plt.show()
```

### 4.3 TF-IDF 기반 중요 단어 추출
```python
from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF 벡터라이저 설정
tfidf_vectorizer = TfidfVectorizer(
    tokenizer=extract_nouns,
    max_features=100,
    min_df=5,  # 최소 5개 문서에 등장
    max_df=0.8  # 80% 이상 문서에 등장하는 단어 제외
)

# TF-IDF 계산
tfidf_matrix = tfidf_vectorizer.fit_transform(train['dialogue'].sample(1000))
feature_names = tfidf_vectorizer.get_feature_names_out()

# 각 단어의 평균 TF-IDF 점수
tfidf_scores = tfidf_matrix.mean(axis=0).A1
tfidf_dict = dict(zip(feature_names, tfidf_scores))

# 상위 단어 출력
top_tfidf = sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True)[:20]
print("\nTF-IDF 상위 20개 단어:")
for word, score in top_tfidf:
    print(f"{word}: {score:.4f}")
```

## 5. 개인정보 마스킹 처리

### 5.1 마스킹 패턴 정의
```python
# 개인정보 마스킹 패턴
PII_MASKS = {
    '#PhoneNumber#': '전화번호',
    '#Address#': '주소',
    '#DateOfBirth#': '생년월일',
    '#PassportNumber#': '여권번호',
    '#SSN#': '사회보장번호',
    '#CardNumber#': '신용카드번호',
    '#CarNumber#': '차량번호',
    '#Email#': '이메일주소'
}
```

### 5.2 마스킹 토큰 추출
```python
def extract_masked_tokens(text):
    """마스킹된 토큰 추출"""
    # 정규표현식: #으로 시작하고 끝나는 패턴
    pattern = r'#\w+#'
    masked_tokens = re.findall(pattern, text)
    return list(set(masked_tokens))

# 발화자 토큰 추출
def extract_speaker_tokens(text):
    """발화자 토큰 추출"""
    # #PersonN# 패턴
    pattern = r'#Person\d+#'
    speakers = re.findall(pattern, text)
    return list(set(speakers))

# 전체 데이터에서 토큰 수집
all_masked_tokens = set()
all_speaker_tokens = set()

for dialogue in train['dialogue']:
    all_masked_tokens.update(extract_masked_tokens(dialogue))
    all_speaker_tokens.update(extract_speaker_tokens(dialogue))

print("마스킹 토큰:", sorted(all_masked_tokens - all_speaker_tokens))
print("\n발화자 토큰:", sorted(all_speaker_tokens))
```

### 5.3 Special Token 추가
```python
# Tokenizer에 special token 추가
special_tokens = list(all_masked_tokens)

# Tokenizer 업데이트
tokenizer = AutoTokenizer.from_pretrained("gogamza/kobart-base-v2")
special_tokens_dict = {'additional_special_tokens': special_tokens}
num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)

print(f"추가된 special token 수: {num_added_tokens}")

# 확인
for token in special_tokens[:5]:
    token_id = tokenizer.convert_tokens_to_ids(token)
    print(f"{token}: {token_id}")
```

## 6. 프로젝트 적용 코드

### 6.1 통합 전처리 파이프라인
```python
class DialoguePreprocessor:
    """대화 데이터 전처리 클래스"""
    
    def __init__(self, tokenizer_name="gogamza/kobart-base-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.okt = Okt()
        self._add_special_tokens()
    
    def _add_special_tokens(self):
        """Special token 추가"""
        special_tokens = [
            '#Person1#', '#Person2#', '#Person3#', '#Person4#', 
            '#Person5#', '#Person6#', '#Person7#',
            '#PhoneNumber#', '#Address#', '#DateOfBirth#',
            '#PassportNumber#', '#SSN#', '#CardNumber#',
            '#CarNumber#', '#Email#'
        ]
        special_tokens_dict = {'additional_special_tokens': special_tokens}
        self.tokenizer.add_special_tokens(special_tokens_dict)
    
    def clean_text(self, text):
        """텍스트 정제"""
        # 구어체 표현 정제
        text = re.sub(r'ㅋ+', '웃음', text)
        text = re.sub(r'ㅎ+', '웃음', text)
        text = re.sub(r'ㅠ+', '슬픔', text)
        text = re.sub(r'ㅜ+', '슬픔', text)
        
        # 중복 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def analyze_dialogue(self, dialogue):
        """대화 분석"""
        turns = dialogue.split('\n')
        speakers = extract_speaker_tokens(dialogue)
        
        return {
            'num_turns': len(turns),
            'num_speakers': len(speakers),
            'speakers': speakers,
            'avg_turn_length': np.mean([len(turn) for turn in turns])
        }
    
    def preprocess(self, df):
        """전체 전처리 파이프라인"""
        # 텍스트 정제
        df['dialogue_clean'] = df['dialogue'].apply(self.clean_text)
        
        # 대화 분석
        df['dialogue_info'] = df['dialogue'].apply(self.analyze_dialogue)
        
        # 토큰 길이 계산
        df['dialogue_tokens'] = df['dialogue_clean'].apply(
            lambda x: len(self.tokenizer.encode(x))
        )
        df['summary_tokens'] = df['summary'].apply(
            lambda x: len(self.tokenizer.encode(x))
        )
        
        return df
```

### 6.2 데이터 로더 통합
```python
# 기존 baseline 코드와 통합
preprocessor = DialoguePreprocessor()

# 데이터 전처리
train_processed = preprocessor.preprocess(train)
dev_processed = preprocessor.preprocess(dev)

# 최적 max_length 설정
ENCODER_MAX_LEN = int(train_processed['dialogue_tokens'].quantile(0.95))
DECODER_MAX_LEN = int(train_processed['summary_tokens'].quantile(0.95))

print(f"Encoder max length: {ENCODER_MAX_LEN}")
print(f"Decoder max length: {DECODER_MAX_LEN}")

# config 업데이트
config['tokenizer']['encoder_max_len'] = ENCODER_MAX_LEN
config['tokenizer']['decoder_max_len'] = DECODER_MAX_LEN
```

### 6.3 시각화 유틸리티
```python
def create_analysis_report(df, output_dir="./analysis"):
    """분석 리포트 생성"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 길이 분포
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0,0].hist(df['dialogue_tokens'], bins=50)
    axes[0,0].set_title('대화 토큰 수 분포')
    
    axes[0,1].hist(df['summary_tokens'], bins=50)
    axes[0,1].set_title('요약문 토큰 수 분포')
    
    # 2. 발화자 수 분포
    num_speakers = [info['num_speakers'] for info in df['dialogue_info']]
    axes[1,0].hist(num_speakers, bins=range(2, 9))
    axes[1,0].set_title('발화자 수 분포')
    
    # 3. 대화 턴 수 분포
    num_turns = [info['num_turns'] for info in df['dialogue_info']]
    axes[1,1].hist(num_turns, bins=30)
    axes[1,1].set_title('대화 턴 수 분포')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/data_distribution.png")
    plt.close()
    
    # 4. 워드클라우드 생성
    all_text = ' '.join(df['dialogue_clean'].sample(min(1000, len(df))))
    nouns = extract_nouns(all_text)
    noun_counts = Counter(nouns)
    
    wc = WordCloud(
        font_path='/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
        background_color='white',
        width=800,
        height=600
    ).generate_from_frequencies(dict(noun_counts.most_common(100)))
    
    plt.figure(figsize=(10, 8))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(f"{output_dir}/wordcloud.png")
    plt.close()
    
    print(f"분석 리포트가 {output_dir}에 저장되었습니다.")

# 리포트 생성
create_analysis_report(train_processed)
```

## 결론

텍스트 데이터 분석과 전처리는 모델 성능 향상의 기초입니다. 주요 포인트:

1. **데이터 이해**: 대화와 요약문의 길이 분포 파악으로 적절한 max_length 설정
2. **전처리**: 구어체 표현 정제로 모델 학습 효율성 향상
3. **토큰화**: Special token 추가로 마스킹된 정보 보존
4. **시각화**: 워드클라우드와 TF-IDF로 데이터 특성 파악

이러한 분석을 통해 데이터의 특성을 이해하고, 모델에 맞는 최적의 전처리 전략을 수립할 수 있습니다.
