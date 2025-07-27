# Solar Chat API를 활용한 대화 요약 상세 분석

## 목차
1. [개요](#1-개요)
2. [코드 구조](#2-코드-구조)
3. [환경 설정](#3-환경-설정)
4. [기본 요약 구현](#4-기본-요약-구현)
5. [성능 평가](#5-성능-평가)
6. [프롬프트 엔지니어링](#6-프롬프트-엔지니어링)
7. [Rate Limit 처리](#7-rate-limit-처리)
8. [주요 개념 설명](#8-주요-개념-설명)

---

## 1. 개요

이 코드는 **Solar Chat API**를 사용하여 대화 요약을 수행하는 방법을 보여줍니다. 주요 특징은:

- **모델**: solar-1-mini-chat (Upstage AI의 경량 언어 모델)
- **접근 방식**: API 기반 (학습 불필요)
- **장점**: 빠른 구현, 유지보수 용이, GPU 불필요
- **단점**: API 비용, Rate Limit, 인터넷 연결 필요

## 2. 코드 구조

### 전체 흐름
1. **환경 설정**: API 키 설정 및 클라이언트 생성
2. **프롬프트 설계**: 대화를 요약 지시문으로 변환
3. **API 호출**: Solar API로 요약 생성
4. **성능 평가**: ROUGE 점수 측정
5. **프롬프트 최적화**: Few-shot 학습 적용

## 3. 환경 설정

### 3.1 필요 라이브러리 설치 (코드 위치: 셀 3-4)

```python
!pip install openai  # OpenAI 라이브러리 설치 (Solar API 호환)

import pandas as pd
import os
import time
from tqdm import tqdm
from rouge import Rouge
from openai import OpenAI  # openai==1.2.0
```

### 3.2 API 클라이언트 생성 (코드 위치: 셀 5)

```python
UPSTAGE_API_KEY = "up_*****************************"  # API 키 입력

client = OpenAI(
    api_key=UPSTAGE_API_KEY,
    base_url="https://api.upstage.ai/v1/solar"  # Solar API 엔드포인트
)
```

**중요 설명**:
- **API 키**: upstage.ai에서 발급받은 개인 키
- **base_url**: Solar API의 기본 URL (OpenAI 형식 호환)

### 3.3 API 테스트 (코드 위치: 셀 6)

```python
stream = client.chat.completions.create(
    model="solar-1-mini-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ],
    stream=True,  # 스트리밍 모드
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

**스트리밍 모드 설명**:
- `stream=True`: 응답을 청크 단위로 실시간 수신
- `stream=False`: 전체 응답을 한 번에 수신

## 4. 기본 요약 구현

### 4.1 프롬프트 생성 함수 (코드 위치: 셀 9)

```python
def build_prompt(dialogue):
    system_prompt = "You are an expert in the field of dialogue summarization. Please summarize the following dialogue."
    
    user_prompt = f"Dialogue:\n{dialogue}\n\nSummary:\n"
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
```

**프롬프트 구조**:
- **system**: AI의 역할 정의 (대화 요약 전문가)
- **user**: 실제 대화 내용과 요약 요청

### 4.2 요약 함수 (코드 위치: 셀 10)

```python
def summarization(dialogue):
    summary = client.chat.completions.create(
        model="solar-1-mini-chat",
        messages=build_prompt(dialogue),
    )
    
    return summary.choices[0].message.content
```

### 4.3 API 파라미터 조정 (코드 위치: 셀 11)

```python
def summarization(dialogue):
    summary = client.chat.completions.create(
        model="solar-1-mini-chat",
        messages=build_prompt(dialogue),
        temperature=0.2,  # 생성 다양성 (0=결정적, 1=창의적)
        top_p=0.3,        # 누적 확률 임계값
    )
    
    return summary.choices[0].message.content
```

**파라미터 설명**:
- **temperature**: 낮을수록 일관된 출력, 높을수록 다양한 출력
- **top_p**: 상위 확률 토큰만 샘플링 (nucleus sampling)

## 5. 성능 평가

### 5.1 평가 지표 정의 (코드 위치: 셀 8)

```python
rouge = Rouge()

def compute_metrics(pred, gold):
    results = rouge.get_scores(pred, gold, avg=True)
    result = {key: value["f"] for key, value in results.items()}
    return result
```

### 5.2 학습 데이터 테스트 (코드 위치: 셀 12-13)

```python
def test_on_train_data(num_samples=3):
    for idx, row in train_df[:num_samples].iterrows():
        dialogue = row['dialogue']
        summary = summarization(dialogue)
        
        print(f"Dialogue:\n{dialogue}\n")
        print(f"Pred Summary: {summary}\n")
        print(f"Gold Summary: {row['summary']}\n")
        print("=="*50)
```

### 5.3 검증 데이터 평가 (코드 위치: 셀 14-15)

```python
def validate(num_samples=-1):
    val_samples = val_df[:num_samples] if num_samples > 0 else val_df
    
    scores = []
    for idx, row in tqdm(val_samples.iterrows(), total=len(val_samples)):
        dialogue = row['dialogue']
        summary = summarization(dialogue)
        results = compute_metrics(summary, row['summary'])
        avg_score = sum(results.values()) / len(results)
        
        scores.append(avg_score)
    
    val_avg_score = sum(scores) / len(scores)
    print(f"Validation Average Score: {val_avg_score}")
```

## 6. 프롬프트 엔지니어링

### 6.1 Few-shot 샘플 준비 (코드 위치: 셀 20)

```python
# 학습 데이터에서 예시 추출
few_shot_samples = train_df.sample(1)

sample_dialogue1 = few_shot_samples.iloc[0]['dialogue']
sample_summary1 = few_shot_samples.iloc[0]['summary']
```

### 6.2 개선된 프롬프트 v1 (코드 위치: 셀 21)

```python
def build_prompt(dialogue):
    system_prompt = "You are a expert in the field of dialogue summarization..."
    
    user_prompt = (
        "Following the instructions below, summarize the given document.\n"
        "Instructions:\n"
        "1. Read the provided sample dialogue and corresponding summary.\n"
        "2. Read the dialogue carefully.\n"
        "3. Following the sample's style of summary, provide a concise summary...\n\n"
        "Sample Dialogue:\n"
        f"{sample_dialogue1}\n\n"
        "Sample Summary:\n"
        f"{sample_summary1}\n\n"
        "Dialogue:\n"
        f"{dialogue}\n\n"
        "Summary:\n"
    )
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
```

**Few-shot 학습의 장점**:
- 원하는 요약 스타일 학습
- 일관된 형식 유지
- 성능 향상

### 6.3 개선된 프롬프트 v2 (코드 위치: 셀 24)

```python
def build_prompt(dialogue):
    system_prompt = "You are a expert in the field of dialogue summarization..."
    
    # Few-shot 예시를 대화 형식으로 제공
    few_shot_user_prompt_1 = (
        "Following the instructions below, summarize the given document.\n"
        "Instructions:\n"
        "1. Read the provided sample dialogue and corresponding summary.\n"
        "2. Read the dialogue carefully.\n"
        "3. Following the sample's style of summary, provide a concise summary...\n\n"
        "Dialogue:\n"
        f"{sample_dialogue1}\n\n"
        "Summary:\n"
    )
    
    few_shot_assistant_prompt_1 = sample_summary1  # 예시 응답
    
    user_prompt = (
        "Dialogue:\n"
        f"{dialogue}\n\n"
        "Summary:\n"
    )
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": few_shot_user_prompt_1},
        {"role": "assistant", "content": few_shot_assistant_prompt_1},
        {"role": "user", "content": user_prompt},
    ]
```

**대화 형식 Few-shot의 장점**:
- 더 명확한 맥락 제공
- 모델이 대화 패턴 학습
- 일반적으로 더 나은 성능

## 7. Rate Limit 처리

### 7.1 추론 함수 (코드 위치: 셀 17)

```python
def inference():
    test_df = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
    
    summary = []
    start_time = time.time()
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        dialogue = row['dialogue']
        summary.append(summarization(dialogue))
        
        # Rate limit 방지: 분당 100개 요청 제한
        if (idx + 1) % 100 == 0:
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            if elapsed_time < 60:  # 60초 미만이면
                wait_time = 60 - elapsed_time + 5  # 5초 여유
                print(f"Elapsed time: {elapsed_time:.2f} sec")
                print(f"Waiting for {wait_time} sec")
                time.sleep(wait_time)
            
            start_time = time.time()
    
    # 결과 저장
    output = pd.DataFrame({
        "fname": test_df['fname'],
        "summary": summary,
    })
    
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)
    output.to_csv(os.path.join(RESULT_PATH, "output_solar.csv"), index=False)
```

**Rate Limit 처리 전략**:
1. 요청 카운터 유지
2. 100개 요청마다 시간 체크
3. 필요시 대기 시간 추가
4. 안전 마진(5초) 포함

## 8. 주요 개념 설명

### 8.1 Large Language Model (LLM)

**Solar-1-mini-chat**:
- Upstage AI가 개발한 경량 언어 모델
- 한국어와 영어 모두 지원
- Chat 형식에 최적화
- 빠른 응답 속도와 효율적인 토큰 사용

### 8.2 API vs Fine-tuning

| 구분 | API 방식 | Fine-tuning 방식 |
|------|----------|------------------|
| 구현 속도 | 빠름 (즉시 사용) | 느림 (학습 필요) |
| 비용 | 사용량 기반 | 초기 학습 비용 |
| 성능 | 범용적 | 태스크 특화 |
| 유지보수 | 간단 | 복잡 |
| 커스터마이징 | 제한적 | 자유로움 |

### 8.3 프롬프트 엔지니어링 기법

#### Zero-shot
```python
"Summarize this dialogue: {dialogue}"
```

#### Few-shot
```python
"Example: {example_dialogue} → {example_summary}
Now summarize: {dialogue}"
```

#### Chain-of-Thought (CoT)
```python
"1. Identify speakers
2. Extract main topics
3. Note key decisions
4. Summarize in 2-3 sentences"
```

### 8.4 토큰과 비용

**토큰 계산**:
- 한국어: 평균 1글자 ≈ 0.7토큰
- 영어: 평균 1단어 ≈ 1.3토큰

**비용 최적화**:
1. 불필요한 프롬프트 제거
2. 응답 길이 제한
3. 캐싱 활용
4. 배치 처리

### 8.5 Temperature와 Top-p

#### Temperature
```
T = 0.0: "오늘은 날씨가 좋습니다."
T = 0.5: "오늘은 날씨가 맑습니다."
T = 1.0: "오늘은 하늘이 청명하네요."
```

#### Top-p (Nucleus Sampling)
```
확률 분포: [좋다:0.4, 맑다:0.3, 흐리다:0.2, 춥다:0.1]
top_p=0.7 → [좋다, 맑다]만 선택 가능
top_p=0.9 → [좋다, 맑다, 흐리다] 선택 가능
```

### 8.6 API 응답 구조

```python
{
    "id": "chatcmpl-...",
    "object": "chat.completion",
    "created": 1234567890,
    "model": "solar-1-mini-chat",
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": "요약된 내용..."
        },
        "finish_reason": "stop"
    }],
    "usage": {
        "prompt_tokens": 150,
        "completion_tokens": 50,
        "total_tokens": 200
    }
}
```

## 성능 향상 팁

### 1. 프롬프트 최적화
- 명확한 지시문 작성
- 적절한 Few-shot 예시 선택
- 출력 형식 명시

### 2. 파라미터 튜닝
```python
# 일관된 요약
temperature=0.1, top_p=0.3

# 창의적 요약
temperature=0.7, top_p=0.9
```

### 3. 후처리
```python
# 불필요한 문구 제거
summary = summary.replace("요약:", "").strip()

# 길이 제한
if len(summary) > 100:
    summary = summary[:100] + "..."
```

### 4. 에러 처리
```python
try:
    summary = summarization(dialogue)
except Exception as e:
    print(f"Error: {e}")
    summary = "요약 생성 실패"
```

## 마무리

Solar Chat API를 활용한 대화 요약은:
- **빠른 프로토타이핑**에 적합
- **프롬프트 엔지니어링**이 핵심
- **Rate Limit** 고려 필요
- **비용 대비 효과** 분석 중요

베이스라인 대비 장단점:
- ✅ 구현 간단, GPU 불필요
- ✅ 즉시 사용 가능
- ❌ API 비용 발생
- ❌ 인터넷 연결 필수
- ❌ 커스터마이징 제한
