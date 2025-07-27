# ROUGE 평가 지표 상세 설명

## ROUGE-N

### 개념
ROUGE-N은 참조 요약본과 모델 요약본 간의 n-gram 겹침을 측정하는 지표입니다.

### 계산 방식
- **Recall** = Gold와 Pred의 겹치는 N-gram의 수 / Gold의 N-gram의 수
- **Precision** = Pred와 Gold의 겹치는 N-gram의 수 / Pred의 N-gram의 수

### ROUGE-1 예시
```
Gold (정답): the cat was under the bed
Pred (모델): under the bed there was the cat

# ROUGE-1 계산
Recall = N(the, cat, was, under, the, bed) / N(the, cat, was, under, the, bed) = 6/6
Precision = N(under, the, bed, was, the, cat) / N(under, the, bed, there, was, the, cat) = 6/7
```

### ROUGE-2 예시
```
Gold (정답): the cat was under the bed
Pred (모델): under the bed there was the cat

# ROUGE-2 계산
Recall = N((the, cat), (under, the), (the, bed)) / N((the, cat), (cat, was), (was, under), (under, the), (the, bed)) = 3/5
Precision = N((under, the), (the, bed), (the, cat)) / N((under, the), (the, bed), (bed, there), (there, was), (was, the), (the, cat)) = 3/6
```

## ROUGE-L

### 개념
ROUGE-L은 가장 긴 공통부분 문자열의 길이(LCS: Longest Common Subsequence)를 기반으로 계산됩니다.

### 계산 방식
- **Recall** = 가장 긴 공통부분 문자열의 길이 / Gold의 1-gram의 수
- **Precision** = 가장 긴 공통부분 문자열의 길이 / Pred의 1-gram의 수

### ROUGE-L 예시
```
Gold (정답): the cat was under the bed
Pred (모델): under the bed there was the cat

# ROUGE-L 계산
LCS = N(under the bed) = 3
Recall = 3/6
Precision = 3/7
```

## ROUGE-F1

### 개념
ROUGE-F1은 ROUGE Recall과 ROUGE Precision의 조화 평균입니다.

### 계산 공식
```
F1 = 2 × (ROUGE_recall × ROUGE_precision) / (ROUGE_recall + ROUGE_precision)
```

## 한국어 문장 토큰화

### 토큰화의 필요성
한국어는 교착어로서, 정확한 ROUGE score 산출을 위해 형태소 단위로 분해가 필요합니다.

### 토큰화 예시
```
원본: 호킨스 의사는 매년 건강검진을 받는 것을 권장합니다.
토큰화: 호킨스 의사 는 매년 건강 검진 을 받 는 것 을 권장 합니다 .
```

### 토큰화 코드 예시
```python
tokenized_encoder_inputs = tokenizer(encoder_input_train, return_tensors="pt", padding=True, 
                                    add_special_tokens=True, truncation=True, 
                                    max_length=config['tokenizer']['encoder_max_len'], 
                                    return_token_type_ids=False)

tokenized_decoder_inputs = tokenizer(decoder_input_train, return_tensors="pt", padding=True,
                                    add_special_tokens=True, truncation=True, 
                                    max_length=config['tokenizer']['decoder_max_len'], 
                                    return_token_type_ids=False)

tokenized_decoder_outputs = tokenizer(decoder_output_train, return_tensors="pt", padding=True,
                                     add_special_tokens=True, truncation=True, 
                                     max_length=config['tokenizer']['decoder_max_len'], 
                                     return_token_type_ids=False)
```

## 대회 평가 방식

### Multi-Reference 평가
- 하나의 대화에 대해 3개의 정답 요약문 사용
- 다양한 관점의 요약 가능성 고려
- 각 정답과의 ROUGE 점수 중 최댓값 선택

### 최종 점수 계산
```
Final Score = max ROUGE-1-F1(pred, gold_i) for i in N
            + max ROUGE-2-F1(pred, gold_i) for i in N  
            + max ROUGE-L-F1(pred, gold_i) for i in N
```

여기서 N은 정답 요약문의 개수(3개)를 의미합니다.

### 평가 예시
하나의 대화에 대한 3개의 정답 요약문과 모델 예측을 비교하여, 각 ROUGE 지표별로 가장 높은 점수를 선택한 후 합산합니다.

## 참고사항
- DialogSum 데이터셋의 Multi-Reference 특성에 맞춘 평가 방식
- 3개 정답 중 하나를 랜덤 선택 시 약 70점 수준
- 완벽한 100점은 현실적으로 달성 불가능
