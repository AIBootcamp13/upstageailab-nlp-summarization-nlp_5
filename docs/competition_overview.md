# Dialogue Summarization | 일상 대화 요약 대회

## 대회 개요

### 소개
**Dialogue Summarization** 경진대회는 주어진 데이터를 활용하여 일상 대화에 대한 요약을 효과적으로 생성하는 모델을 개발하는 대회입니다.

일상생활에서 대화는 **항상** 이루어지고 있습니다. 회의나 토의는 물론이고, 사소한 일상 대화 중에도 서로 다양한 주제와 입장들을 주고 받습니다. 나누는 대화를 녹음해두더라도 대화 전체를 항상 다시 들을 수는 없기 때문에 요약이 필요하고, 이를 위한 통화 비서와 같은 서비스들도 등장하고 있습니다.

그러나 하나의 대화에서도 관점, 주제별로 정리하면 수 많은 요약을 만들 수 있습니다. 대화를 하는 도중에 이를 요약하게 되면 대화에 집중할 수 없으며, 대화 이후에 기억에 의존해 요약하게 되면 오해나 누락이 추가되어 주관이 많이 개입되게 됩니다.

이를 돕기 위해, 우리는 이번 대회에서 **일상 대화를 바탕으로 요약문을 생성하는 모델**을 구축합니다!

### 대회 목표
- 정확하고 일반화된 모델을 개발하여 요약문을 생성
- 대화의 핵심적인 부분을 모델이 자동으로 요약
- 업무 효율성 향상 및 관계 개선에 기여
- 자연어 딥러닝 모델링 분야의 실전 경험 획득

### 대회 정보
- **대회 기간**: 2025.07.25 10:00 ~ 2025.08.06 19:00
- **참가 대상**: AI 부트캠프 13기 수강생
- **태그**: #비공개대회 #AI부트캠프13기 #NLPAdvanced

## 데이터

### 데이터 구성
- **입력**: 249개의 대화문
- **출력**: 249개의 대화 요약문
- **형식**: CSV 파일

### 데이터셋 건수
- **train**: 12,457개
- **dev**: 499개
- **test**: 250개
- **hidden-test**: 249개

### 데이터 구조
| 컬럼명 | 설명 |
|--------|------|
| fname | 대화 고유번호 (중복 없음) |
| dialogue | 최소 2명에서 최대 7명이 등장하는 대화 내용<br>- 발화자 구분: `#Person"N"#:`<br>- 발화 종료: `\n` |
| summary | 해당 대화를 바탕으로 작성된 요약문 |

### 데이터 특징
- 최소 2턴, 최대 60턴으로 구성된 대화
- 다양한 주제: 학교 생활, 직장, 치료, 쇼핑, 여가, 여행 등
- 한국어로 번역된 DialogSum 데이터셋 활용

### 데이터 노이즈
데이터에는 다양한 형태의 노이즈가 포함될 수 있습니다:
- 맞춤법 오류
- 문장 부호의 누락 또는 과다 사용
- 발화 구분 기호의 불일치
- 화자 표기의 불명확함
- newline character 변형 (예: `"\n"` → `"\\n"`)
- HTML 태그 포함 (예: `<br>`)

### 데이터 다운로드
```bash
wget https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000357/data/20250422073240/data.tar.gz
```

## 평가 방법

### 평가 지표
대회에서는 3가지 ROUGE 지표의 평균 점수를 합산하여 최종 점수를 계산합니다:
- **ROUGE-1-F1**
- **ROUGE-2-F1**
- **ROUGE-L-F1**

### ROUGE 지표 설명

#### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
텍스트 요약, 기계 번역 평가를 위한 대표적인 metric으로, 모델 생성 요약본과 참조 요약본을 비교합니다.

#### ROUGE 구성요소
- **ROUGE-Recall**: 참조 요약본 단어 중 모델 요약본과 겹치는 비율
- **ROUGE-Precision**: 모델 요약본 단어 중 참조 요약본과 겹치는 비율
- **ROUGE-F1**: Recall과 Precision의 조화 평균

#### ROUGE 유형
1. **ROUGE-N**: n-gram 기반 비교
   - ROUGE-1: unigram 비교
   - ROUGE-2: bigram 비교
   
2. **ROUGE-L**: LCS(Longest Common Subsequence) 기반
   - 최장 길이 매칭 문자열 측정
   - 단어 등장 순서 고려
   - 더 유연한 성능 비교 가능

### 평가 방식 특징
- **Multi-Reference 평가**: 하나의 대화에 대해 3개의 정답 요약문과 비교
- 다양한 관점의 요약 가능성 고려
- 3개 정답의 평균 점수 활용
- 문장 토큰화 후 평가 진행

### 문장 토큰화
한국어 데이터 특성상 정확한 ROUGE score 산출을 위해:
- 한국어 형태소 분석기 사용
- 형태소 단위로 문장 분해
- 토큰화된 문장으로 비교 평가

#### 토큰화 예시
```
[Original text]
호킨스 의사는 매년 건강검진을 받는 것을 권장합니다.

[Tokenized text]
호킨스 의사 는 매년 건강 검진 을 받 는 것 을 권장 합니다 .
```

### 최종 점수 계산
```
Final Score = max ROUGE-1-F1(pred, gold_i)
            + max ROUGE-2-F1(pred, gold_i)
            + max ROUGE-L-F1(pred, gold_i)
```

## 제출 규칙

### 리더보드 제출
- **일일 제출 횟수**: 팀당 12회 제한
- **초기화 시간**: 한국시간 자정
- **최종 제출물**: 팀별 최대 2개 결과 선택 가능

### 제한사항

#### 외부 데이터셋 규정
- 외부 데이터셋 기본적으로 **허용**
- **DialogSum 데이터셋** 직/간접 사용 **전면 금지**
  - 직접 사용 금지
  - 파생 데이터셋 사용 금지
  - DialogSum으로 학습된 모델 사용 금지

#### 평가 데이터 활용
- 평가 데이터 분석: 가능
- 평가 데이터 학습 활용: 금지

#### 사전학습 가중치 사용
- DialogSum 기반 가중치: 사용 금지
- 기타 사전학습 가중치: 사용 가능

#### API 사용 규정
- 무료 API만 사용 가능
- Solar 모델 사용 가능

### 최종 검증
- 상위권 참가자 코드 검수 가능
- 결과 재현 요구 가능
- 재현 불가 시 순위 제외

## 베이스라인 코드

### 코드 구조
```
code (폴더)
├── baseline.ipynb
├── config.yaml
├── requirements.txt
└── solar_api.ipynb
```

### baseline.ipynb 주요 구성

#### 1. 데이터 전처리
- `class Preprocess`: 데이터 전처리 클래스
- `DatasetForTrain`, `DatasetForVal`, `DatasetForInference`: 데이터셋 클래스
- `prepare_train_dataset`: 토큰화 및 모델 입력 데이터 생성

#### 2. 모델 학습 설정
- `compute_metrics`: ROUGE 평가 지표 정의
- `load_trainer_for_train`: Trainer 클래스 설정
- `load_tokenizer_and_model_for_train`: KoBART 모델 및 토크나이저 로드

#### 3. 모델 추론
- `prepare_test_dataset`: 테스트 데이터 준비
- `load_tokenizer_and_model_for_test`: 학습된 모델 로드
- `inference`: 요약문 생성

### solar_api.ipynb 주요 구성

#### 1. Solar Chat API 활용
- `build_prompt`: 프롬프트 생성
- `summarization`: API를 통한 요약 수행
- `test_on_train_data`: 학습 데이터 테스트
- `validation`: 검증 데이터 평가

#### 2. 추론 및 제출
- `inference`: 테스트 데이터 요약 및 제출 파일 생성
- RPM 제한 고려 (분당 최대 100개 요청)

### 성능 정보
- **Training time**: ~20분 (Stages GPU 서버)
- **Inference time**: ~12.6초
- **Public data 성능**: ROUGE-F1 score 47.1244

## 라이센스
- **DialogSum Dataset**: CC BY-NC-SA 4.0 license
- **베이스라인 코드**: 2차 저작물 외부 공개 가능 (코드 1줄 이상 수정 시)

## 세부 일정
- **프로젝트 기간**: 2025.07.25 (금) 10:00 ~ 2025.08.06 (수) 19:00
- **팀 병합 가능**: 2025.07.28 (월) 16:00까지
- **팀명 컨벤션**: NLP (팀번호)조 (예: NLP 1조)
- **GPU 서버 운영**: 2025.07.25 (금) 10:00 ~ 2025.09.05 (금) 16:00

## 참고사항
- 공개 토론 게시판을 통한 아이디어 및 코드 공유 권장
- 대회 종료 3일 전 전체 코드 공유 지양
- 결과물은 CSV 형식으로 제출
