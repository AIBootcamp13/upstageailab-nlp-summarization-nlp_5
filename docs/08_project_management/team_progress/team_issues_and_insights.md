# 팀 진행 상황 및 이슈 정리

## 목차
1. [환경 설정 이슈](#1-환경-설정-이슈)
2. [데이터셋 분석 인사이트](#2-데이터셋-분석-인사이트)
3. [기술적 발견사항](#3-기술적-발견사항)
4. [팀원별 작업 현황](#4-팀원별-작업-현황)
5. [향후 작업 계획](#5-향후-작업-계획)

---

## 1. 환경 설정 이슈

### 1.1 ModuleNotFoundError 해결
**문제**: `ModuleNotFoundError: No module named 'src'`

**원인**: 
- project_dir 경로 설정 오류
- Python 모듈 경로 문제

**해결 방법**:
1. **project_dir 올바르게 설정**
   ```python
   # generate_config.ipynb
   project_dir = "/data/ephemeral/home/upstageailab-nlp-summarization-nlp_5/"
   
   # main_base.py line 22
   project_dir = "/data/ephemeral/home/upstageailab-nlp-summarization-nlp_5/"
   ```

2. **sys.path 추가**
   ```python
   import sys
   sys.path.append(project_dir)
   ```

3. **임시 해결책**: main_base_all.py 생성
   - import 문 제거하고 모든 코드를 하나의 파일에 통합

### 1.2 WandB 권한 오류
**문제**: `wandb: ERROR Error while calling W&B API: permission denied (<Response [403]>)`

**해결 방법**:
1. **entity 설정 확인**
   ```yaml
   wandb:
     entity: "your_username"  # 본인 계정으로 변경
   ```

2. **재로그인**
   ```python
   wandb login --relogin
   ```

### 1.3 Python 버전 및 의존성 관리
**환경 재구성 필요**:
- 표준 환경: Python 3.11
- Unsloth 라이브러리: torch 2.4+ 요구
- 기존 requirements.txt: torch 2.1

**변경사항**:
- `evaluation_strategy` → `eval_strategy`
- 새로운 requirements.txt 작성
- UV를 사용한 재설치

## 2. 데이터셋 분석 인사이트

### 2.1 DialogSum 원본과의 비교
**데이터셋 크기 차이**:
| 구분 | 원본 (영어) | 대회 (한글) |
|------|------------|------------|
| Train | 12,460 | 12,457 |
| Validation | 500 | 499 |
| Test | 1,500 | 250 |

**제외된 데이터**:
- train_10933, train_10972, train_11473
- 공통점: topic이 "after-sales service"
- dev_475: topic이 "buy furniture"

### 2.2 데이터 특성 분석

#### 2.2.1 요약문 품질 이슈
```python
# train_5000 예시
dialogue: 동물 선호/혐오에 대한 대화
summary에서 누락된 정보:
- Person1: 곰과 판다 좋아함, 쥐가 무서움
- Person2: 거미 싫어함
```

**문제점**:
1. 중요 정보 누락
2. 감정 표현 혼용 (무섭다/싫다, 좋다/멋있다)
3. topic 표현의 불완전성

#### 2.2.2 Topic 분석
- **원칙**: 3단어 이하로 표현
- **특징**: 
  - 영어 약어 다수 포함 (EDD, ETV, BBC 등)
  - 인물명 포함 (알버트 아인슈타인, 에이브러햄 링컨 등)
  - 한글 번역 시 일관성 부족

### 2.3 논문 기반 인사이트

#### 2.3.1 대화 요약의 핵심 과제
1. **Discourse Structure (담화 구조)**
   - 멀리 떨어진 발화 간 관계 파악
   - 인과관계 등 담화 관계 이해

2. **Coreference & Ellipsis (상호참조 및 생략)**
   - 화자 구분 및 행동 연결
   - Transformer: 60% 오류율

3. **Intent Identification (화자 의도 파악)**
   - 대화 결과뿐만 아니라 동기 파악
   - Transformer: 84.6% 실패율

4. **Pragmatics & Common Sense**
   - "Here you are" → "결제하다"
   - 맥락적 이해 필요

#### 2.3.2 주요 오류 유형
1. 잘못된 상호참조 (94%/60%)
2. 중요 정보 누락 (64%/32%)
3. 중복 정보 (62%/44%)
4. 사실 오류 (74%/22%)
5. 문법 오류 (72%/22%)

## 3. 기술적 발견사항

### 3.1 Tokenizer 이슈
```python
# digit82/kobart-summarization의 문제
tokenizer.tokenize("ATM") → ['_A', 'T', 'M']  # 부적절한 분리
```

**해결 방안**:
1. Special token으로 약어 추가
2. 인물명 special token 처리
3. mT5 등 다른 tokenizer 고려

### 3.2 Solar API 활용 방안
- **목적**: 자동화된 데이터 전처리
- **활용 분야**:
  - 약어 감지 및 처리
  - 인물명 추출
  - 동의어 정규화
  - 누락된 정보 보완

### 3.3 평가 지표 고려사항
- **ROUGE의 한계**: 토큰 일치 기반
- **동의어 처리**: 점수에 불리
- **Test 데이터**: summary 3개로 다양성 보완

## 4. 팀원별 작업 현황

### 4.1 송규헌
- **작업 TODO**: 번역 시 발생할 수 있는 노이즈 처리
- **접근 방법**: 
  1. 의식의 흐름대로 데이터 분석
  2. 특이점 발견 및 정리
  3. 코드 구현 고민
- **참고**: 송원호 강사님 Kaggle 작업 스타일

### 4.2 이상현
- **작업**: 데이터 탐색 및 분석
- **기여**: 
  - Google Sheets에 데이터 정리
  - 논문 분석 (Moonlight AI 활용)
  - 데이터셋 문제점 발견

### 4.3 이영준
- **작업**: 자동화 스크립트
- **목표**: 주무실 때도 계속 실행 가능한 자동화

## 5. 향후 작업 계획

### 5.1 즉시 적용 가능한 개선사항

#### 5.1.1 데이터 전처리
```python
class DialoguePreprocessor:
    def __init__(self):
        self.abbreviations = ['EDD', 'ETV', 'BBC', 'ATM', 'CEO', 'CPR']
        self.person_names = ['알버트 아인슈타인', '에이브러햄 링컨']
        
    def process_abbreviations(self, text):
        """약어를 special token으로 처리"""
        for abbr in self.abbreviations:
            text = text.replace(abbr, f"#ABBR_{abbr}#")
        return text
    
    def process_person_names(self, text):
        """인물명을 special token으로 처리"""
        for name in self.person_names:
            text = text.replace(name, f"#NAME_{name.replace(' ', '_')}#")
        return text
```

#### 5.1.2 Topic Dictionary 활용
```python
# topic_dict_cleaned.yaml 활용
topic_dict = load_yaml('data/topic_dict_cleaned.yaml')

def normalize_topic(topic_en):
    """영어 topic을 한글로 정규화"""
    return topic_dict.get(topic_en, topic_en)
```

### 5.2 중장기 개선 방안

#### 5.2.1 데이터 증강
1. **Clustering 기반 증강** (이전 기수 방법)
2. **Manual Summary 추가**
3. **Paraphrasing 활용**

#### 5.2.2 모델 개선
1. **Discourse-aware Model**
   - 담화 구조 인식 강화
   - Coreference resolution 개선

2. **Multi-task Learning**
   - Topic classification
   - Intent identification
   - Summary generation

#### 5.2.3 Solar API 자동화
```python
async def enhance_data_with_solar(dialogue, summary):
    """Solar API를 활용한 데이터 개선"""
    # 1. 약어 감지
    abbreviations = await detect_abbreviations(dialogue)
    
    # 2. 누락 정보 확인
    missing_info = await check_missing_info(dialogue, summary)
    
    # 3. 동의어 정규화
    normalized = await normalize_synonyms(summary)
    
    return enhanced_data
```

### 5.3 실험 관리 체계

#### 5.3.1 Git Workflow
```bash
# 1. 각자 브랜치 작업
git checkout -b feature/data-preprocessing-{name}

# 2. TODO 선택 및 작업
# 3. 노트북 파일 생성 (EDA_taskname.ipynb)
# 4. 결과 정리 및 커밋
```

#### 5.3.2 결과 공유
- **Notion TODO**: 분석 결과 정리
- **Jupyter Notebook**: 코드 및 시각화
- **함수화**: 재사용 가능한 코드

## 결론

현재 팀은 다음과 같은 도전과제를 가지고 있습니다:

1. **기술적 이슈**: 환경 설정 및 의존성 관리
2. **데이터 품질**: 번역 노이즈, 정보 누락, 일관성 부족
3. **평가 지표**: ROUGE의 한계와 동의어 처리

이를 해결하기 위한 접근 방법:
1. **즉시**: 환경 통일, 기본 전처리, Special Token 추가
2. **단기**: Solar API 활용, 데이터 증강, 자동화
3. **중기**: 담화 구조 인식, Multi-task Learning

각 팀원이 TODO를 통해 체계적으로 작업하고, 결과를 공유하며 점진적으로 개선해나가는 것이 중요합니다.
