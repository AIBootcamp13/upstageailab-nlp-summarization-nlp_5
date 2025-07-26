# 팀 이슈 및 DialogSum 분석 통합 가이드

## 목차
1. [현재 프로젝트 상태](#1-현재-프로젝트-상태)
2. [새로운 발견사항 요약](#2-새로운-발견사항-요약)
3. [즉시 적용 가능한 개선사항](#3-즉시-적용-가능한-개선사항)
4. [단계별 통합 전략](#4-단계별-통합-전략)
5. [팀 협업 방안](#5-팀-협업-방안)

---

## 1. 현재 프로젝트 상태

### 1.1 완료된 작업
✅ **기본 인프라**
- UV 패키지 관리자 설정
- 베이스라인 코드 구현
- WandB 설정 가이드
- 기본 데이터 분석

✅ **문서화**
- 환경 설정 가이드
- 하이퍼파라미터 튜닝 가이드
- 텍스트 데이터 분석 가이드

### 1.2 진행 중인 이슈
⚠️ **기술적 이슈**
- ModuleNotFoundError (src 모듈)
- WandB 권한 오류
- Torch 버전 충돌 (2.1 vs 2.4)

⚠️ **데이터 품질 이슈**
- 요약문 정보 누락
- 약어 토큰화 문제
- Topic 번역 일관성

## 2. 새로운 발견사항 요약

### 2.1 DialogSum 데이터셋 특성
1. **원본과의 차이**
   - Train: 3개 제외 (after-sales service)
   - Validation: 1개 제외 (buy furniture)
   - Test: 구조 완전히 다름 (1,500→250)

2. **주요 문제점**
   - 약어 다수 포함 (EDD, ETV, BBC 등)
   - 인물명 토큰화 이슈
   - 감정 표현 불일치 (무섭다/싫다)

3. **논문 인사이트**
   - Coreference 오류: 60-94%
   - Intent 파악 실패: 30-84%
   - 담화 구조 이해 필수

### 2.2 팀 작업 현황
- **송규헌**: 번역 노이즈 처리 (00_EDA_origin-DialogSum.ipynb)
- **이상현**: 데이터 분석 및 Google Sheets 정리
- **이영준**: 자동화 스크립트 개발

## 3. 즉시 적용 가능한 개선사항

### 3.1 환경 설정 수정

#### Step 1: project_dir 통일
```python
# 모든 파일에서 동일하게 설정
project_dir = "/data/ephemeral/home/upstageailab-nlp-summarization-nlp_5/"

# sys.path 추가
import sys
sys.path.append(project_dir)
```

#### Step 2: requirements.txt 업데이트
```bash
# UV로 재설치
uv pip uninstall torch torchvision -y
uv pip install torch==2.4.0 torchvision --index-url https://download.pytorch.org/whl/cu118
uv pip install -r requirements_updated.txt --system
```

### 3.2 데이터 전처리 통합

#### Step 1: Special Token 추가
```python
# src/dataset/preprocess.py에 추가
SPECIAL_TOKENS = {
    'person_tokens': [f'#Person{i}#' for i in range(1, 8)],
    'pii_tokens': [
        '#PhoneNumber#', '#Address#', '#DateOfBirth#',
        '#PassportNumber#', '#SSN#', '#CardNumber#',
        '#CarNumber#', '#Email#'
    ],
    'abbreviations': [
        'ATM', 'AS', 'BBC', 'CEO', 'CPR', 'EDD', 
        'ETV', 'GPS', 'USB', 'LCD'
    ]
}

def add_special_tokens_to_tokenizer(tokenizer):
    """토크나이저에 special token 추가"""
    all_tokens = []
    all_tokens.extend(SPECIAL_TOKENS['person_tokens'])
    all_tokens.extend(SPECIAL_TOKENS['pii_tokens'])
    all_tokens.extend([f'#{abbr}#' for abbr in SPECIAL_TOKENS['abbreviations']])
    
    special_tokens_dict = {'additional_special_tokens': all_tokens}
    num_added = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"추가된 special token 수: {num_added}")
    
    return tokenizer
```

#### Step 2: 전처리 함수 통합
```python
# src/dataset/preprocess.py
import re
import yaml

class DialoguePreprocessor:
    def __init__(self, topic_dict_path='data/topic_dict_cleaned.yaml'):
        with open(topic_dict_path, 'r', encoding='utf-8') as f:
            self.topic_dict = yaml.safe_load(f)
    
    def preprocess_dialogue(self, text):
        """대화문 전처리"""
        # 1. A/S -> AS 통일
        text = text.replace('A/S', 'AS')
        
        # 2. 약어를 special token으로
        for abbr in SPECIAL_TOKENS['abbreviations']:
            text = re.sub(f'\\b{abbr}\\b', f'#{abbr}#', text)
        
        # 3. 구어체 정제
        text = re.sub(r'ㅋ+', '웃음', text)
        text = re.sub(r'ㅎ+', '웃음', text)
        text = re.sub(r'ㅠ+|ㅜ+', '슬픔', text)
        
        # 4. 중복 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def normalize_topic(self, topic):
        """Topic 정규화"""
        return self.topic_dict.get(topic, topic)
```

### 3.3 Solar API 활용 설정

```python
# src/utils/solar_enhancer.py
import os
from openai import AsyncOpenAI

class SolarEnhancer:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=os.environ.get("UPSTAGE_API_KEY"),
            base_url="https://api.upstage.ai/v1/solar"
        )
    
    async def detect_missing_info(self, dialogue, summary):
        """요약문에서 누락된 정보 감지"""
        prompt = f"""
        대화문과 요약문을 비교하여 누락된 중요 정보를 찾아주세요.
        
        대화문:
        {dialogue}
        
        요약문:
        {summary}
        
        누락된 정보를 JSON 형식으로 반환하세요:
        {{"missing_info": ["정보1", "정보2", ...]}}
        """
        
        response = await self.client.chat.completions.create(
            model="solar-1-mini-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        return response.choices[0].message.content
```

## 4. 단계별 통합 전략

### Phase 1: 기본 설정 (Day 1-2)
1. **환경 통일**
   ```bash
   # 팀 전체 동일 환경 구축
   bash scripts/setup_team_env.sh
   ```

2. **데이터 전처리 적용**
   ```python
   # baseline.ipynb 수정
   from src.dataset.preprocess import DialoguePreprocessor, add_special_tokens_to_tokenizer
   
   # 전처리기 초기화
   preprocessor = DialoguePreprocessor()
   
   # 토크나이저 업데이트
   tokenizer = add_special_tokens_to_tokenizer(tokenizer)
   ```

3. **베이스라인 재실행**
   - Special token 추가 후 성능 측정
   - WandB에 "baseline_v2" 태그로 기록

### Phase 2: 데이터 개선 (Day 3-4)
1. **Solar API 데이터 증강**
   ```python
   # 누락 정보 보완
   enhanced_summaries = await enhance_dataset_with_solar(train_df.sample(1000))
   ```

2. **데이터 필터링**
   - 품질 낮은 샘플 제거
   - 토큰 길이 기준 필터링

3. **데이터 증강**
   - Paraphrasing
   - Back-translation

### Phase 3: 모델 최적화 (Day 5-7)
1. **Optuna 하이퍼파라미터 탐색**
2. **모델 앙상블**
3. **후처리 최적화**

## 5. 팀 협업 방안

### 5.1 작업 분담
| 팀원 | 담당 영역 | 산출물 |
|------|----------|--------|
| 송규헌 | 데이터 전처리 | preprocess.py, EDA 노트북 |
| 이상현 | 데이터 분석 | Google Sheets, 품질 리포트 |
| 이영준 | 자동화 | 실험 자동화 스크립트 |
| 기타 | 모델 실험 | 하이퍼파라미터 탐색 |

### 5.2 일일 체크인
```markdown
## 2025-01-27 체크인

### 완료한 작업
- [ ] Special token 추가
- [ ] 전처리 함수 구현
- [ ] 베이스라인 재실행

### 발견한 이슈
- 

### 내일 계획
- 

### 공유사항
- 
```

### 5.3 코드 리뷰 프로세스
1. **브랜치 생성**
   ```bash
   git checkout -b feature/preprocess-abbreviations
   ```

2. **PR 템플릿**
   ```markdown
   ## 변경사항
   - 약어 special token 처리 추가
   - 구어체 정제 함수 구현
   
   ## 성능 변화
   - Before: ROUGE-L 0.4712
   - After: ROUGE-L 0.4823 (+0.0111)
   
   ## 테스트
   - [ ] 단위 테스트 통과
   - [ ] 전체 파이프라인 실행 확인
   ```

### 5.4 실험 추적
```python
# WandB 태그 전략
tags = [
    f"v{version}",           # 버전
    f"preprocess_{method}",  # 전처리 방법
    f"model_{model_name}",   # 모델
    f"exp_{experiment_type}" # 실험 유형
]

wandb.init(
    project="dialogue-summarization",
    tags=tags,
    notes=f"{date} - {description}"
)
```

## 결론

현재 팀이 직면한 주요 과제:
1. **기술적**: 환경 설정 통일, 버전 충돌 해결
2. **데이터**: 약어/인물명 처리, 정보 누락 보완
3. **협업**: 체계적인 실험 관리, 코드 통합

제안하는 접근 방법:
1. **즉시**: Special token 추가, 전처리 통합
2. **단기**: Solar API 활용, 데이터 품질 개선
3. **중기**: 모델 최적화, 앙상블

각자의 강점을 살려 체계적으로 작업하면 좋은 성과를 얻을 수 있을 것입니다!
