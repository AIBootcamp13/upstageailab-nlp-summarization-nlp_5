# 팀 커뮤니케이션 기반 개선사항 요약

## 개요
슬랙 대화와 00_EDA_origin-DialogSum.ipynb 분석을 통해 도출한 주요 개선사항을 정리했습니다.

## 1. 환경 설정 이슈 해결

### 1.1 ModuleNotFoundError 해결
```python
# 모든 파일에서 통일
project_dir = "/data/ephemeral/home/upstageailab-nlp-summarization-nlp_5/"
import sys
sys.path.append(project_dir)
```

### 1.2 의존성 버전 통일
- Python 3.11 (프로젝트 표준)
- Torch 2.4+ (Unsloth 요구사항)
- `evaluation_strategy` → `eval_strategy` 변경

## 2. DialogSum 데이터셋 주요 발견사항

### 2.1 데이터 차이
- **제외된 데이터**: 
  - Train: 3개 (after-sales service 관련)
  - Validation: 1개 (buy furniture)
- **Test 구조**: 원본 1,500개 → 대회 250개

### 2.2 품질 이슈
1. **정보 누락**: 중요한 선호/혐오 정보 누락
2. **표현 불일치**: 무섭다/싫다, 좋다/멋있다 혼용
3. **약어 문제**: EDD, ETV, BBC 등 토큰화 이슈
4. **인물명**: 알버트 아인슈타인 등 처리 필요

### 2.3 논문 인사이트
- **Coreference 오류**: 60-94%
- **Intent 파악 실패**: 30-84%
- **담화 구조 이해 중요성**

## 3. 즉시 적용 가능한 개선사항

### 3.1 Special Token 추가
```python
special_tokens = [
    # 발화자
    '#Person1#', '#Person2#', ..., '#Person7#',
    # PII 마스킹
    '#PhoneNumber#', '#Address#', '#DateOfBirth#',
    '#PassportNumber#', '#SSN#', '#CardNumber#',
    '#CarNumber#', '#Email#',
    # 약어 (새로 추가)
    '#ATM#', '#AS#', '#BBC#', '#CEO#', '#EDD#', '#ETV#'
]
```

### 3.2 전처리 개선
```python
# A/S → AS 통일
text = text.replace('A/S', 'AS')

# 구어체 정제
text = re.sub(r'ㅋ+', '웃음', text)
text = re.sub(r'ㅎ+', '웃음', text)
text = re.sub(r'ㅠ+|ㅜ+', '슬픔', text)
```

### 3.3 Topic Dictionary 활용
- topic_dict_cleaned.yaml: 영어→한글 매핑 6,526개
- 일관된 topic 번역 적용

## 4. Solar API 활용 방안

### 4.1 자동화 대상
1. **약어 감지**: 대화문에서 약어 자동 추출
2. **누락 정보 확인**: 요약문 품질 개선
3. **동의어 정규화**: 표현 일관성 확보

### 4.2 구현 예시
```python
async def detect_abbreviations(text):
    """Solar API로 약어 자동 감지"""
    prompt = f"다음 텍스트에서 약어를 찾아주세요: {text}"
    # API 호출 및 처리
```

## 5. 팀 작업 현황 및 계획

### 5.1 현재 진행 상황
- **송규헌**: 번역 노이즈 처리 (00_EDA_origin-DialogSum.ipynb)
- **이상현**: 데이터 분석, Google Sheets 정리
- **이영준**: 자동화 스크립트 개발

### 5.2 TODO 작업 프로세스
1. 각자 브랜치에서 작업
2. TODO 선택 후 Person 지정
3. ipynb 파일 생성 및 분석
4. 결과를 TODO 페이지에 정리
5. 함수화하여 재사용 가능하게

### 5.3 우선순위
1. **즉시**: 환경 설정 통일, Special Token 추가
2. **단기**: 전처리 개선, Solar API 테스트
3. **중기**: 데이터 증강, 모델 최적화

## 6. 성능 향상 전략

### 6.1 데이터 측면
- Special Token으로 약어/인물명 보호
- 누락된 정보 보완
- 동의어 정규화

### 6.2 모델 측면
- Discourse-aware 접근
- Coreference resolution 강화
- Multi-task learning 고려

### 6.3 평가 측면
- ROUGE의 한계 인식 (토큰 일치 기반)
- Test에서 summary 3개로 다양성 평가
- 동의어 처리 전략 필요

## 결론

팀이 발견한 주요 이슈들을 체계적으로 해결하면:
1. **기술적 이슈**: 환경 통일로 개발 효율성 향상
2. **데이터 품질**: 전처리로 모델 입력 개선
3. **자동화**: Solar API로 수작업 최소화

각자의 TODO를 통해 점진적으로 개선하며, 결과를 공유하여 팀 전체의 성과로 만들어가는 것이 중요합니다.
