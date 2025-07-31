# Import Error 패턴

## 발생 모듈
- SmartDataCollatorForSeq2Seq
- utils 모듈들
- 상대/절대 경로 혼란

## 반복 횟수
5회 이상

## 주요 증상
- ModuleNotFoundError
- ImportError
- 순환 import 문제

## 실패한 수정 시도들
- import 경로만 수정 → 다른 모듈에서 에러
- sys.path 조작 → 일시적 해결
- 상대 경로와 절대 경로 혼용

## 주의사항
- 프로젝트 구조 먼저 파악
- __init__.py 파일 확인
- 순환 import 회피
- 환경별 경로 차이 고려
