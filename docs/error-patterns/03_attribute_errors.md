# AttributeError 패턴

## 발생 파일
- data_utils.py
- trainer.py

## 주요 에러
1. `'DataProcessor' object has no attribute 'load_data'`
2. `'output_dir' not defined`
3. 클래스 속성/메서드 접근 실패

## 반복 원인
- 메서드가 클래스 외부에 정의됨
- __init__ 메서드에서 속성 초기화 누락
- self. 접두사 누락

## 실패한 수정 시도들
- load_data를 여러 위치로 이동 → 매번 실패
- output_dir 하드코딩 → 근본 해결 안 됨
- 임시 방편으로 회피 → 다른 AttributeError 발생

## 주의사항
- 클래스 구조 전체를 이해한 후 수정
- __init__ 메서드의 초기화 확인
- 메서드는 클래스 내부에 적절한 들여쓰기로 정의
