# 📚 NLP 대화 요약 프로젝트 문서

한국어 대화 요약 AI 모델 개발 프로젝트의 통합 문서 센터입니다.

## 🎯 프로젝트 개요

**목표**: 베이스라인 성능(ROUGE-F1 47.12%)을 55-60%로 향상시키는 한국어 대화 요약 모델 개발

**✨ 최신 업데이트 (2024.12)**: 조장님 프로젝트 기술 스택 통합 완료
- 🚀 torch 2.6.0, transformers 4.54.0, pytorch_lightning 2.5.2 업그레이드
- 💾 unsloth + QLoRA로 메모리 사용량 75% 감소
- ⚡ 학습 속도 20-30% 향상, decoder_max_len 200으로 고품질 요약
- 🛠️ 완전 자동화된 환경 관리 시스템

## 📁 문서 구조

### 🚀 [01_getting_started](./01_getting_started/README.md)
**프로젝트를 처음 시작하는 분들을 위한 필수 가이드**
- 대회 개요 및 목표 이해
- 환경 설정 (로컬/AIStages)
- 빠른 시작 가이드

### 📖 [02_user_guides](./02_user_guides/README.md)
**일반 사용자를 위한 작업별 상세 가이드**
- 📊 데이터 분석 - 데이터셋 구조 및 전처리
- 🤖 모델 학습 - 베이스라인 학습 및 튜닝
- 🧪 실험 관리 - WandB 추적 및 자동화
- 📈 평가 - ROUGE 메트릭 및 성능 분석

### 🔧 [03_technical_docs](./03_technical_docs/README.md)
**개발자 및 고급 사용자를 위한 기술 문서**
- 🔌 API 참조 - 핵심 모듈 및 클래스
- 🏗️ 아키텍처 - 프로젝트 구조 및 설계
- 📊 **[코드-문서 일치성 분석](./03_technical_docs/architecture/code_documentation_analysis.md)** - 프로젝트 품질 종합 분석
- 🧮 알고리즘 - Solar API 앙상블 기법

### 🧪 [04_experiments](./04_experiments/README.md)
**실험 관리 및 결과 분석**
- 📊 베이스라인 결과 - 재현 실험 및 벤치마크
- 📝 실험 로그 - 단계별 실험 과정 및 분석
- ⚙️ 자동화 - 실험 자동화 도구 및 모범 사례

### 🚀 [05_deployment](./05_deployment/README.md)
**모델 배포 및 운영**
- 운영 환경 구성 및 모델 서빙
- 대회 제출 가이드 및 최종 보고서

### 🔧 [06_troubleshooting](./06_troubleshooting/README.md)
**문제 해결 및 디버깅**
- 환경 설정, 메모리, 학습 관련 문제 해결
- 알려진 이슈 및 디버깅 가이드

### 👥 [07_development](./07_development/README.md)
**개발자를 위한 기여 가이드**
- 개발 종합 가이드 및 코딩 표준
- 구현 체크리스트 및 우선순위별 요구사항

### 📊 [08_project_management](./08_project_management/README.md)
**팀 협업 및 프로젝트 관리**
- 팀 진행 상황 및 통합 액션 플랜
- 커뮤니케이션 요약 및 인사이트 공유

### 📚 [09_references](./09_references/README.md)
**외부 참고 자료**
- 관련 논문 및 학습 리소스
- 도구 및 라이브러리 참고 문서

## 🎯 사용자별 추천 경로

### 🆕 처음 시작하시는 분
1. [시작하기](./01_getting_started/README.md) - 환경 설정
2. [사용자 가이드 > 모델 학습](./02_user_guides/README.md) - 베이스라인 실행
3. [실험 관리](./04_experiments/README.md) - 성능 개선 실험

### 🧑‍💻 개발자
1. [기술 문서](./03_technical_docs/README.md) - 아키텍처 이해
2. [개발자 가이드](./07_development/README.md) - 기여 방법
3. [API 참조](./03_technical_docs/api_reference/README.md) - 구현 세부사항

### 📊 연구자/분석가
1. [사용자 가이드 > 데이터 분석](./02_user_guides/data_analysis/README.md) - 데이터 탐색
2. [실험 관리](./04_experiments/README.md) - 실험 설계 및 분석
3. [참고 자료](./09_references/README.md) - 관련 연구

### 🛠️ 문제 해결이 필요한 경우
1. [문제 해결](./06_troubleshooting/README.md) - 일반적인 문제 및 버그 수정
2. [개발자 가이드](./07_development/README.md) - 구현 관련 지원
3. [프로젝트 관리](./08_project_management/README.md) - 팀 소통 채널

## ✅ 문서 구조 개편 완료

새로운 9단계 구조로 성공적으로 개편되었습니다:

- ✅ **체계적 재구성** - 사용자 여정 기반 9단계 카테고리 완성
- ✅ **문서 통합 완료** - 기존 분산된 문서들을 논리적으로 재배치
- ✅ **네비게이션 개선** - 직관적인 문서 발견 및 교차 참조 시스템
- ✅ **정리 완료** - 빈 폴더 제거 및 중복 문서 정리

## 📊 개편 성과

### Before (기존 구조)
- 14개 폴더 + 8개 루트 파일로 분산
- 중복 및 불명확한 카테고리 분류
- 사용자 목적별 탐색 어려움

### After (새로운 구조)
- 9개 단계별 체계적 카테고리
- 사용자 여정 기반 논리적 구성
- 명확한 네비게이션 및 교차 참조

## 📞 도움이 필요하시면

- **빠른 문제 해결**: [문제해결 가이드](./06_troubleshooting/README.md)
- **팀 소통**: [프로젝트 관리](./08_project_management/README.md)
- **추가 학습**: [참고 자료](./09_references/README.md)

---

📝 **최종 업데이트**: 2025-07-27  
🎉 **문서 구조 개편**: 완료 ✅
