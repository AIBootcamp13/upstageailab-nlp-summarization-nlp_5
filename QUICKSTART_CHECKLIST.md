# 🚀 AIStages 대회 빠른 시작 체크리스트

## 1️⃣ Day 1: 환경 설정 (2시간)
- [ ] AIStages 서버 접속
- [ ] 자동 설정 스크립트 실행
  ```bash
  bash code/scripts/setup_aistages.sh
  ```
- [ ] UV로 패키지 설치 (10배 빠름!)
  ```bash
  uv pip install -r requirements.txt --system
  ```
- [ ] Jupyter Notebook 테스트
- [ ] GPU 사용 가능 확인

## 2️⃣ Day 2: 데이터 분석 & 전처리 (3시간)
- [ ] 데이터 EDA 실행
  ```python
  # 데이터 로드
  train = pd.read_csv('data/train.csv')
  
  # 길이 분석
  train['dialogue_length'] = train['dialogue'].apply(len)
  print(train['dialogue_length'].describe())
  ```
- [ ] Special Token 추가
  ```python
  special_tokens = ['#Person1#', '#Person2#', ..., '#Email#']
  tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
  ```
- [ ] 구어체 정제
  ```python
  train['dialogue'] = train['dialogue'].apply(clean_dialogue)
  ```
- [ ] 워드클라우드 생성

## 3️⃣ Day 3: WandB 설정 & 베이스라인 (2시간)
- [ ] WandB 가입 및 API Key 설정
- [ ] 팀 생성/참여
- [ ] 프로젝트 생성
- [ ] 베이스라인 실행
  ```python
  wandb.init(
      project="dialogue-summarization",
      entity="your-team",
      config=config
  )
  ```
- [ ] 베이스라인 점수 기록

## 4️⃣ Day 4-5: 하이퍼파라미터 튜닝 (4시간)
- [ ] WandB Sweep 설정
  ```bash
  # WandB 로그인 (이미 완료되어 있다면 스킵)
  wandb login
  - [ ] WandB Sweep YAML 설정 확인
    ```yaml
    # config/sweep/hyperparameter_sweep.yaml 예시
    method: bayes
    metric:
      name: rouge_combined_f1
      goal: maximize
    parameters:
      learning_rate:
        distribution: log_uniform_values
        min: 1.0e-6
        max: 1.0e-4
    ```
  - [ ] Sweep 실행
    ```bash
    # 하이퍼파라미터 튜닝 (50회)
    python code/sweep_runner.py \
        --base-config config/base_config.yaml \
        --sweep-config hyperparameter_sweep \
        --count 50
    ```

## 5️⃣ Day 6-7: 성능 개선 (6시간)
- [ ] 데이터 증강 적용
- [ ] 학습률 스케줄러 실험
- [ ] 앙상블 준비
- [ ] 최종 모델 선정

## 📋 일일 루틴
### 아침 (30분)
- [ ] Git pull로 코드 동기화
- [ ] WandB에서 밤새 실행한 실험 확인
- [ ] GPU 서버 상태 확인

### 실험 전 (15분)
- [ ] Config 파일 백업
- [ ] 이전 실험 결과 리뷰
- [ ] 실험 계획 문서화

### 실험 후 (15분)
- [ ] 결과 스크린샷 저장
- [ ] 실험 로그 작성
- [ ] 팀 채널에 공유

## 🎯 성능 목표
| 단계 | ROUGE-L 목표 | 비고 |
|------|-------------|------|
| 베이스라인 | 0.47+ | 기본 설정 |
| HP 튜닝 후 | 0.49+ | WandB Sweep 최적화 |
| 데이터 증강 | 0.50+ | 전처리 개선 |
| 최종 | 0.52+ | 앙상블 포함 |

## 💡 꿀팁
1. **UV 사용**: 패키지 설치가 10배 빨라집니다
2. **WandB Group**: 실험을 그룹으로 묶어 비교하기 쉽게
3. **Early Stopping**: 과적합 방지 및 시간 절약
4. **Mixed Precision**: fp16=True로 2배 빠른 학습
5. **Git 브랜치**: 실험별로 브랜치 생성하여 관리

## ⚠️ 주의사항
- 일일 제출 횟수: 12회 제한
- 최종 제출: 최대 2개 선택
- DialogSum 데이터셋 사용 금지
- 평가 데이터 학습 사용 금지

## 📞 도움이 필요하면
1. 팀 슬랙 채널에 질문
2. [통합 가이드](docs/competition_guides/competition_integration_guide.md) 참고
3. 각 단계별 상세 가이드 확인

---

**화이팅! 🔥 최고의 성능을 달성해봅시다!**
