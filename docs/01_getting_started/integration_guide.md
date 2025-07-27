# 기존 프로젝트와 새 환경 설정 통합 가이드

## 개요
이 문서는 기존 `nlp-sum-lyj` 프로젝트에 새로운 AIStages 환경 설정을 적용하는 방법을 설명합니다.

## 1. 현재 상황 분석

### 1.1 기존 문서화 상태

#### 이미 문서화된 내용
- **UV 패키지 관리자**: `docs/uv_package_manager_guide.md`에 상세히 문서화
  - UV 설치 방법
  - 기본 사용법
  - 성능 비교
  - 트러블슈팅

#### 새로 추가된 내용
- **AIStages 특화 설정**: `docs/setup_guides/aistages_environment_setup.md`
  - Base 가상환경 초기화
  - AIStages Github 설정
  - Fork & Clone 워크플로우
  - Config 파일 구조
  - Main 파일 실행 방법

### 1.2 기존 코드 구조
```
code/
├── baseline.ipynb          # KoBART 기반 학습/추론
├── solar_api.ipynb        # Solar API 활용
├── config.yaml            # 간단한 설정 파일
└── requirements.txt       # 의존성 목록
```

### 1.3 새로운 구조 요구사항
```
src/
├── main_base.py           # 통합 실행 스크립트
├── configs/
│   └── generate_config.ipynb  # 설정 파일 생성기
└── [기타 모듈]
```

## 2. 통합 전략

### 2.1 단계별 접근

#### Phase 1: 환경 설정 (즉시 적용 가능)
1. UV 설치 및 환경 설정
2. Git 설정
3. 시스템 라이브러리 설치

#### Phase 2: 코드 구조 개선 (선택적)
1. 기존 notebook 코드를 모듈화
2. Config 관리 시스템 통합
3. 실행 스크립트 통합

### 2.2 호환성 유지
- 기존 `baseline.ipynb`는 그대로 사용 가능
- 새로운 환경에서도 정상 작동
- 점진적 마이그레이션 지원

## 3. 실제 적용 방법

### 3.1 최소 변경으로 시작하기

#### Step 1: UV 환경 설정
```bash
# 기존 pip 환경을 UV로 전환
conda activate base
uv pip install -r code/requirements.txt --system
```

#### Step 2: 기존 코드 실행
```bash
# 기존 방식 그대로 실행
jupyter notebook code/baseline.ipynb
```

### 3.2 권장 설정 적용

#### Step 1: 자동 설정 스크립트 실행
```bash
bash code/scripts/setup_aistages.sh
```

#### Step 2: Config 업데이트
```python
# code/config.yaml 수정
general:
  data_path: "./data/"
  model_name: "digit82/kobart-summarization"
  output_path: "./outputs/"
  
training:
  num_train_epochs: 20
  learning_rate: 3e-5
  per_device_train_batch_size: 16
  
wandb:
  entity: "your_team"
  project: "dialogue-summarization"
```

## 4. 차이점 및 장점

### 4.1 기존 방식
- 수동 pip 설치 (느림)
- 개별 Git 설정
- 분산된 설정 파일
- 수동 실행

### 4.2 새로운 방식
- UV 자동 설치 (10배 빠름)
- 통합 환경 설정
- 체계적인 Config 관리
- 자동화된 실행 스크립트

### 4.3 성능 비교
| 작업 | 기존 (pip) | 새로운 (uv) | 개선율 |
|-----|-----------|------------|--------|
| 패키지 설치 | 90초 | 7초 | 93% |
| 환경 재설정 | 5분 | 30초 | 90% |
| 캐시 활용 | 제한적 | 최적화 | - |

## 5. 마이그레이션 체크리스트

### 5.1 필수 작업
- [ ] UV 설치
- [ ] Git 설정
- [ ] 시스템 라이브러리 설치
- [ ] requirements.txt로 패키지 설치

### 5.2 선택 작업
- [ ] 자동화 스크립트 활용
- [ ] Config 시스템 업그레이드
- [ ] 코드 구조 개선
- [ ] CI/CD 파이프라인 구축

## 6. FAQ

### Q1: 기존 코드를 꼭 수정해야 하나요?
**A**: 아니요. 환경 설정만 변경해도 기존 코드는 그대로 사용 가능합니다.

### Q2: UV가 꼭 필요한가요?
**A**: 필수는 아니지만, 설치 속도가 10배 이상 빨라져 개발 효율이 크게 향상됩니다.

### Q3: 새로운 Config 구조를 사용해야 하나요?
**A**: 기존 config.yaml도 계속 사용 가능합니다. 더 체계적인 관리가 필요할 때 전환하면 됩니다.

### Q4: 팀원들도 같은 설정을 해야 하나요?
**A**: 네, 일관된 환경을 위해 팀원 모두 동일한 설정을 사용하는 것이 좋습니다.

## 7. 결론

새로운 AIStages 환경 설정은:
1. **즉시 적용 가능**: 기존 코드 수정 없이 환경만 개선
2. **점진적 개선**: 필요에 따라 단계별 적용
3. **성능 향상**: 개발 속도 10배 이상 개선
4. **팀 협업 강화**: 일관된 환경과 설정 관리

기존 프로젝트의 안정성을 유지하면서도 개발 효율을 크게 향상시킬 수 있습니다.
