# Test.csv 추론 및 채점용 CSV 생성 시스템 설계

## 📋 **프로젝트 개요**

### **목적**
현재 시스템에서 test.csv 추론 및 대회 채점용 CSV 파일 생성 기능을 완전히 구현하여, baseline.py와 동일한 수준의 제출 파일 생성 기능을 제공한다.

### **현재 문제점**
1. **❌ 체크포인트 경로 오류**: 잘못된 경로에서 체크포인트 탐색
2. **❌ 추론 단계 실행 안됨**: 체크포인트를 못 찾아서 test.csv 추론 건너뛰기  
3. **❌ 채점용 CSV 생성 안됨**: 추론이 안되니 제출 파일도 없음
4. **❌ 대회 표준 형식 미지원**: `./prediction/output.csv` 형태 없음
5. **❌ 다중 실험 구분 없음**: 여러 실험 결과를 구분할 수 없음

### **목표**
- ✅ 모든 실험에서 test.csv → 추론 → 채점용 CSV 자동 생성
- ✅ 대회 표준 형식 지원 (`./prediction/output.csv`)
- ✅ 다중 실험 결과 구분 및 추적
- ✅ baseline.py와 동일한 워크플로우 지원
- ✅ 기존 고도화 기능 유지

## 🗂️ **최종 파일 구조**

```
📁 프로젝트/
├── 📁 prediction/                              # 대회 채점용 영역
│   ├── 📁 baseline_kobart_20250802_143022/
│   │   ├── 📄 output.csv                       # 실험 1 채점용
│   │   └── 📄 experiment_metadata.json         # 실험 1 메타데이터
│   ├── 📁 mt5_xlsum_20250802_151055/
│   │   ├── 📄 output.csv                       # 실험 2 채점용  
│   │   └── 📄 experiment_metadata.json
│   ├── 📁 eenzeenee_t5_20250802_164233/
│   │   ├── 📄 output.csv                       # 실험 3 채점용
│   │   └── 📄 experiment_metadata.json
│   ├── 📄 latest_output.csv                    # 최신 실험 결과 (덮어쓰기)
│   ├── 📄 experiment_index.csv                 # 모든 실험 추적
│   └── 📁 history/                             # 백업 보관소
│       ├── 📄 output_baseline_kobart_20250802_143022.csv
│       └── 📄 output_mt5_xlsum_20250802_151055.csv
└── 📁 outputs/                                 # 기존 시스템 영역
    ├── 📁 submissions/                         # 시스템 내부용
    └── 📁 auto_experiments/                    # 실험 관리
```

## 🎯 **핵심 구현 컴포넌트**

### **1. CheckpointFinder**
- **파일**: `utils/checkpoint_finder.py`
- **목적**: 정확한 체크포인트 경로 탐색
- **해결**: 현재 `outputs/dialogue_summarization_*/checkpoints/` 구조에 맞춤

### **2. CompetitionCSVManager**  
- **파일**: `utils/competition_csv_manager.py`
- **목적**: 대회 채점용 CSV 생성 및 다중 실험 관리
- **기능**: 실험별 폴더, 인덱스 관리, 히스토리 백업

### **3. AutoExperimentRunner 수정**
- **파일**: `code/auto_experiment_runner.py`
- **수정**: 추론 로직 강화 및 2단계 폴백 시스템
- **통합**: CheckpointFinder + CompetitionCSVManager

### **4. run_main_5_experiments.sh 수정**
- **목적**: 채점용 파일 생성 확인 및 결과 출력 개선
- **추가**: 실험별 파일 경로 표시 및 최종 요약

## 🔄 **실행 흐름**

```
📁 run_main_5_experiments.sh
└── 📁 auto_experiment_runner.py (각 실험마다)
    ├── 🔧 학습 실행 (trainer.py)
    ├── 📊 체크포인트 생성 (outputs/dialogue_summarization_*/checkpoints/)
    ├── 🔍 CheckpointFinder → 체크포인트 탐색
    ├── 📊 추론 실행 (post_training_inference.py 또는 run_inference.py)
    │   ├── data/test.csv → 추론
    │   └── 결과 DataFrame 생성 (fname, summary)
    └── 💾 CompetitionCSVManager → 채점용 CSV 생성
        ├── ./prediction/{실험명}_{타임스탬프}/output.csv
        ├── ./prediction/latest_output.csv (덮어쓰기)
        ├── ./prediction/experiment_index.csv (추가)
        └── ./prediction/history/ (백업)
```

## 📊 **baseline.py와의 비교**

| 구분 | baseline.py | 설계된 시스템 |
|------|------------|-------------|
| **데이터 흐름** | train.csv → 학습 → test.csv → 추론 | ✅ 동일 |
| **채점용 파일** | `./prediction/output.csv` (1개) | ✅ `./prediction/{실험명}/output.csv` (다중) |
| **파일 형식** | `fname,summary` | ✅ 동일 |
| **실험 구분** | ❌ 단일 실험만 | ✅ 다중 실험 + 타임스탬프 |
| **추적 기능** | ❌ 없음 | ✅ experiment_index.csv |
| **최신 결과** | ❌ 덮어쓰기만 | ✅ latest_output.csv + 개별 보관 |

## 🚀 **예상 실행 결과**

```bash
bash run_main_5_experiments.sh

# 실행 완료 후:
✅ 실험 1 완료!
📁 생성된 채점용 파일들:
  📤 실험별 제출: ./prediction/baseline_kobart_20250802_143022/output.csv
  📤 최신 제출: ./prediction/latest_output.csv  
  📋 실험 인덱스: ./prediction/experiment_index.csv

🏆 채점용 파일 최종 요약:
📊 총 실험 수: 5
🥇 최고 성능 실험: mt5_xlsum_20250802_151055 → ./prediction/mt5_xlsum_20250802_151055/output.csv

📝 채점 제출 방법:
  1. 최신 결과: ./prediction/latest_output.csv
  2. 특정 실험: ./prediction/{실험명}_{타임스탬프}/output.csv  
  3. 인덱스 확인: cat ./prediction/experiment_index.csv
```

## 📚 **관련 문서**

- [체크포인트 탐색기 상세 설계](./01_checkpoint_finder.md)
- [채점용 CSV 관리자 설계](./02_competition_csv_manager.md)  
- [자동 실험 러너 수정 사항](./03_auto_experiment_runner.md)
- [실행 스크립트 개선 사항](./04_run_script_improvements.md)
- [구현 체크리스트](./05_implementation_checklist.md)

---

**작성일**: 2025-08-02  
**최종 업데이트**: 2025-08-02  
**작성자**: Claude + Human  
**상태**: 설계 완료, 구현 대기
