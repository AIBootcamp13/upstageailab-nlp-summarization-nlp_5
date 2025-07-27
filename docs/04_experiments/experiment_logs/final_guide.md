# 최종 실험 분석 및 제출 준비 가이드

## 현재 상태

### 완료된 작업
1. **베이스라인 재현** (ROUGE-F1: 47.12%)
2. **1차 개선 구현** (우선순위 1-5)
   - 간단한 데이터 증강
   - 후처리 파이프라인
   - 학습률 스케줄링
   - 텍스트 정규화
   - 배치 크기 최적화
3. **1차 조합 실험** (우선순위 6)
   - 최적 조합 발견
4. **고급 기능 구현** (우선순위 7-9)
   - 특수 토큰 가중치
   - 빔 서치 최적화
   - 백트랜슬레이션
5. **2차 조합 실험 준비** (우선순위 10)
   - YAML 설정 파일 생성
   - 분석 스크립트 준비

### 진행 중인 작업
- **2차 조합 실험 실행** (taskId: 79598b91-a29e-479f-9c31-d880296b9bb8)
  - 10a: Phase1 + Token Weight
  - 10b: Phase1 + BackTranslation
  - 10c: All Optimizations

### 남은 작업
1. **Solar API 앙상블 통합** (우선순위 11)
2. **최종 통합 테스트 및 제출 준비**

## 실행 가이드

### 1. 2차 조합 실험 실행
```bash
# 전체 2차 실험 실행 (약 20-25시간 소요)
./run_phase2_experiments.sh

# 또는 개별 실행
# 토큰 가중치 실험 (약 4-5시간)
python code/auto_experiment_runner.py \
    --config config/experiments/10_combination_phase2/10a_phase1_plus_token_weight.yaml

# 백트랜슬레이션 실험 (약 6-7시간)
python code/auto_experiment_runner.py \
    --config config/experiments/10_combination_phase2/10b_phase1_plus_backtrans.yaml

# 전체 통합 실험 (약 8-10시간)
python code/auto_experiment_runner.py \
    --config config/experiments/10_combination_phase2/10c_all_optimizations.yaml
```

### 2. 실험 모니터링
```bash
# WandB 대시보드에서 실시간 모니터링
# https://wandb.ai/your-project/nlp-summarization

# 로컬 로그 확인
tail -f logs/10_combination_phase2/*/train.log

# GPU 사용량 모니터링
watch -n 1 nvidia-smi
```

### 3. 결과 분석
```bash
# 2차 실험 분석 실행
python scripts/analyze_combination_phase2.py

# 분석 결과 확인
cat outputs/phase2_analysis/final_analysis_phase2.md
```

## 예상 결과 및 의사결정 기준

### 예상 성능
| 실험 | 예상 ROUGE-F1 | 실제 달성 | 차이 |
|------|--------------|-----------|------|
| 베이스라인 | 47.12% | - | - |
| 1차 최적 조합 | 50-51% | TBD | TBD |
| 10a (토큰 가중치) | 52-53% | TBD | TBD |
| 10b (백트랜슬레이션) | 53-54% | TBD | TBD |
| 10c (전체 통합) | 54-56% | TBD | TBD |

### 최종 모델 선택 기준
1. **성능**: ROUGE-F1 점수 (가중치 40%)
2. **효율성**: 학습/추론 시간 (가중치 20%)
3. **안정성**: 검증 세트 성능 분산 (가중치 20%)
4. **실용성**: 메모리 사용량, 배포 용이성 (가중치 20%)

### 의사결정 트리
```
IF ROUGE-F1 >= 55% AND 추론시간 < 2초/샘플:
    → 10c (전체 통합) 선택
ELIF ROUGE-F1 >= 53% AND 메모리 < 14GB:
    → 10b (백트랜슬레이션) 선택
ELIF ROUGE-F1 >= 52%:
    → 10a (토큰 가중치) 선택
ELSE:
    → Solar API 앙상블 시도
```

## 다음 단계 상세 계획

### Phase 1: 2차 실험 완료 (현재)
- [ ] 10a 실험 실행 및 모니터링
- [ ] 10b 실험 실행 및 모니터링
- [ ] 10c 실험 실행 및 모니터링
- [ ] 결과 분석 및 최적 모델 선택

### Phase 2: Solar API 통합 (선택적)
목표 성능(55%)에 도달하지 못한 경우:
```bash
# Solar API 설정
export SOLAR_API_KEY="your-api-key"

# 앙상블 실험 실행
python code/run_solar_ensemble.py \
    --base_model "outputs/phase2_results/best_model" \
    --solar_model "solar-1-mini-chat" \
    --ensemble_method "weighted_average"
```

### Phase 3: 최종 제출 준비
1. **전체 데이터로 재학습**
   ```bash
   python code/trainer.py \
       --config config/experiments/final_submission.yaml \
       --full_data true \
       --save_dir models/final_submission
   ```

2. **테스트 세트 추론**
   ```bash
   python code/run_inference.py \
       --model_path models/final_submission \
       --test_file test.csv \
       --output_file submission.csv \
       --apply_postprocessing true
   ```

3. **제출 파일 검증**
   ```bash
   python scripts/validate_submission.py \
       --submission_file submission.csv \
       --sample_file sample_submission.csv
   ```

## 트러블슈팅 가이드

### 일반적인 문제 해결
1. **메모리 부족**
   - gradient_accumulation_steps 증가
   - per_device_batch_size 감소
   - fp16 활성화 확인

2. **학습 불안정**
   - learning_rate 감소
   - warmup_ratio 증가
   - gradient_clipping 적용

3. **과적합**
   - dropout 증가
   - weight_decay 증가
   - 데이터 증강 비율 증가

### 성능 미달 시 추가 시도
1. **앙상블 방법**
   - 다른 에폭의 체크포인트 앙상블
   - 다른 시드로 학습한 모델 앙상블
   - Solar API와의 앙상블

2. **추가 최적화**
   - Mixed precision training
   - Gradient accumulation 최적화
   - Learning rate finder 실행

## 코드 품질 체크리스트

- [ ] 모든 실험이 재현 가능한가?
- [ ] 설정 파일이 명확히 문서화되었는가?
- [ ] 로깅이 충분히 상세한가?
- [ ] 에러 처리가 적절한가?
- [ ] 메모리 누수가 없는가?

## 최종 체크리스트

제출 전 확인사항:
- [ ] 최고 성능 모델 선택 완료
- [ ] 전체 데이터로 재학습 완료
- [ ] 테스트 세트 추론 완료
- [ ] 후처리 적용 완료
- [ ] 제출 파일 형식 검증 완료
- [ ] 특수 토큰 포함 여부 확인
- [ ] ROUGE 점수 예상치 계산
- [ ] 실행 시간 요구사항 충족
- [ ] 메모리 사용량 요구사항 충족
- [ ] 코드 정리 및 문서화 완료

## 연락처 및 지원

문제 발생 시:
1. 프로젝트 로그 확인: `logs/` 디렉토리
2. WandB 대시보드 확인
3. 에러 메시지와 함께 이슈 생성

---

마지막 업데이트: 2025-01-27
