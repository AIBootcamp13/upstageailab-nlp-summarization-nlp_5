# 데이터셋 성능 비교 분석 결과

## 기본 통계 비교

| 데이터셋 | 행 수 | 파일 크기 | 평균 라인 길이 |
|---------|-------|-----------|----------------|
| train_18000_2.csv | 179,061 | 18.5 MB | 108.2 bytes/line |
| train_japan_backtranslation_filtered.csv | 236,178 | 24.5 MB | 108.7 bytes/line |

**차이점:**
- train_japan_backtranslation_filtered.csv가 31.9% 더 많은 데이터 보유
- 평균 라인 길이는 거의 동일

## 성능 비교 결과

### train_japan_backtranslation_filtered.csv
- 최종 성능 (Epoch 4.81): ROUGE-1=0.3695, ROUGE-2=0.1389, ROUGE-L=0.3485
- 훈련 안정성: 5 epoch 내에서 안정적 수렴

### train_18000_2.csv  
- 최종 성능 (Epoch 7.69): ROUGE-1=0.3655, ROUGE-2=0.1369, ROUGE-L=0.3435
- 훈련 안정성: 더 긴 훈련 시간 필요, 성능 정체 현상

## 결론

**권장사항: train_japan_backtranslation_filtered.csv 사용**

### 이유:
1. **우수한 성능**: 모든 ROUGE 지표에서 1.1-1.5% 높은 성능
2. **더 많은 데이터**: 31.9% 더 많은 훈련 데이터로 모델 일반화 능력 향상
3. **훈련 효율성**: 더 적은 epoch으로 좋은 성능 달성
4. **안정성**: 일관된 성능 향상 곡선

### 적용:
- config_base.yaml의 train_data를 train_japan_backtranslation_filtered.csv로 변경 완료
- 향후 모든 실험은 이 데이터셋을 기준으로 진행

## 데이터 품질 분석
- 두 데이터셋 모두 동일한 구조와 형식 사용
- train_japan_backtranslation_filtered.csv는 역번역(backtranslation)을 통한 데이터 증강이 포함되어 있어 다양성과 품질이 더 우수한 것으로 판단됨
