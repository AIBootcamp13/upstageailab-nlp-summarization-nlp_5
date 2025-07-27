# 🔗 앙상블 기법 가이드

다중 모델을 결합하여 성능을 향상시키는 앙상블 기법들에 대한 상세 가이드입니다.

## 📋 목차
1. [앙상블 개요](#앙상블-개요)
2. [앙상블 전략](#앙상블-전략)
3. [구현 방법](#구현-방법)
4. [성능 최적화](#성능-최적화)
5. [실험 결과](#실험-결과)
6. [실무 적용](#실무-적용)

---

## 앙상블 개요

### 앙상블의 기본 원리

앙상블은 여러 개의 독립적인 모델을 결합하여 단일 모델보다 더 나은 성능을 얻는 기법입니다.

**핵심 개념:**
- **다양성 (Diversity)**: 서로 다른 특성을 가진 모델들
- **집단 지성 (Collective Intelligence)**: 개별 모델의 약점을 서로 보완
- **분산 감소 (Variance Reduction)**: 예측의 안정성 향상

### 대화 요약에서의 앙상블 효과

```python
# 단일 모델 vs 앙상블 성능 비교
PERFORMANCE_COMPARISON = {
    "KoBART (단일)": {
        "rouge1_f1": 0.471,
        "rouge2_f1": 0.312,
        "rougeL_f1": 0.395
    },
    "3-Model Ensemble": {
        "rouge1_f1": 0.485,  # +1.4%p 향상
        "rouge2_f1": 0.329,  # +1.7%p 향상
        "rougeL_f1": 0.408   # +1.3%p 향상
    }
}
```

---

## 앙상블 전략

### 1. 모델 다양성 확보

#### A. 아키텍처 다양성
```python
ENSEMBLE_MODELS = {
    "kobart": {
        "model_path": "outputs/kobart_model",
        "architecture": "encoder-decoder",
        "strength": "한국어 이해",
        "weight": 0.4
    },
    "kt5": {
        "model_path": "outputs/kt5_model", 
        "architecture": "encoder-decoder",
        "strength": "텍스트 생성",
        "weight": 0.3
    },
    "mt5": {
        "model_path": "outputs/mt5_model",
        "architecture": "encoder-decoder", 
        "strength": "다국어 지원",
        "weight": 0.3
    }
}
```

### 2. 앙상블 결합 방법

#### A. Voting 기반 앙상블
```python
class VotingEnsemble:
    """투표 기반 앙상블"""
    
    def predict(self, dialogue: str) -> str:
        predictions = []
        weights = []
        
        for model_config in self.models:
            model = self._load_model(model_config["path"])
            pred = model.generate_summary(dialogue)
            
            predictions.append(pred)
            weights.append(model_config["weight"])
        
        return self._weighted_vote(predictions, weights)
```

#### B. 다단계 앙상블
```python
class MultiStageEnsemble:
    """다단계 앙상블 (조건부 결합)"""
    
    def predict(self, dialogue: str) -> str:
        # 1. 입력 분석
        features = self._analyze_dialogue(dialogue)
        
        # 2. 전략 선택
        strategy = self._select_strategy(features)
        
        # 3. 선택된 전략으로 앙상블
        if strategy == "simple":
            return self._simple_ensemble(dialogue)
        elif strategy == "weighted":
            return self._weighted_ensemble(dialogue, features)
        else:
            return self._adaptive_ensemble(dialogue, features)
```

---

## 구현 방법

### 기본 앙상블 프레임워크

```python
# core/ensemble/ensemble_model.py
class EnsemblePredictor:
    """다중 모델 앙상블 예측기"""
    
    def __init__(self, model_configs: List[Dict[str, Any]]):
        self.models = []
        self.weights = []
        
        for config in model_configs:
            model = self._load_model(config['path'])
            weight = config.get('weight', 1.0)
            
            self.models.append(model)
            self.weights.append(weight)
        
        # 가중치 정규화
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
    
    def predict_ensemble(self, dialogue: str, strategy: str = "weighted") -> str:
        """앙상블 예측"""
        
        predictions = []
        for model in self.models:
            pred = model.generate_summary(dialogue)
            predictions.append(pred)
        
        if strategy == "weighted":
            return self._weighted_combine(predictions)
        elif strategy == "voting":
            return self._majority_vote(predictions)
        elif strategy == "best":
            return self._select_best(predictions, dialogue)
        else:
            return predictions[0]
    
    def _weighted_combine(self, predictions: List[str]) -> str:
        """가중치 기반 결합"""
        
        # 문장 레벨에서 가중 선택
        sentence_groups = [pred.split('.') for pred in predictions]
        combined_sentences = []
        
        max_sentences = max(len(group) for group in sentence_groups)
        
        for i in range(max_sentences):
            sentence_scores = {}
            
            for j, (sentences, weight) in enumerate(zip(sentence_groups, self.weights)):
                if i < len(sentences) and sentences[i].strip():
                    sent = sentences[i].strip()
                    sentence_scores[sent] = sentence_scores.get(sent, 0) + weight
            
            if sentence_scores:
                best_sentence = max(sentence_scores, key=sentence_scores.get)
                combined_sentences.append(best_sentence)
        
        return '. '.join(combined_sentences)
    
    def _majority_vote(self, predictions: List[str]) -> str:
        """다수결 투표"""
        
        from collections import defaultdict
        import difflib
        
        # 유사한 예측 그룹화
        groups = defaultdict(list)
        
        for i, pred in enumerate(predictions):
            assigned = False
            for group_rep in groups:
                similarity = difflib.SequenceMatcher(None, pred, group_rep).ratio()
                if similarity > 0.7:
                    groups[group_rep].append((i, pred))
                    assigned = True
                    break
            
            if not assigned:
                groups[pred].append((i, pred))
        
        # 최대 그룹에서 최고 가중치 선택
        largest_group = max(groups.values(), key=len)
        
        best_pred = None
        best_weight = 0
        
        for idx, pred in largest_group:
            if self.weights[idx] > best_weight:
                best_weight = self.weights[idx]
                best_pred = pred
        
        return best_pred
```

---

## 성능 최적화

### 1. 메모리 효율적 앙상블

```python
class MemoryEfficientEnsemble:
    """메모리 효율적 앙상블"""
    
    def predict_streaming(self, dialogue: str) -> str:
        """스트리밍 방식 예측"""
        
        predictions = []
        
        for config in self.model_configs:
            # 모델 로딩
            model = self._load_model_lazy(config['path'])
            
            # 예측
            pred = model.generate_summary(dialogue)
            predictions.append((pred, config['weight']))
            
            # 메모리 정리
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return self._weighted_combine(predictions)
```

### 2. 배치 처리 최적화

```python
class BatchEnsembleProcessor:
    """배치 앙상블 처리"""
    
    def process_batch(self, dialogues: List[str]) -> List[str]:
        """효율적인 배치 처리"""
        
        # 모든 모델의 배치 예측 수집
        all_predictions = []
        
        for model in self.models:
            batch_preds = model.predict_batch(dialogues)
            all_predictions.append(batch_preds)
        
        # 샘플별 앙상블
        results = []
        for i in range(len(dialogues)):
            sample_preds = [preds[i] for preds in all_predictions]
            ensemble_result = self._combine_predictions(sample_preds)
            results.append(ensemble_result)
        
        return results
```

---

## 실험 결과

### 앙상블 성능 비교

```python
ENSEMBLE_RESULTS = {
    "베이스라인 (단일 KoBART)": {
        "rouge1_f1": 0.471,
        "rouge2_f1": 0.312,
        "rougeL_f1": 0.395,
        "inference_time": "1.2s/sample"
    },
    "2모델 앙상블 (KoBART + KE-T5)": {
        "rouge1_f1": 0.482,
        "rouge2_f1": 0.324,
        "rougeL_f1": 0.403,
        "inference_time": "2.8s/sample"
    },
    "3모델 앙상블 (+ mT5)": {
        "rouge1_f1": 0.485,
        "rouge2_f1": 0.329,
        "rougeL_f1": 0.408,
        "inference_time": "3.8s/sample"
    },
    "5모델 앙상블": {
        "rouge1_f1": 0.491,
        "rouge2_f1": 0.337,
        "rougeL_f1": 0.415,
        "inference_time": "6.2s/sample"
    }
}
```

### 모델 조합 효과

| 조합 | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | 개선률 |
|------|-------------|-------------|-------------|--------|
| KoBART 단일 | 0.471 | 0.312 | 0.395 | - |
| + KE-T5 | 0.482 | 0.324 | 0.403 | +2.3% |
| + mT5 | 0.485 | 0.329 | 0.408 | +3.0% |
| + Solar API | 0.488 | 0.332 | 0.411 | +3.6% |

---

## 실무 적용

### 1. 프로덕션 배포

```python
# deployment/ensemble_api.py
class EnsembleAPI:
    """앙상블 모델 API"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.ensemble = self._setup_ensemble()
        
    def summarize(self, dialogue: str, strategy: str = "auto") -> Dict[str, Any]:
        """앙상블 요약 생성"""
        
        start_time = time.time()
        
        # 자동 전략 선택
        if strategy == "auto":
            strategy = self._select_optimal_strategy(dialogue)
        
        # 앙상블 예측
        summary = self.ensemble.predict_ensemble(dialogue, strategy)
        
        processing_time = time.time() - start_time
        
        return {
            "summary": summary,
            "strategy_used": strategy,
            "processing_time": processing_time,
            "model_count": len(self.ensemble.models)
        }
    
    def _select_optimal_strategy(self, dialogue: str) -> str:
        """대화 특성에 따른 최적 전략 선택"""
        
        length = len(dialogue.split())
        
        if length < 50:
            return "simple"  # 짧은 대화는 단순 전략
        elif length > 300:
            return "weighted"  # 긴 대화는 가중 전략
        else:
            return "voting"  # 중간 길이는 투표 전략
```

### 2. 성능 모니터링

```python
class EnsembleMonitor:
    """앙상블 성능 모니터링"""
    
    def __init__(self):
        self.metrics = {
            "total_requests": 0,
            "average_latency": 0,
            "strategy_usage": defaultdict(int),
            "model_agreement": []
        }
    
    def log_prediction(self, dialogue: str, predictions: List[str], strategy: str, latency: float):
        """예측 결과 로깅"""
        
        self.metrics["total_requests"] += 1
        self.metrics["average_latency"] = (
            (self.metrics["average_latency"] * (self.metrics["total_requests"] - 1) + latency) 
            / self.metrics["total_requests"]
        )
        self.metrics["strategy_usage"][strategy] += 1
        
        # 모델 간 일치도 계산
        agreement = self._calculate_agreement(predictions)
        self.metrics["model_agreement"].append(agreement)
    
    def _calculate_agreement(self, predictions: List[str]) -> float:
        """모델 간 예측 일치도 계산"""
        
        total_pairs = 0
        agreement_sum = 0
        
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                similarity = self._calculate_similarity(predictions[i], predictions[j])
                agreement_sum += similarity
                total_pairs += 1
        
        return agreement_sum / total_pairs if total_pairs > 0 else 0
    
    def get_performance_report(self) -> Dict[str, Any]:
        """성능 리포트 생성"""
        
        return {
            "총 요청 수": self.metrics["total_requests"],
            "평균 응답시간": f"{self.metrics['average_latency']:.2f}초",
            "전략 사용률": dict(self.metrics["strategy_usage"]),
            "평균 모델 일치도": f"{np.mean(self.metrics['model_agreement']):.3f}",
            "일치도 표준편차": f"{np.std(self.metrics['model_agreement']):.3f}"
        }
```

### 3. A/B 테스트 프레임워크

```python
class EnsembleABTest:
    """앙상블 A/B 테스트"""
    
    def __init__(self, control_model, test_ensemble, split_ratio: float = 0.5):
        self.control_model = control_model
        self.test_ensemble = test_ensemble
        self.split_ratio = split_ratio
        self.results = {"control": [], "test": []}
    
    def predict_with_ab_test(self, dialogue: str) -> Dict[str, Any]:
        """A/B 테스트를 통한 예측"""
        
        import random
        
        # 사용자 그룹 결정
        use_ensemble = random.random() < self.split_ratio
        
        start_time = time.time()
        
        if use_ensemble:
            summary = self.test_ensemble.predict_ensemble(dialogue)
            group = "test"
        else:
            summary = self.control_model.generate_summary(dialogue)
            group = "control"
        
        latency = time.time() - start_time
        
        # 결과 기록
        self.results[group].append({
            "dialogue_length": len(dialogue.split()),
            "summary_length": len(summary.split()),
            "latency": latency
        })
        
        return {
            "summary": summary,
            "group": group,
            "latency": latency
        }
    
    def get_ab_test_results(self) -> Dict[str, Any]:
        """A/B 테스트 결과 분석"""
        
        control_latencies = [r["latency"] for r in self.results["control"]]
        test_latencies = [r["latency"] for r in self.results["test"]]
        
        return {
            "control_group": {
                "requests": len(self.results["control"]),
                "avg_latency": np.mean(control_latencies),
                "avg_summary_length": np.mean([r["summary_length"] for r in self.results["control"]])
            },
            "test_group": {
                "requests": len(self.results["test"]),
                "avg_latency": np.mean(test_latencies),
                "avg_summary_length": np.mean([r["summary_length"] for r in self.results["test"]])
            },
            "performance_gain": {
                "latency_ratio": np.mean(test_latencies) / np.mean(control_latencies),
                "request_distribution": f"Control: {len(control_latencies)}, Test: {len(test_latencies)}"
            }
        }
```

---

## 🎯 앙상블 활용 가이드

### 언제 앙상블을 사용해야 할까?

#### 추천 상황
- **높은 정확도가 중요한 경우**: 의료, 법률 등 도메인
- **다양한 입력이 예상되는 경우**: 여러 도메인의 대화
- **안정적인 성능이 필요한 경우**: 프로덕션 환경

#### 비추천 상황
- **실시간 처리가 중요한 경우**: 채팅봇, 실시간 서비스
- **리소스가 제한적인 경우**: 모바일, 엣지 디바이스
- **단순한 태스크의 경우**: 기본적인 요약만 필요

### 최적 앙상블 구성

#### 2-3개 모델 앙상블 (권장)
```python
RECOMMENDED_ENSEMBLE = {
    "모델 수": 2-3,
    "다양성": "아키텍처 또는 학습 전략",
    "가중치": "성능 기반 설정",
    "전략": "가중 투표",
    "예상 성능 향상": "2-4%p",
    "지연시간 증가": "2-3배"
}
```

#### 고성능 앙상블 (연구용)
```python
RESEARCH_ENSEMBLE = {
    "모델 수": 5-7,
    "다양성": "아키텍처 + 데이터 + 학습",
    "가중치": "동적 조정",
    "전략": "다단계 앙상블",
    "예상 성능 향상": "4-6%p",
    "지연시간 증가": "5-10배"
}
```

---

## 🔗 관련 문서

- [Solar API 통합](./solar_api_integration.md) - 외부 API 앙상블
- [성능 분석](../../02_user_guides/evaluation/performance_analysis.md) - 성능 평가
- [배포 가이드](../../05_deployment/README.md) - 프로덕션 배포
- [실험 관리](../../04_experiments/README.md) - 앙상블 실험

---

앙상블 기법을 통해 단일 모델의 한계를 극복하고 더 안정적이고 정확한 대화 요약 시스템을 구축하세요.
