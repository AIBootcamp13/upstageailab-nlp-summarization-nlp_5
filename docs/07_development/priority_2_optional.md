            cpu_info = analysis['cpu_memory']
            f.write(f"  CPU Peak Usage: {cpu_info['max_mb']:.1f} MB ({cpu_info['peak_usage_pct']:.1f}%)\n")
            f.write(f"  CPU Average Usage: {cpu_info['avg_mb']:.1f} MB\n")
            
            if self.gpu_available:
                gpu_info = analysis['gpu_memory']
                f.write(f"  GPU Peak Usage: {gpu_info['max_mb']:.1f} MB ({gpu_info['peak_usage_pct']:.1f}%)\n")
                f.write(f"  GPU Average Usage: {gpu_info['avg_mb']:.1f} MB\n")
                
                leak_info = gpu_info['memory_leaks']
                f.write(f"  Memory Leak Status: {leak_info['status']}\n")
                if leak_info['status'] == 'leak_detected':
                    f.write(f"  Estimated Leak Rate: {leak_info['estimated_leak_rate']:.2f} MB/step\n")
            
            f.write("\nRecommendations:\n")
            for i, rec in enumerate(analysis['recommendations'], 1):
                f.write(f"  {i}. {rec}\n")
        
        # 그래프 생성
        if self.save_plots and self.snapshots:
            self._create_memory_plots(output_dir)
        
        print(f"Memory profiling report saved to: {output_dir}")
    
    def _create_memory_plots(self, output_dir: Path):
        """메모리 사용량 그래프 생성"""
        import matplotlib.pyplot as plt
        plt.style.use('default')
        
        # 데이터 준비
        steps = [s.step for s in self.snapshots]
        cpu_memory = [s.cpu_memory_mb for s in self.snapshots]
        gpu_memory = [s.gpu_memory_mb for s in self.snapshots]
        gpu_cached = [s.gpu_memory_cached_mb for s in self.snapshots]
        
        # 그래프 생성
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # CPU 메모리 그래프
        ax1.plot(steps, cpu_memory, 'b-', label='CPU Memory Used', linewidth=2)
        ax1.axhline(y=self.cpu_total_memory, color='r', linestyle='--', alpha=0.7, label='CPU Total Memory')
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Memory (MB)')
        ax1.set_title('CPU Memory Usage')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # GPU 메모리 그래프
        if self.gpu_available:
            ax2.plot(steps, gpu_memory, 'g-', label='GPU Memory Used', linewidth=2)
            ax2.plot(steps, gpu_cached, 'orange', linestyle=':', label='GPU Memory Cached', linewidth=2)
            ax2.axhline(y=self.gpu_total_memory, color='r', linestyle='--', alpha=0.7, label='GPU Total Memory')
            ax2.set_xlabel('Training Step')
            ax2.set_ylabel('Memory (MB)')
            ax2.set_title('GPU Memory Usage')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'GPU Not Available', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('GPU Memory Usage')
        
        plt.tight_layout()
        plot_file = output_dir / "memory_usage_plots.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Memory usage plots saved to: {plot_file}")

class PerformanceBenchmark:
    """성능 벤치마킹 도구"""
    
    def __init__(self):
        """벤치마크 초기화"""
        self.results = []
        
    def benchmark_model_loading(self, model_path: str) -> Dict[str, float]:
        """모델 로딩 속도 벤치마크"""
        from core.inference import InferenceEngine
        
        start_time = time.time()
        engine = InferenceEngine(model_path)
        loading_time = time.time() - start_time
        
        # 모델 크기 계산
        model_size_mb = sum(p.numel() * p.element_size() for p in engine.model.parameters()) / 1024**2
        
        result = {
            'model_path': model_path,
            'loading_time_sec': loading_time,
            'model_size_mb': model_size_mb,
            'loading_speed_mb_per_sec': model_size_mb / loading_time if loading_time > 0 else 0
        }
        
        self.results.append(result)
        return result
    
    def benchmark_inference_speed(self, 
                                 model_path: str,
                                 test_dialogues: List[str],
                                 batch_sizes: List[int] = [1, 4, 8, 16]) -> Dict[str, Any]:
        """추론 속도 벤치마크"""
        from core.inference import InferenceEngine
        
        engine = InferenceEngine(model_path)
        
        benchmark_results = {
            'model_path': model_path,
            'batch_results': []
        }
        
        for batch_size in batch_sizes:
            # 테스트 데이터 준비
            test_data = test_dialogues[:batch_size * 10]  # 10 배치 테스트
            
            # 워밍업
            if test_data:
                engine.predict_batch(test_data[:batch_size], show_progress=False)
            
            # 실제 벤치마크
            start_time = time.time()
            predictions = engine.predict_batch(test_data, batch_size=batch_size, show_progress=False)
            end_time = time.time()
            
            total_time = end_time - start_time
            samples_per_second = len(test_data) / total_time if total_time > 0 else 0
            
            batch_result = {
                'batch_size': batch_size,
                'total_samples': len(test_data),
                'total_time_sec': total_time,
                'samples_per_second': samples_per_second,
                'avg_time_per_sample': total_time / len(test_data) if test_data else 0
            }
            
            benchmark_results['batch_results'].append(batch_result)
            print(f"Batch size {batch_size}: {samples_per_second:.2f} samples/sec")
        
        return benchmark_results
    
    def find_optimal_batch_size(self, 
                               model_path: str,
                               test_dialogues: List[str],
                               max_batch_size: int = 32) -> int:
        """최적 배치 크기 찾기"""
        profiler = MemoryProfiler()
        
        # 점진적으로 배치 크기 증가
        optimal_batch_size = 1
        
        for batch_size in [1, 2, 4, 8, 16, 32, 64]:
            if batch_size > max_batch_size:
                break
                
            try:
                # 메모리 사용량 체크
                profiler.take_snapshot(0, f"batch_size_{batch_size}_start")
                
                # 추론 테스트
                benchmark = self.benchmark_inference_speed(
                    model_path, test_dialogues, [batch_size]
                )
                
                profiler.take_snapshot(1, f"batch_size_{batch_size}_end")
                
                # GPU 메모리 사용률 체크
                if profiler.gpu_available:
                    memory_usage_pct = (torch.cuda.memory_allocated(0) / 
                                      torch.cuda.get_device_properties(0).total_memory) * 100
                    
                    if memory_usage_pct > 85:  # 85% 이상 사용 시 중단
                        print(f"Memory usage too high ({memory_usage_pct:.1f}%) for batch size {batch_size}")
                        break
                
                optimal_batch_size = batch_size
                print(f"Batch size {batch_size} is feasible")
                
            except Exception as e:
                print(f"Batch size {batch_size} failed: {e}")
                break
        
        print(f"Recommended optimal batch size: {optimal_batch_size}")
        return optimal_batch_size

# 편의 함수들
def quick_memory_profile(trainer, duration_steps: int = 100) -> Dict[str, Any]:
    """빠른 메모리 프로파일링"""
    profiler = MemoryProfiler()
    profiler.monitor_training_loop(trainer, duration_steps)
    return profiler.analyze_memory_patterns()

def benchmark_model_performance(model_path: str, 
                               test_data: List[str],
                               output_dir: Optional[str] = None) -> Dict[str, Any]:
    """모델 성능 종합 벤치마크"""
    benchmark = PerformanceBenchmark()
    
    # 모델 로딩 벤치마크
    loading_result = benchmark.benchmark_model_loading(model_path)
    
    # 추론 속도 벤치마크
    inference_result = benchmark.benchmark_inference_speed(model_path, test_data)
    
    # 최적 배치 크기 찾기
    optimal_batch = benchmark.find_optimal_batch_size(model_path, test_data)
    
    # 결과 정리
    comprehensive_result = {
        'model_loading': loading_result,
        'inference_speed': inference_result,
        'optimal_batch_size': optimal_batch,
        'recommendations': []
    }
    
    # 권장사항 생성
    if loading_result['loading_time_sec'] > 30:
        comprehensive_result['recommendations'].append(
            "Model loading is slow. Consider model quantization or using smaller models for development."
        )
    
    best_throughput = max(
        batch['samples_per_second'] for batch in inference_result['batch_results']
    )
    if best_throughput < 1.0:
        comprehensive_result['recommendations'].append(
            "Low inference throughput. Consider using GPU acceleration or optimizing model architecture."
        )
    
    # 결과 저장
    if output_dir:
        output_dir = PathManager.resolve_path(output_dir)
        PathManager.ensure_dir(output_dir)
        
        import json
        result_file = output_dir / "performance_benchmark.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_result, f, ensure_ascii=False, indent=2)
        
        print(f"Benchmark results saved to: {result_file}")
    
    return comprehensive_result
```

### 구현 우선순위: 🟡 선택적
### 예상 작업 시간: 10-14시간
### 권장 시기: 성능 최적화가 필요한 시점

---

## 🟡 4. 고급 실험 기능 (Research Enhancement)

### 4.1 Ablation Study 자동화

#### 가치 제안
- 모델 구성 요소별 기여도 분석
- 최적 설정 조합 발견
- 논문 작성용 실험 데이터 제공

#### 구현 방안
```python
# config/sweep/ablation_study_sweep.yaml 완성
name: "Ablation Study - Component Analysis"
method: "grid"  # 모든 조합 테스트
metric:
  name: "rouge_combined_f1"
  goal: "maximize"

parameters:
  # 전처리 구성요소
  use_special_tokens:
    values: [true, false]
  
  text_cleaning_level:
    values: ["basic", "advanced", "none"]
  
  # 모델 구성요소
  use_attention_dropout:
    values: [true, false]
  
  attention_dropout_rate:
    values: [0.0, 0.1, 0.3]
  
  # 학습 전략
  use_warmup:
    values: [true, false]
  
  use_weight_decay:
    values: [true, false]
  
  # 생성 전략
  generation_strategy:
    values: ["beam_search", "nucleus_sampling", "top_k"]
  
  length_penalty:
    values: [0.8, 1.0, 1.2, 1.5]

# 실험 제약 조건
constraints:
  - name: sampling_constraint
    condition: generation_strategy != "beam_search"
    then:
      num_beams: 1
      do_sample: true
```

### 4.2 자동 A/B 테스트 시스템

#### 구현 방안
```python
# code/experiments/ab_testing.py (신규 생성)
from typing import List, Dict, Any, Tuple
import numpy as np
from scipy import stats
import pandas as pd

class ABTestFramework:
    """A/B 테스트 프레임워크"""
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self.experiments = []
    
    def run_ab_test(self, 
                   model_a_results: List[float],
                   model_b_results: List[float],
                   metric_name: str = "rouge_combined") -> Dict[str, Any]:
        """A/B 테스트 실행"""
        
        # 기본 통계
        stats_a = {
            'mean': np.mean(model_a_results),
            'std': np.std(model_a_results),
            'n': len(model_a_results)
        }
        
        stats_b = {
            'mean': np.mean(model_b_results),
            'std': np.std(model_b_results),
            'n': len(model_b_results)
        }
        
        # 통계적 유의성 검정 (t-test)
        t_stat, p_value = stats.ttest_ind(model_a_results, model_b_results)
        
        # 효과 크기 (Cohen's d)
        pooled_std = np.sqrt(((stats_a['n']-1) * stats_a['std']**2 + 
                             (stats_b['n']-1) * stats_b['std']**2) / 
                            (stats_a['n'] + stats_b['n'] - 2))
        cohens_d = (stats_b['mean'] - stats_a['mean']) / pooled_std
        
        # 신뢰구간
        se_diff = pooled_std * np.sqrt(1/stats_a['n'] + 1/stats_b['n'])
        mean_diff = stats_b['mean'] - stats_a['mean']
        ci_lower = mean_diff - 1.96 * se_diff
        ci_upper = mean_diff + 1.96 * se_diff
        
        result = {
            'metric_name': metric_name,
            'model_a_stats': stats_a,
            'model_b_stats': stats_b,
            'difference': mean_diff,
            'difference_pct': (mean_diff / stats_a['mean']) * 100,
            'p_value': p_value,
            'is_significant': p_value < self.significance_level,
            'cohens_d': cohens_d,
            'effect_size': self._interpret_effect_size(cohens_d),
            'confidence_interval': (ci_lower, ci_upper),
            'winner': 'Model B' if mean_diff > 0 and p_value < self.significance_level else 
                     'Model A' if mean_diff < 0 and p_value < self.significance_level else 'No significant difference'
        }
        
        return result
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Cohen's d 해석"""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
```

### 구현 우선순위: 🟡 선택적
### 예상 작업 시간: 8-12시간
### 권장 시기: 연구 목적이나 논문 작성이 필요할 때

---

## 📋 선택적 개발 사항 우선순위

| 기능 | 비즈니스 가치 | 기술적 난이도 | 구현 시간 | 권장 시기 |
|------|---------------|---------------|-----------|-----------|
| 앙상블 시스템 | 🔥🔥🔥 높음 | 🔧🔧🔧 높음 | 16-20h | 대회 후반 |
| 자동 하이퍼파라미터 제안 | 🔥🔥 중간 | 🔧🔧 중간 | 12-16h | 프로젝트 중반 |
| 성능 프로파일링 | 🔥🔥 중간 | 🔧🔧 중간 | 10-14h | 최적화 필요시 |
| Ablation Study | 🔥 낮음 | 🔧 낮음 | 8-12h | 연구 목적 |

---

## 🎯 구현 전략

### Phase 1: 고효과 기능 우선 (Week 3-4)
1. **자동 하이퍼파라미터 제안** - 실험 효율성 극대화
2. **성능 프로파일링** - 리소스 최적화

### Phase 2: 성능 극대화 (Week 5-6)
1. **앙상블 시스템** - 최종 성능 향상
2. **고급 최적화 도구**

### Phase 3: 연구 확장 (선택적)
1. **Ablation Study 자동화**
2. **A/B 테스트 프레임워크**

---

## ⚠️ 주의사항

1. **리소스 고려**: 선택적 기능들은 추가 계산 리소스 필요
2. **복잡성 관리**: 핵심 기능 완성 후 점진적 추가
3. **유지보수성**: 선택적 기능도 동일한 코딩 표준 적용
4. **문서화**: 고급 기능일수록 상세한 문서화 필요

---

## 📊 ROI 분석

### 높은 ROI
- **자동 하이퍼파라미터 제안**: 적은 노력으로 큰 효율성 향상
- **성능 프로파일링**: 메모리 최적화로 더 큰 모델 사용 가능

### 중간 ROI  
- **앙상블 시스템**: 높은 성능 향상이지만 구현 복잡

### 낮은 ROI
- **Ablation Study**: 연구 목적이 아니면 투자 대비 효과 제한

---

## 🎯 최종 권장사항

1. **필수 구현 완료 후 시작**: Priority 1 사항들이 완전히 구현된 후
2. **단계적 접근**: 한 번에 모든 선택적 기능을 구현하지 말고 필요에 따라
3. **성능 측정**: 각 기능 추가 후 실제 성능 향상 측정
4. **팀 역량 고려**: 팀의 기술 수준과 사용 가능한 시간 고려

기본 기능이 안정적으로 작동한 후, 실제 필요성에 따라 선택적으로 구현하는 것을 권장합니다.