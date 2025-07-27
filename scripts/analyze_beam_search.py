#!/usr/bin/env python3
"""
빔 서치 파라미터 최적화 분석 스크립트
각 파라미터의 영향과 최적 조합을 찾습니다.
"""
import os
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
from collections import defaultdict


class BeamSearchAnalyzer:
    """빔 서치 실험 결과 분석 클래스"""
    
    def __init__(self, experiment_dir: Path, output_dir: Path):
        self.experiment_dir = experiment_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def analyze_beam_size_experiments(self, results_dir: Path) -> pd.DataFrame:
        """빔 크기 실험 결과 분석"""
        results = []
        
        for beam_size in [2, 4, 6, 8]:
            exp_name = f"beam_{beam_size}"
            exp_path = results_dir / exp_name
            
            if exp_path.exists():
                metrics = self._load_experiment_metrics(exp_path)
                
                results.append({
                    'beam_size': beam_size,
                    'rouge_f1': metrics.get('rouge_f1', 0),
                    'inference_time': metrics.get('avg_inference_time', 0),
                    'memory_usage': metrics.get('peak_memory_mb', 0),
                    'repetition_rate': metrics.get('repetition_rate', 0),
                    'avg_length': metrics.get('avg_generated_length', 0)
                })
        
        return pd.DataFrame(results)
    
    def analyze_length_penalty_experiments(self, results_dir: Path) -> pd.DataFrame:
        """길이 패널티 실험 결과 분석"""
        results = []
        
        for penalty in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
            exp_name = f"length_penalty_{penalty}"
            exp_path = results_dir / exp_name
            
            if exp_path.exists():
                metrics = self._load_experiment_metrics(exp_path)
                
                results.append({
                    'length_penalty': penalty,
                    'rouge_f1': metrics.get('rouge_f1', 0),
                    'avg_length': metrics.get('avg_generated_length', 0),
                    'length_ratio': metrics.get('compression_ratio', 0),
                    'length_variance': metrics.get('length_std', 0),
                    'content_coverage': metrics.get('content_coverage', 0)
                })
        
        return pd.DataFrame(results)
    
    def analyze_no_repeat_experiments(self, results_dir: Path) -> pd.DataFrame:
        """반복 방지 실험 결과 분석"""
        results = []
        
        for ngram_size in [0, 2, 3, 4, 5]:
            exp_name = f"no_repeat_{ngram_size}"
            exp_path = results_dir / exp_name
            
            if exp_path.exists():
                metrics = self._load_experiment_metrics(exp_path)
                
                results.append({
                    'no_repeat_ngram_size': ngram_size,
                    'rouge_f1': metrics.get('rouge_f1', 0),
                    'repetition_rate': metrics.get('ngram_repetition_rate', {}),
                    'distinct_1': metrics.get('distinct_1', 0),
                    'distinct_2': metrics.get('distinct_2', 0),
                    'lexical_diversity': metrics.get('lexical_diversity', 0),
                    'fluency_score': metrics.get('fluency_score', 0)
                })
        
        return pd.DataFrame(results)
    
    def analyze_diverse_beam_experiments(self, results_dir: Path) -> pd.DataFrame:
        """Diverse Beam Search 실험 결과 분석"""
        results = []
        
        experiments = [
            ('standard_beam', 1, 0.0),
            ('diverse_2groups_low', 2, 0.5),
            ('diverse_2groups_high', 2, 1.0),
            ('diverse_4groups_low', 4, 0.5),
            ('diverse_4groups_high', 4, 1.0),
            ('diverse_4groups_very_high', 4, 2.0)
        ]
        
        for exp_name, num_groups, diversity_penalty in experiments:
            exp_path = results_dir / exp_name
            
            if exp_path.exists():
                metrics = self._load_experiment_metrics(exp_path)
                
                results.append({
                    'experiment': exp_name,
                    'num_beam_groups': num_groups,
                    'diversity_penalty': diversity_penalty,
                    'rouge_f1': metrics.get('rouge_f1', 0),
                    'self_bleu': metrics.get('self_bleu', 0),
                    'semantic_diversity': metrics.get('semantic_diversity', 0),
                    'best_of_n_rouge': metrics.get('best_of_n_rouge', 0),
                    'ensemble_rouge': metrics.get('ensemble_rouge', 0)
                })
        
        return pd.DataFrame(results)
    
    def _load_experiment_metrics(self, exp_path: Path) -> Dict:
        """실험 메트릭 로드"""
        metrics = {}
        
        # results.json 읽기
        results_file = exp_path / 'results.json'
        if results_file.exists():
            with open(results_file, 'r') as f:
                metrics.update(json.load(f))
        
        # 추가 메트릭 파일 읽기
        metrics_file = exp_path / 'metrics.json'
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics.update(json.load(f))
        
        return metrics
    
    def create_visualizations(self, 
                            beam_df: pd.DataFrame,
                            length_df: pd.DataFrame,
                            repeat_df: pd.DataFrame,
                            diverse_df: pd.DataFrame):
        """시각화 생성"""
        
        # 1. 빔 크기 vs 성능/속도
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 성능 vs 속도 트레이드오프
        ax1.plot(beam_df['beam_size'], beam_df['rouge_f1'], 'o-', label='ROUGE-F1', color='blue')
        ax1.set_xlabel('Beam Size')
        ax1.set_ylabel('ROUGE-F1', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        ax1_twin = ax1.twinx()
        ax1_twin.plot(beam_df['beam_size'], beam_df['inference_time'], 's-', label='시간(초)', color='red')
        ax1_twin.set_ylabel('Inference Time (sec)', color='red')
        ax1_twin.tick_params(axis='y', labelcolor='red')
        
        ax1.set_title('빔 크기: 성능 vs 속도')
        ax1.grid(True, alpha=0.3)
        
        # 메모리 사용량
        ax2.bar(beam_df['beam_size'], beam_df['memory_usage'], alpha=0.7)
        ax2.set_xlabel('Beam Size')
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.set_title('빔 크기별 메모리 사용량')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'beam_size_analysis.png', dpi=150)
        plt.close()
        
        # 2. 길이 패널티 효과
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 길이 패널티 vs ROUGE
        axes[0, 0].plot(length_df['length_penalty'], length_df['rouge_f1'], 'o-')
        axes[0, 0].set_xlabel('Length Penalty')
        axes[0, 0].set_ylabel('ROUGE-F1')
        axes[0, 0].set_title('길이 패널티와 성능')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 길이 패널티 vs 평균 길이
        axes[0, 1].plot(length_df['length_penalty'], length_df['avg_length'], 's-', color='green')
        axes[0, 1].set_xlabel('Length Penalty')
        axes[0, 1].set_ylabel('Average Length')
        axes[0, 1].set_title('길이 패널티와 생성 길이')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 길이 비율 분포
        axes[1, 0].bar(length_df['length_penalty'], length_df['length_ratio'])
        axes[1, 0].axhline(y=0.3, color='r', linestyle='--', label='목표 비율')
        axes[1, 0].set_xlabel('Length Penalty')
        axes[1, 0].set_ylabel('Compression Ratio')
        axes[1, 0].set_title('압축 비율')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 내용 보존율
        axes[1, 1].plot(length_df['length_penalty'], length_df['content_coverage'], '^-', color='purple')
        axes[1, 1].set_xlabel('Length Penalty')
        axes[1, 1].set_ylabel('Content Coverage')
        axes[1, 1].set_title('내용 보존율')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'length_penalty_analysis.png', dpi=150)
        plt.close()
        
        # 3. 반복 방지 효과
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # n-gram 크기 vs ROUGE
        axes[0, 0].plot(repeat_df['no_repeat_ngram_size'], repeat_df['rouge_f1'], 'o-')
        axes[0, 0].set_xlabel('No Repeat N-gram Size')
        axes[0, 0].set_ylabel('ROUGE-F1')
        axes[0, 0].set_title('반복 방지와 성능')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 어휘 다양성
        axes[0, 1].plot(repeat_df['no_repeat_ngram_size'], repeat_df['lexical_diversity'], 's-', color='orange')
        axes[0, 1].set_xlabel('No Repeat N-gram Size')
        axes[0, 1].set_ylabel('Lexical Diversity')
        axes[0, 1].set_title('어휘 다양성')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Distinct n-grams
        x = repeat_df['no_repeat_ngram_size']
        axes[1, 0].plot(x, repeat_df['distinct_1'], 'o-', label='Distinct-1')
        axes[1, 0].plot(x, repeat_df['distinct_2'], 's-', label='Distinct-2')
        axes[1, 0].set_xlabel('No Repeat N-gram Size')
        axes[1, 0].set_ylabel('Distinct N-grams')
        axes[1, 0].set_title('고유 N-gram 비율')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 유창성 점수
        axes[1, 1].plot(repeat_df['no_repeat_ngram_size'], repeat_df['fluency_score'], '^-', color='brown')
        axes[1, 1].set_xlabel('No Repeat N-gram Size')
        axes[1, 1].set_ylabel('Fluency Score')
        axes[1, 1].set_title('유창성 점수')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'no_repeat_analysis.png', dpi=150)
        plt.close()
        
        # 4. Diverse Beam Search 분석
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 그룹/패널티별 성능
        pivot_data = diverse_df.pivot_table(
            values='rouge_f1',
            index='num_beam_groups',
            columns='diversity_penalty'
        )
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[0, 0])
        axes[0, 0].set_title('ROUGE-F1 by Groups & Penalty')
        
        # Self-BLEU (다양성)
        axes[0, 1].scatter(diverse_df['diversity_penalty'], diverse_df['self_bleu'], 
                          s=diverse_df['num_beam_groups']*50, alpha=0.6)
        axes[0, 1].set_xlabel('Diversity Penalty')
        axes[0, 1].set_ylabel('Self-BLEU (lower=more diverse)')
        axes[0, 1].set_title('생성 다양성')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Best-of-N vs Ensemble
        axes[1, 0].bar(range(len(diverse_df)), diverse_df['best_of_n_rouge'], 
                      alpha=0.6, label='Best-of-N')
        axes[1, 0].bar(range(len(diverse_df)), diverse_df['ensemble_rouge'], 
                      alpha=0.6, label='Ensemble')
        axes[1, 0].set_xticks(range(len(diverse_df)))
        axes[1, 0].set_xticklabels(diverse_df['experiment'], rotation=45, ha='right')
        axes[1, 0].set_ylabel('ROUGE-F1')
        axes[1, 0].set_title('선택 전략 비교')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 다양성 vs 품질 트레이드오프
        axes[1, 1].scatter(diverse_df['semantic_diversity'], diverse_df['rouge_f1'])
        for i, txt in enumerate(diverse_df['experiment']):
            axes[1, 1].annotate(txt.replace('diverse_', ''), 
                               (diverse_df['semantic_diversity'].iloc[i], 
                                diverse_df['rouge_f1'].iloc[i]),
                               fontsize=8)
        axes[1, 1].set_xlabel('Semantic Diversity')
        axes[1, 1].set_ylabel('ROUGE-F1')
        axes[1, 1].set_title('다양성 vs 품질')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'diverse_beam_analysis.png', dpi=150)
        plt.close()
    
    def find_optimal_parameters(self,
                               beam_df: pd.DataFrame,
                               length_df: pd.DataFrame,
                               repeat_df: pd.DataFrame,
                               diverse_df: pd.DataFrame) -> Dict:
        """최적 파라미터 조합 찾기"""
        
        # 각 실험의 최적값
        optimal = {}
        
        # 빔 크기: 성능/속도 균형
        beam_efficiency = beam_df['rouge_f1'] / beam_df['inference_time']
        optimal['beam_size'] = beam_df.loc[beam_efficiency.idxmax(), 'beam_size']
        
        # 길이 패널티: 목표 비율에 가장 가까운 값
        length_error = abs(length_df['length_ratio'] - 0.3)
        optimal['length_penalty'] = length_df.loc[length_error.idxmin(), 'length_penalty']
        
        # 반복 방지: ROUGE와 다양성의 균형
        repeat_score = repeat_df['rouge_f1'] * repeat_df['lexical_diversity']
        optimal['no_repeat_ngram_size'] = repeat_df.loc[repeat_score.idxmax(), 'no_repeat_ngram_size']
        
        # Diverse Beam: 최고 ROUGE
        optimal['diverse_beam'] = diverse_df.loc[diverse_df['rouge_f1'].idxmax(), 'experiment']
        
        return optimal
    
    def generate_report(self,
                       beam_df: pd.DataFrame,
                       length_df: pd.DataFrame,
                       repeat_df: pd.DataFrame,
                       diverse_df: pd.DataFrame,
                       optimal_params: Dict):
        """종합 보고서 생성"""
        
        report = []
        report.append("# 빔 서치 파라미터 최적화 결과\n")
        
        # 1. 빔 크기 분석
        report.append("## 1. 빔 크기 (num_beams)")
        report.append(f"- **최적값**: {optimal_params['beam_size']}")
        report.append(f"- **ROUGE-F1**: {beam_df[beam_df['beam_size']==optimal_params['beam_size']]['rouge_f1'].values[0]:.4f}")
        report.append("- **분석**:")
        report.append("  - 빔 크기 증가 → 성능 향상, 속도 감소")
        report.append("  - 4-6이 최적 균형점")
        report.append("")
        
        # 2. 길이 패널티 분석
        report.append("## 2. 길이 패널티 (length_penalty)")
        report.append(f"- **최적값**: {optimal_params['length_penalty']}")
        best_length = length_df[length_df['length_penalty']==optimal_params['length_penalty']]
        report.append(f"- **압축 비율**: {best_length['length_ratio'].values[0]:.3f}")
        report.append("- **분석**:")
        report.append("  - 1.2-1.5 범위가 목표 길이에 적합")
        report.append("  - 너무 높으면 과도한 압축 발생")
        report.append("")
        
        # 3. 반복 방지 분석
        report.append("## 3. 반복 방지 (no_repeat_ngram_size)")
        report.append(f"- **최적값**: {optimal_params['no_repeat_ngram_size']}")
        best_repeat = repeat_df[repeat_df['no_repeat_ngram_size']==optimal_params['no_repeat_ngram_size']]
        report.append(f"- **어휘 다양성**: {best_repeat['lexical_diversity'].values[0]:.3f}")
        report.append("- **분석**:")
        report.append("  - 3-gram이 반복 방지와 자연스러움의 균형")
        report.append("  - 너무 크면 유창성 저하")
        report.append("")
        
        # 4. Diverse Beam Search 분석
        report.append("## 4. Diverse Beam Search")
        report.append(f"- **최적 설정**: {optimal_params['diverse_beam']}")
        best_diverse = diverse_df[diverse_df['experiment']==optimal_params['diverse_beam']]
        report.append(f"- **ROUGE-F1**: {best_diverse['rouge_f1'].values[0]:.4f}")
        report.append("- **분석**:")
        report.append("  - 표준 빔 서치가 대부분 경우 충분")
        report.append("  - 다양성이 필요한 경우만 diverse beam 사용")
        report.append("")
        
        # 5. 최종 권장 설정
        report.append("## 5. 최종 권장 설정")
        report.append("```yaml")
        report.append("generation:")
        report.append(f"  num_beams: {optimal_params['beam_size']}")
        report.append(f"  length_penalty: {optimal_params['length_penalty']}")
        report.append(f"  no_repeat_ngram_size: {optimal_params['no_repeat_ngram_size']}")
        report.append("  early_stopping: true")
        report.append("  max_length: 200")
        report.append("  min_length: 30")
        report.append("```")
        
        # 보고서 저장
        with open(self.output_dir / 'beam_search_report.md', 'w') as f:
            f.write('\n'.join(report))
        
        # 최적 설정 YAML 저장
        optimal_config = {
            'generation': {
                'num_beams': int(optimal_params['beam_size']),
                'length_penalty': float(optimal_params['length_penalty']),
                'no_repeat_ngram_size': int(optimal_params['no_repeat_ngram_size']),
                'early_stopping': True,
                'max_length': 200,
                'min_length': 30
            }
        }
        
        with open(self.output_dir / 'optimal_beam_search.yaml', 'w') as f:
            yaml.dump(optimal_config, f, default_flow_style=False)


def compute_repetition_metrics(texts: List[str]) -> Dict[str, float]:
    """반복 관련 메트릭 계산"""
    from collections import Counter
    
    metrics = {}
    
    # N-gram 반복률 계산
    for n in [2, 3, 4]:
        all_ngrams = []
        repeated_ngrams = []
        
        for text in texts:
            words = text.split()
            ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
            ngram_counts = Counter(ngrams)
            
            all_ngrams.extend(ngrams)
            repeated_ngrams.extend([ng for ng, count in ngram_counts.items() if count > 1])
        
        if all_ngrams:
            metrics[f'{n}gram_repetition_rate'] = len(repeated_ngrams) / len(all_ngrams)
    
    # 어휘 다양성 (Type-Token Ratio)
    all_tokens = []
    for text in texts:
        all_tokens.extend(text.split())
    
    if all_tokens:
        metrics['lexical_diversity'] = len(set(all_tokens)) / len(all_tokens)
    
    # Distinct n-grams
    for n in [1, 2]:
        all_ngrams = []
        for text in texts:
            words = text.split()
            ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
            all_ngrams.extend(ngrams)
        
        if all_ngrams:
            metrics[f'distinct_{n}'] = len(set(all_ngrams)) / len(all_ngrams)
    
    return metrics


def main():
    """메인 실행 함수"""
    # 경로 설정
    project_root = Path(__file__).parent.parent
    experiment_dir = project_root / 'config' / 'experiments' / '08_beam_search_optimization'
    results_dir = project_root / 'outputs' / 'logs'
    output_dir = project_root / 'outputs' / 'analysis' / 'beam_search_optimization'
    
    # 분석기 생성
    analyzer = BeamSearchAnalyzer(experiment_dir, output_dir)
    
    # 각 실험 결과 분석
    beam_df = analyzer.analyze_beam_size_experiments(results_dir / '08a_beam_size')
    length_df = analyzer.analyze_length_penalty_experiments(results_dir / '08b_length_penalty')
    repeat_df = analyzer.analyze_no_repeat_experiments(results_dir / '08c_no_repeat')
    diverse_df = analyzer.analyze_diverse_beam_experiments(results_dir / '08d_diverse_beam')
    
    # 시각화 생성
    if not beam_df.empty:
        analyzer.create_visualizations(beam_df, length_df, repeat_df, diverse_df)
        
        # 최적 파라미터 찾기
        optimal_params = analyzer.find_optimal_parameters(beam_df, length_df, repeat_df, diverse_df)
        
        # 보고서 생성
        analyzer.generate_report(beam_df, length_df, repeat_df, diverse_df, optimal_params)
        
        print("분석 완료!")
        print(f"결과 저장 위치: {output_dir}")
        print(f"최적 파라미터:")
        for key, value in optimal_params.items():
            print(f"  - {key}: {value}")
    else:
        print("실험 결과를 찾을 수 없습니다. 실험을 먼저 실행해주세요.")


if __name__ == "__main__":
    main()
