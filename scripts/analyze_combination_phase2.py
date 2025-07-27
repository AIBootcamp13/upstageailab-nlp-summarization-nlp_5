#!/usr/bin/env python3
"""
조합 실험 2차 분석 스크립트
고급 기능 통합의 효과와 최종 성능을 분석합니다.
"""
import os
import json
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

def load_experiment_results(log_dir: str, phase1_prefix: str = "06", phase2_prefix: str = "10") -> Dict:
    """1차 및 2차 실험 결과 로드"""
    results = {}
    
    # 베이스라인 결과
    baseline_dir = Path(log_dir) / "00_baseline_reproduction"
    if baseline_dir.exists():
        results['baseline'] = parse_experiment_logs(baseline_dir)
    
    # 1차 조합 실험 최고 결과
    phase1_dirs = [d for d in Path(log_dir).iterdir() 
                   if d.is_dir() and d.name.startswith(phase1_prefix)]
    best_phase1 = None
    best_phase1_score = 0
    
    for exp_dir in phase1_dirs:
        metrics = parse_experiment_logs(exp_dir)
        if metrics['best_rouge_f1'] > best_phase1_score:
            best_phase1_score = metrics['best_rouge_f1']
            best_phase1 = exp_dir.name
            results['best_phase1'] = metrics
    
    # 개별 고급 기능 결과
    advanced_features = {
        'token_weighting': '07_token_weighting',
        'beam_search': '08_beam_search_optimization/08d_diverse_beam',
        'backtranslation': '09_backtranslation'
    }
    
    for feat_name, feat_dir in advanced_features.items():
        exp_dir = Path(log_dir) / feat_dir
        if exp_dir.exists():
            results[feat_name] = parse_experiment_logs(exp_dir)
    
    # 2차 조합 실험 결과
    phase2_dirs = [d for d in Path(log_dir).iterdir() 
                   if d.is_dir() and d.name.startswith(phase2_prefix)]
    
    for exp_dir in phase2_dirs:
        results[exp_dir.name] = parse_experiment_logs(exp_dir)
    
    return results

def parse_experiment_logs(exp_dir: Path) -> Dict:
    """실험 로그 상세 파싱"""
    metrics = {
        'best_rouge_f1': 0,
        'best_rouge1': 0,
        'best_rouge2': 0,
        'best_rougeL': 0,
        'final_rouge_f1': 0,
        'training_time': 0,
        'inference_time': 0,
        'convergence_epoch': 0,
        'memory_peak': 0,
        'special_token_accuracy': 0,
        'pii_recall': 0,
        'augmentation_quality': 0
    }
    
    # WandB 로그 분석
    wandb_dir = exp_dir / 'wandb'
    if wandb_dir.exists():
        latest_run = sorted(wandb_dir.glob('run-*'))[-1] if list(wandb_dir.glob('run-*')) else None
        if latest_run:
            history_file = latest_run / 'files' / 'wandb-history.jsonl'
            if history_file.exists():
                rouge_scores = {'f1': [], 'rouge1': [], 'rouge2': [], 'rougeL': []}
                timestamps = []
                memory_usage = []
                
                with open(history_file, 'r') as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            
                            # ROUGE 점수
                            if 'eval_rouge_combined_f1' in data:
                                rouge_scores['f1'].append(data['eval_rouge_combined_f1'])
                            if 'eval_rouge1' in data:
                                rouge_scores['rouge1'].append(data['eval_rouge1'])
                            if 'eval_rouge2' in data:
                                rouge_scores['rouge2'].append(data['eval_rouge2'])
                            if 'eval_rougeL' in data:
                                rouge_scores['rougeL'].append(data['eval_rougeL'])
                            
                            # 특수 토큰 관련
                            if 'special_token_accuracy' in data:
                                metrics['special_token_accuracy'] = data['special_token_accuracy']
                            if 'pii_recall' in data:
                                metrics['pii_recall'] = data['pii_recall']
                            
                            # 시간 및 메모리
                            if '_timestamp' in data:
                                timestamps.append(data['_timestamp'])
                            if 'memory_usage_gb' in data:
                                memory_usage.append(data['memory_usage_gb'])
                        except:
                            continue
                
                # 최고 점수 계산
                if rouge_scores['f1']:
                    metrics['best_rouge_f1'] = max(rouge_scores['f1'])
                    metrics['final_rouge_f1'] = rouge_scores['f1'][-1]
                    metrics['convergence_epoch'] = rouge_scores['f1'].index(max(rouge_scores['f1'])) + 1
                
                if rouge_scores['rouge1']:
                    metrics['best_rouge1'] = max(rouge_scores['rouge1'])
                if rouge_scores['rouge2']:
                    metrics['best_rouge2'] = max(rouge_scores['rouge2'])
                if rouge_scores['rougeL']:
                    metrics['best_rougeL'] = max(rouge_scores['rougeL'])
                
                if timestamps:
                    metrics['training_time'] = (timestamps[-1] - timestamps[0]) / 3600  # 시간 단위
                
                if memory_usage:
                    metrics['memory_peak'] = max(memory_usage)
    
    # 결과 파일에서 추가 정보
    results_file = exp_dir / 'results.json'
    if results_file.exists():
        with open(results_file, 'r') as f:
            saved_results = json.load(f)
            metrics.update(saved_results.get('metrics', {}))
    
    return metrics

def analyze_phase2_improvements(results: Dict) -> pd.DataFrame:
    """2차 실험의 개선 효과 분석"""
    analysis_data = []
    
    baseline_score = results.get('baseline', {}).get('best_rouge_f1', 0.4712)
    phase1_score = results.get('best_phase1', {}).get('best_rouge_f1', baseline_score)
    
    # 2차 실험들 분석
    phase2_experiments = {
        '10a_phase1_plus_token_weight': 'Phase1 + Token Weight',
        '10b_phase1_plus_backtrans': 'Phase1 + BackTranslation',
        '10c_all_optimizations': 'All Optimizations'
    }
    
    for exp_name, display_name in phase2_experiments.items():
        if exp_name in results:
            metrics = results[exp_name]
            
            improvement_from_baseline = metrics['best_rouge_f1'] - baseline_score
            improvement_from_phase1 = metrics['best_rouge_f1'] - phase1_score
            
            analysis_data.append({
                'Experiment': display_name,
                'ROUGE-F1': metrics['best_rouge_f1'],
                'ROUGE-1': metrics['best_rouge1'],
                'ROUGE-2': metrics['best_rouge2'],
                'ROUGE-L': metrics['best_rougeL'],
                'Improvement from Baseline': improvement_from_baseline,
                'Improvement from Phase1': improvement_from_phase1,
                'Training Time (h)': metrics['training_time'],
                'Memory Peak (GB)': metrics['memory_peak'],
                'Convergence Epoch': metrics['convergence_epoch'],
                'Special Token Acc': metrics.get('special_token_accuracy', 0),
                'PII Recall': metrics.get('pii_recall', 0)
            })
    
    return pd.DataFrame(analysis_data)

def plot_performance_comparison(results: Dict, output_dir: str):
    """성능 비교 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. ROUGE 점수 비교
    ax1 = axes[0, 0]
    experiments = ['baseline', 'best_phase1', 'token_weighting', 
                  'backtranslation', '10a_phase1_plus_token_weight',
                  '10b_phase1_plus_backtrans', '10c_all_optimizations']
    
    rouge_scores = []
    exp_names = []
    
    for exp in experiments:
        if exp in results:
            rouge_scores.append([
                results[exp].get('best_rouge1', 0),
                results[exp].get('best_rouge2', 0),
                results[exp].get('best_rougeL', 0),
                results[exp].get('best_rouge_f1', 0)
            ])
            exp_names.append(exp.replace('_', ' ').title())
    
    rouge_df = pd.DataFrame(rouge_scores, 
                           columns=['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'ROUGE-F1'],
                           index=exp_names)
    
    rouge_df.plot(kind='bar', ax=ax1, rot=45)
    ax1.set_title('ROUGE Score Comparison')
    ax1.set_ylabel('Score')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 2. 학습 효율성 (시간 대비 성능)
    ax2 = axes[0, 1]
    efficiency_data = []
    
    for exp in ['10a_phase1_plus_token_weight', '10b_phase1_plus_backtrans', '10c_all_optimizations']:
        if exp in results:
            time = results[exp].get('training_time', 1)
            score = results[exp].get('best_rouge_f1', 0)
            efficiency = score / time if time > 0 else 0
            efficiency_data.append({
                'Experiment': exp.split('_')[-1],
                'Training Time': time,
                'ROUGE-F1': score,
                'Efficiency': efficiency
            })
    
    eff_df = pd.DataFrame(efficiency_data)
    
    # 버블 차트
    scatter = ax2.scatter(eff_df['Training Time'], 
                         eff_df['ROUGE-F1'],
                         s=eff_df['Efficiency'] * 1000,
                         alpha=0.6)
    
    for idx, row in eff_df.iterrows():
        ax2.annotate(row['Experiment'], 
                    (row['Training Time'], row['ROUGE-F1']),
                    xytext=(5, 5), textcoords='offset points')
    
    ax2.set_xlabel('Training Time (hours)')
    ax2.set_ylabel('ROUGE-F1 Score')
    ax2.set_title('Training Efficiency Analysis')
    ax2.grid(True, alpha=0.3)
    
    # 3. 메모리 사용량 비교
    ax3 = axes[1, 0]
    memory_data = []
    
    for exp in ['best_phase1', '10a_phase1_plus_token_weight', 
                '10b_phase1_plus_backtrans', '10c_all_optimizations']:
        if exp in results:
            memory_data.append({
                'Experiment': exp.replace('10', 'Phase2-').replace('_', ' '),
                'Memory (GB)': results[exp].get('memory_peak', 0),
                'ROUGE-F1': results[exp].get('best_rouge_f1', 0)
            })
    
    mem_df = pd.DataFrame(memory_data)
    
    ax3.bar(range(len(mem_df)), mem_df['Memory (GB)'])
    ax3.set_xticks(range(len(mem_df)))
    ax3.set_xticklabels(mem_df['Experiment'], rotation=45)
    ax3.set_ylabel('Peak Memory Usage (GB)')
    ax3.set_title('Memory Usage Comparison')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 수렴 속도 비교
    ax4 = axes[1, 1]
    convergence_data = []
    
    for exp, name in [('10a_phase1_plus_token_weight', 'Token Weight'),
                      ('10b_phase1_plus_backtrans', 'BackTranslation'),
                      ('10c_all_optimizations', 'All Optimizations')]:
        if exp in results:
            convergence_data.append({
                'Experiment': name,
                'Convergence Epoch': results[exp].get('convergence_epoch', 0),
                'Best Score': results[exp].get('best_rouge_f1', 0)
            })
    
    conv_df = pd.DataFrame(convergence_data)
    
    bars = ax4.bar(conv_df['Experiment'], conv_df['Convergence Epoch'])
    
    # 막대 위에 최고 점수 표시
    for bar, score in zip(bars, conv_df['Best Score']):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}', ha='center', va='bottom')
    
    ax4.set_ylabel('Convergence Epoch')
    ax4.set_title('Convergence Speed Comparison')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/phase2_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_final_report(results: Dict, analysis_df: pd.DataFrame, output_dir: str):
    """최종 분석 보고서 생성"""
    report_path = f"{output_dir}/final_analysis_phase2.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 조합 실험 2차 - 최종 분석 보고서\n\n")
        f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 전체 요약
        f.write("## 전체 요약\n\n")
        
        baseline = results.get('baseline', {}).get('best_rouge_f1', 0.4712)
        phase1_best = results.get('best_phase1', {}).get('best_rouge_f1', baseline)
        
        if '10c_all_optimizations' in results:
            final_best = results['10c_all_optimizations']['best_rouge_f1']
            total_improvement = final_best - baseline
            f.write(f"- 베이스라인 성능: {baseline:.4f}\n")
            f.write(f"- 1차 최적 성능: {phase1_best:.4f} (+{phase1_best-baseline:.4f})\n")
            f.write(f"- 최종 달성 성능: {final_best:.4f} (+{total_improvement:.4f})\n")
            f.write(f"- 총 개선율: {(total_improvement/baseline)*100:.1f}%\n\n")
        
        # 상세 실험 결과
        f.write("## 상세 실험 결과\n\n")
        f.write(analysis_df.to_markdown(index=False))
        f.write("\n\n")
        
        # 주요 발견사항
        f.write("## 주요 발견사항\n\n")
        
        # 토큰 가중치 효과
        if '10a_phase1_plus_token_weight' in results:
            token_imp = results['10a_phase1_plus_token_weight']['best_rouge_f1'] - phase1_best
            f.write(f"### 1. 특수 토큰 가중치 효과\n")
            f.write(f"- ROUGE-F1 개선: +{token_imp:.4f}\n")
            f.write(f"- PII 재현율: {results['10a_phase1_plus_token_weight'].get('pii_recall', 0):.2%}\n")
            f.write(f"- 특수 토큰 정확도: {results['10a_phase1_plus_token_weight'].get('special_token_accuracy', 0):.2%}\n\n")
        
        # 백트랜슬레이션 효과
        if '10b_phase1_plus_backtrans' in results:
            back_imp = results['10b_phase1_plus_backtrans']['best_rouge_f1'] - phase1_best
            f.write(f"### 2. 백트랜슬레이션 효과\n")
            f.write(f"- ROUGE-F1 개선: +{back_imp:.4f}\n")
            f.write(f"- 학습 시간 증가: {results['10b_phase1_plus_backtrans']['training_time']:.1f}시간\n")
            f.write(f"- 데이터 다양성 증가로 인한 일반화 성능 향상\n\n")
        
        # 통합 효과
        if '10c_all_optimizations' in results:
            all_imp = results['10c_all_optimizations']['best_rouge_f1'] - phase1_best
            f.write(f"### 3. 전체 통합 효과\n")
            f.write(f"- ROUGE-F1 개선: +{all_imp:.4f}\n")
            f.write(f"- 시너지 효과 확인\n")
            f.write(f"- 메모리 사용량: {results['10c_all_optimizations']['memory_peak']:.1f}GB\n\n")
        
        # 권장사항
        f.write("## 권장사항\n\n")
        
        # 최적 구성 선택
        best_config = max([(k, v['best_rouge_f1']) for k, v in results.items() 
                          if k.startswith('10')], key=lambda x: x[1])
        
        f.write(f"### 최적 구성: {best_config[0]}\n")
        f.write(f"- 최고 성능: ROUGE-F1 {best_config[1]:.4f}\n")
        
        # 실용성 고려
        if best_config[0] == '10c_all_optimizations':
            f.write("- 주의: 학습 시간과 메모리 요구사항이 높음\n")
            f.write("- 대안: 10b_phase1_plus_backtrans (성능과 효율성의 균형)\n")
        
        f.write("\n### 최종 제출 준비\n")
        f.write("1. 선택된 구성으로 전체 데이터 재학습\n")
        f.write("2. 테스트 세트 추론 실행\n")
        f.write("3. 후처리 파이프라인 적용\n")
        f.write("4. 제출 파일 생성 및 검증\n")

def main():
    """메인 실행 함수"""
    # 경로 설정
    log_dir = "logs"
    output_dir = "outputs/phase2_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    print("2차 조합 실험 분석 시작...")
    
    # 결과 로드
    print("실험 결과 로드 중...")
    results = load_experiment_results(log_dir)
    
    # 분석 수행
    print("성능 분석 중...")
    analysis_df = analyze_phase2_improvements(results)
    
    # 시각화
    print("시각화 생성 중...")
    plot_performance_comparison(results, output_dir)
    
    # 보고서 생성
    print("최종 보고서 생성 중...")
    generate_final_report(results, analysis_df, output_dir)
    
    # 결과 출력
    print("\n=== 2차 조합 실험 결과 ===")
    print(analysis_df.to_string(index=False))
    
    print(f"\n분석 완료! 결과는 {output_dir}에 저장되었습니다.")

if __name__ == "__main__":
    main()
