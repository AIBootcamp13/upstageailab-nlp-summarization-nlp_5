#!/usr/bin/env python3
"""
조합 실험 1차 분석 스크립트
각 구성요소의 기여도와 시너지 효과를 분석합니다.
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

def load_experiment_results(log_dir, experiment_prefix="06"):
    """실험 결과 로드"""
    results = {}
    
    # 베이스라인 결과 로드
    baseline_dir = Path(log_dir) / "00_baseline_reproduction"
    if baseline_dir.exists():
        results['baseline'] = parse_experiment_logs(baseline_dir)
    
    # 개별 구성요소 결과 로드
    component_results = {
        'augmentation': '01_simple_augmentation',
        'postprocessing': '02_postprocessing',
        'lr_scheduling': '03_lr_scheduling/03a_cosine_annealing',
        'normalization': '04_text_normalization'
    }
    
    for comp_name, comp_dir in component_results.items():
        exp_dir = Path(log_dir) / comp_dir
        if exp_dir.exists():
            results[comp_name] = parse_experiment_logs(exp_dir)
    
    # 조합 실험 결과 로드
    for exp_dir in Path(log_dir).iterdir():
        if exp_dir.is_dir() and exp_dir.name.startswith(experiment_prefix):
            results[exp_dir.name] = parse_experiment_logs(exp_dir)
    
    return results

def parse_experiment_logs(exp_dir):
    """실험 로그 파싱"""
    metrics = {
        'best_rouge_f1': 0,
        'final_rouge_f1': 0,
        'training_time': 0,
        'convergence_epoch': 0,
        'memory_usage': 0
    }
    
    # wandb 로그 파일 찾기
    wandb_dir = exp_dir / 'wandb'
    if wandb_dir.exists():
        latest_run = sorted(wandb_dir.glob('run-*'))[-1] if list(wandb_dir.glob('run-*')) else None
        if latest_run:
            # 메트릭 파일 읽기
            history_file = latest_run / 'files' / 'wandb-history.jsonl'
            if history_file.exists():
                rouge_scores = []
                timestamps = []
                
                with open(history_file, 'r') as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            if 'eval_rouge_combined_f1' in data:
                                rouge_scores.append(data['eval_rouge_combined_f1'])
                            if '_timestamp' in data:
                                timestamps.append(data['_timestamp'])
                        except:
                            continue
                
                if rouge_scores:
                    metrics['best_rouge_f1'] = max(rouge_scores)
                    metrics['final_rouge_f1'] = rouge_scores[-1]
                    metrics['convergence_epoch'] = rouge_scores.index(max(rouge_scores)) + 1
                
                if timestamps:
                    metrics['training_time'] = (timestamps[-1] - timestamps[0]) / 60  # 분 단위
    
    # 결과 파일에서 추가 메트릭 읽기
    results_file = exp_dir / 'results.json'
    if results_file.exists():
        with open(results_file, 'r') as f:
            saved_results = json.load(f)
            metrics.update(saved_results.get('metrics', {}))
    
    return metrics

def analyze_combinations(results):
    """조합 효과 분석"""
    analysis = {}
    
    # 베이스라인 성능
    baseline_score = results.get('baseline', {}).get('best_rouge_f1', 0.4712)
    
    # 개별 구성요소 효과
    individual_effects = {}
    for comp in ['augmentation', 'postprocessing', 'lr_scheduling', 'normalization']:
        if comp in results:
            individual_effects[comp] = results[comp]['best_rouge_f1'] - baseline_score
    
    # 조합 효과 분석
    combinations = {
        '06a_aug_plus_post': ['augmentation', 'postprocessing'],
        '06b_aug_plus_lr': ['augmentation', 'lr_scheduling'],
        '06c_all_simple': ['augmentation', 'postprocessing', 'lr_scheduling', 'normalization']
    }
    
    for comb_name, components in combinations.items():
        if comb_name in results:
            # 실제 조합 성능
            actual_score = results[comb_name]['best_rouge_f1']
            actual_improvement = actual_score - baseline_score
            
            # 예상 성능 (개별 효과의 합)
            expected_improvement = sum(individual_effects.get(comp, 0) for comp in components)
            expected_score = baseline_score + expected_improvement
            
            # 시너지 효과
            synergy = actual_improvement - expected_improvement
            
            analysis[comb_name] = {
                'components': components,
                'actual_score': actual_score,
                'expected_score': expected_score,
                'actual_improvement': actual_improvement,
                'expected_improvement': expected_improvement,
                'synergy': synergy,
                'synergy_ratio': synergy / expected_improvement if expected_improvement > 0 else 0
            }
    
    return analysis, individual_effects

def visualize_results(results, analysis, output_dir):
    """결과 시각화"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 개별 구성요소 vs 조합 성능 비교
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # 데이터 준비
    experiments = []
    scores = []
    categories = []
    
    # 베이스라인
    baseline_score = results.get('baseline', {}).get('best_rouge_f1', 0.4712)
    experiments.append('Baseline')
    scores.append(baseline_score)
    categories.append('baseline')
    
    # 개별 구성요소
    for comp in ['augmentation', 'postprocessing', 'lr_scheduling', 'normalization']:
        if comp in results:
            experiments.append(comp.replace('_', ' ').title())
            scores.append(results[comp]['best_rouge_f1'])
            categories.append('individual')
    
    # 조합
    for comb_name in ['06a_aug_plus_post', '06b_aug_plus_lr', '06c_all_simple']:
        if comb_name in results:
            display_name = {
                '06a_aug_plus_post': 'Aug + Post',
                '06b_aug_plus_lr': 'Aug + LR',
                '06c_all_simple': 'All Simple'
            }.get(comb_name, comb_name)
            experiments.append(display_name)
            scores.append(results[comb_name]['best_rouge_f1'])
            categories.append('combination')
    
    # 색상 매핑
    color_map = {'baseline': '#gray', 'individual': '#skyblue', 'combination': '#orange'}
    colors = [color_map[cat] for cat in categories]
    
    # 막대 그래프
    bars = ax.bar(experiments, scores, color=colors)
    
    # 목표선 추가
    ax.axhline(y=0.52, color='red', linestyle='--', label='목표 (52%)')
    
    # 값 표시
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                f'{score:.3f}', ha='center', va='bottom')
    
    ax.set_ylabel('ROUGE-F1 Score')
    ax.set_title('개별 구성요소 vs 조합 성능 비교')
    ax.set_xticklabels(experiments, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'component_comparison.png'), dpi=150)
    plt.close()
    
    # 2. 시너지 효과 분석
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    comb_names = []
    synergies = []
    expected_improvements = []
    actual_improvements = []
    
    for comb_name, data in analysis.items():
        display_name = {
            '06a_aug_plus_post': 'Aug + Post',
            '06b_aug_plus_lr': 'Aug + LR',
            '06c_all_simple': 'All Simple'
        }.get(comb_name, comb_name)
        
        comb_names.append(display_name)
        synergies.append(data['synergy'] * 100)  # 퍼센트로 변환
        expected_improvements.append(data['expected_improvement'] * 100)
        actual_improvements.append(data['actual_improvement'] * 100)
    
    x = np.arange(len(comb_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, expected_improvements, width, label='예상 향상', color='lightblue')
    bars2 = ax.bar(x + width/2, actual_improvements, width, label='실제 향상', color='darkblue')
    
    # 시너지 표시
    for i, (exp, act, syn) in enumerate(zip(expected_improvements, actual_improvements, synergies)):
        if syn > 0:
            ax.annotate(f'+{syn:.1f}%', xy=(i, act), xytext=(i, act + 0.5),
                       ha='center', va='bottom', color='green', fontweight='bold')
        else:
            ax.annotate(f'{syn:.1f}%', xy=(i, act), xytext=(i, act - 0.5),
                       ha='center', va='top', color='red', fontweight='bold')
    
    ax.set_ylabel('향상율 (%)')
    ax.set_title('예상 vs 실제 성능 향상 (시너지 효과)')
    ax.set_xticks(x)
    ax.set_xticklabels(comb_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'synergy_analysis.png'), dpi=150)
    plt.close()
    
    # 3. 효율성 분석 (성능 vs 학습 시간)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    exp_names = []
    rouge_scores = []
    training_times = []
    
    for exp_name, metrics in results.items():
        if exp_name.startswith('06') or exp_name in ['augmentation', 'postprocessing', 'lr_scheduling', 'normalization']:
            display_name = exp_name.replace('_', ' ').title() if exp_name in ['augmentation', 'postprocessing', 'lr_scheduling', 'normalization'] else {
                '06a_aug_plus_post': 'Aug + Post',
                '06b_aug_plus_lr': 'Aug + LR',
                '06c_all_simple': 'All Simple'
            }.get(exp_name, exp_name)
            
            exp_names.append(display_name)
            rouge_scores.append(metrics['best_rouge_f1'])
            training_times.append(metrics.get('training_time', 0))
    
    # 효율성 점수 계산 (ROUGE 향상 / 학습 시간)
    efficiencies = [(score - baseline_score) / time * 100 if time > 0 else 0 
                   for score, time in zip(rouge_scores, training_times)]
    
    scatter = ax.scatter(training_times, rouge_scores, s=[eff*100 for eff in efficiencies], 
                        c=efficiencies, cmap='viridis', alpha=0.6)
    
    for name, time, score in zip(exp_names, training_times, rouge_scores):
        ax.annotate(name, (time, score), xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax.axhline(y=0.52, color='red', linestyle='--', alpha=0.5, label='목표 (52%)')
    ax.axhline(y=baseline_score, color='gray', linestyle='--', alpha=0.5, label='베이스라인')
    
    ax.set_xlabel('학습 시간 (분)')
    ax.set_ylabel('ROUGE-F1 Score')
    ax.set_title('효율성 분석 (점 크기 = 효율성)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('효율성 점수')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'efficiency_analysis.png'), dpi=150)
    plt.close()

def generate_report(results, analysis, individual_effects, output_dir):
    """분석 보고서 생성"""
    report = []
    report.append("# 조합 실험 1차 분석 결과\n")
    report.append(f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 베이스라인
    baseline_score = results.get('baseline', {}).get('best_rouge_f1', 0.4712)
    report.append("## 1. 베이스라인 성능")
    report.append(f"- ROUGE-F1: {baseline_score:.4f}")
    report.append(f"- 목표: 0.5200 (향상 필요: {0.52 - baseline_score:.4f})\n")
    
    # 개별 구성요소 효과
    report.append("## 2. 개별 구성요소 효과")
    report.append("| 구성요소 | ROUGE-F1 | 향상폭 | 향상율(%) |")
    report.append("|---------|----------|--------|----------|")
    
    for comp, effect in individual_effects.items():
        if comp in results:
            score = results[comp]['best_rouge_f1']
            improvement = score - baseline_score
            improvement_rate = (improvement / baseline_score) * 100
            report.append(f"| {comp.replace('_', ' ').title()} | {score:.4f} | +{improvement:.4f} | +{improvement_rate:.2f}% |")
    
    report.append("")
    
    # 조합 효과 분석
    report.append("## 3. 조합 효과 분석")
    
    for comb_name, data in analysis.items():
        display_name = {
            '06a_aug_plus_post': '데이터 증강 + 후처리',
            '06b_aug_plus_lr': '데이터 증강 + 학습률 스케줄링',
            '06c_all_simple': '모든 간단한 개선사항'
        }.get(comb_name, comb_name)
        
        report.append(f"\n### {display_name}")
        report.append(f"- 구성요소: {', '.join(data['components'])}")
        report.append(f"- 실제 점수: {data['actual_score']:.4f}")
        report.append(f"- 예상 점수: {data['expected_score']:.4f}")
        report.append(f"- 시너지 효과: {data['synergy']:.4f} ({data['synergy_ratio']*100:.1f}%)")
        
        if data['synergy'] > 0:
            report.append("- ✅ 긍정적 시너지: 구성요소들이 서로 보완적으로 작용")
        else:
            report.append("- ⚠️ 부정적 시너지: 구성요소 간 간섭 발생")
    
    # 최적 구성 선정
    report.append("\n## 4. 최적 구성 선정")
    
    best_comb = max(analysis.items(), key=lambda x: x[1]['actual_score'])
    best_name = best_comb[0]
    best_data = best_comb[1]
    
    report.append(f"### 최고 성능 조합: {best_name}")
    report.append(f"- ROUGE-F1: {best_data['actual_score']:.4f}")
    report.append(f"- 베이스라인 대비 향상: +{best_data['actual_improvement']:.4f} (+{(best_data['actual_improvement']/baseline_score)*100:.2f}%)")
    report.append(f"- 목표 달성률: {(best_data['actual_score']/0.52)*100:.1f}%")
    
    # 효율성 분석
    if best_name in results:
        training_time = results[best_name].get('training_time', 0)
        if training_time > 0:
            efficiency = (best_data['actual_improvement'] / training_time) * 100
            report.append(f"- 학습 시간: {training_time:.1f}분")
            report.append(f"- 효율성: {efficiency:.3f} (향상율/분)")
    
    # 권장사항
    report.append("\n## 5. 권장사항")
    
    if best_data['actual_score'] >= 0.52:
        report.append("✅ **목표 달성!** 현재 조합으로 목표 성능을 달성했습니다.")
        report.append("- 2차 실험에서는 추가 고급 기능으로 더 높은 성능을 목표로 합니다.")
    else:
        gap = 0.52 - best_data['actual_score']
        report.append(f"⚠️ **추가 개선 필요** (부족: {gap:.4f})")
        report.append("- 2차 실험에서 고급 기능 통합이 필수적입니다.")
        report.append("- 특수 토큰 가중치, 빔 서치 최적화 등이 필요합니다.")
    
    # 다음 단계
    report.append("\n## 6. 다음 단계")
    report.append("1. 최적 조합을 기반으로 2차 고급 기능 실험 진행")
    report.append("2. 특수 토큰 가중치 조정 구현")
    report.append("3. 빔 서치 파라미터 최적화")
    report.append("4. 백트랜슬레이션 데이터 증강")
    
    # 보고서 저장
    report_path = os.path.join(output_dir, 'phase1_combination_report.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    # 최적 구성 YAML 저장
    if best_name in ['06a_aug_plus_post', '06b_aug_plus_lr', '06c_all_simple']:
        config_path = Path(__file__).parent.parent / 'config' / 'experiments' / '06_combination_phase1' / f'{best_name}.yaml'
        if config_path.exists():
            with open(config_path, 'r') as f:
                best_config = yaml.safe_load(f)
            
            # 최적 구성으로 저장
            best_config['experiment_name'] = 'phase1_best_combination'
            best_config['description'] = f'1차 실험 최적 조합 - {best_name} 기반'
            best_config['wandb']['name'] = '06_best_combination'
            best_config['wandb']['notes'] = f'Phase 1 최적 조합 (ROUGE-F1: {best_data["actual_score"]:.4f})'
            
            best_config_path = Path(__file__).parent.parent / 'config' / 'experiments' / '06_combination_phase1' / '06_best_combination.yaml'
            with open(best_config_path, 'w') as f:
                yaml.dump(best_config, f, default_flow_style=False, allow_unicode=True)
            
            report.append(f"\n최적 구성이 '{best_config_path.name}'으로 저장되었습니다.")
    
    print("분석 완료!")
    print(f"보고서 저장 위치: {report_path}")
    
    return best_name, best_data

def create_ablation_table(results, analysis, output_dir):
    """Ablation Study 테이블 생성"""
    # 데이터 준비
    rows = []
    
    # 베이스라인
    baseline_score = results.get('baseline', {}).get('best_rouge_f1', 0.4712)
    rows.append({
        'Configuration': 'Baseline',
        'Augmentation': '✗',
        'Postprocessing': '✗',
        'LR Scheduling': '✗',
        'Normalization': '✗',
        'ROUGE-F1': baseline_score,
        'Δ from Baseline': 0,
        'Training Time (min)': results.get('baseline', {}).get('training_time', 0)
    })
    
    # 개별 구성요소
    component_map = {
        'augmentation': ('✓', '✗', '✗', '✗'),
        'postprocessing': ('✗', '✓', '✗', '✗'),
        'lr_scheduling': ('✗', '✗', '✓', '✗'),
        'normalization': ('✗', '✗', '✗', '✓')
    }
    
    for comp, marks in component_map.items():
        if comp in results:
            rows.append({
                'Configuration': comp.replace('_', ' ').title(),
                'Augmentation': marks[0],
                'Postprocessing': marks[1],
                'LR Scheduling': marks[2],
                'Normalization': marks[3],
                'ROUGE-F1': results[comp]['best_rouge_f1'],
                'Δ from Baseline': results[comp]['best_rouge_f1'] - baseline_score,
                'Training Time (min)': results[comp].get('training_time', 0)
            })
    
    # 조합
    combination_map = {
        '06a_aug_plus_post': ('Aug + Post', '✓', '✓', '✗', '✗'),
        '06b_aug_plus_lr': ('Aug + LR', '✓', '✗', '✓', '✗'),
        '06c_all_simple': ('All Simple', '✓', '✓', '✓', '✓')
    }
    
    for comb_name, (display_name, *marks) in combination_map.items():
        if comb_name in results:
            rows.append({
                'Configuration': display_name,
                'Augmentation': marks[0],
                'Postprocessing': marks[1],
                'LR Scheduling': marks[2],
                'Normalization': marks[3],
                'ROUGE-F1': results[comb_name]['best_rouge_f1'],
                'Δ from Baseline': results[comb_name]['best_rouge_f1'] - baseline_score,
                'Training Time (min)': results[comb_name].get('training_time', 0)
            })
    
    # 데이터프레임 생성 및 저장
    df = pd.DataFrame(rows)
    df = df.round({'ROUGE-F1': 4, 'Δ from Baseline': 4, 'Training Time (min)': 1})
    
    # CSV 저장
    csv_path = os.path.join(output_dir, 'ablation_study_results.csv')
    df.to_csv(csv_path, index=False)
    
    # 시각화
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # 최고 점수 하이라이트
    best_idx = df['ROUGE-F1'].idxmax()
    for col in range(len(df.columns)):
        table[(best_idx + 1, col)].set_facecolor('#90EE90')
    
    plt.title('Ablation Study: 구성요소별 성능 분석', fontsize=14, pad=20)
    plt.savefig(os.path.join(output_dir, 'ablation_study_table.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return df

def main():
    """메인 실행 함수"""
    # 경로 설정
    project_root = Path(__file__).parent.parent
    log_dir = project_root / 'outputs' / 'logs'
    output_dir = project_root / 'outputs' / 'analysis' / 'combination_phase1'
    
    # 결과 로드 및 분석
    results = load_experiment_results(log_dir)
    
    if len(results) < 3:
        print("실험 결과가 충분하지 않습니다. 실험을 먼저 실행해주세요.")
        return
    
    # 조합 효과 분석
    analysis, individual_effects = analyze_combinations(results)
    
    # 시각화
    visualize_results(results, analysis, output_dir)
    
    # Ablation Study 테이블
    ablation_df = create_ablation_table(results, analysis, output_dir)
    
    # 보고서 생성
    best_name, best_data = generate_report(results, analysis, individual_effects, output_dir)
    
    print(f"\n최적 조합: {best_name}")
    print(f"ROUGE-F1: {best_data['actual_score']:.4f}")
    print(f"목표 달성 여부: {'✅ 달성' if best_data['actual_score'] >= 0.52 else '❌ 미달성'}")

if __name__ == "__main__":
    main()
