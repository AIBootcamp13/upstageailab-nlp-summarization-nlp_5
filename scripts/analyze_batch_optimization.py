#!/usr/bin/env python3
"""
배치 크기 최적화 실험 분석 스크립트
각 실험의 메모리 사용량, 학습 속도, 성능을 분석합니다.
"""
import os
import json
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def parse_wandb_logs(log_dir):
    """WandB 로그에서 실험 메트릭 추출"""
    experiments = {}
    
    for exp_dir in Path(log_dir).iterdir():
        if not exp_dir.is_dir():
            continue
            
        exp_name = exp_dir.name
        if not exp_name.startswith('05'):
            continue
            
        # wandb 실행 로그 찾기
        wandb_dir = exp_dir / 'wandb'
        if not wandb_dir.exists():
            continue
            
        metrics = {
            'gpu_memory_usage': [],
            'training_speed': [],
            'eval_scores': [],
            'training_loss': [],
            'wall_time': []
        }
        
        # 가장 최근 실행 찾기
        latest_run = sorted(wandb_dir.glob('run-*'))[-1] if list(wandb_dir.glob('run-*')) else None
        if not latest_run:
            continue
            
        # 메트릭 파일 읽기
        history_file = latest_run / 'files' / 'wandb-history.jsonl'
        if history_file.exists():
            with open(history_file, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        
                        # GPU 메모리 사용량
                        if 'gpu_memory_allocated' in data:
                            metrics['gpu_memory_usage'].append(data['gpu_memory_allocated'])
                        
                        # 학습 속도 (steps/sec)
                        if 'train_steps_per_second' in data:
                            metrics['training_speed'].append(data['train_steps_per_second'])
                        
                        # 평가 점수
                        if 'eval_rouge_combined_f1' in data:
                            metrics['eval_scores'].append(data['eval_rouge_combined_f1'])
                        
                        # 학습 손실
                        if 'train_loss' in data:
                            metrics['training_loss'].append(data['train_loss'])
                        
                        # 시간
                        if '_timestamp' in data:
                            metrics['wall_time'].append(data['_timestamp'])
                    except:
                        continue
        
        experiments[exp_name] = metrics
    
    return experiments

def analyze_batch_experiments(config_dir, log_dir):
    """배치 실험 결과 분석"""
    # 실험 설정 읽기
    batch_configs = {}
    batch_dir = Path(config_dir) / '05_batch_optimization'
    
    for config_file in batch_dir.glob('*.yaml'):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            name = config_file.stem
            batch_configs[name] = {
                'batch_size': config['training']['per_device_train_batch_size'],
                'accumulation': config['training']['gradient_accumulation_steps'],
                'effective_batch': config['training']['per_device_train_batch_size'] * 
                                 config['training']['gradient_accumulation_steps']
            }
    
    # 로그 데이터 파싱
    experiments = parse_wandb_logs(log_dir)
    
    # 결과 분석
    results = []
    for exp_name, config in batch_configs.items():
        if exp_name in experiments:
            metrics = experiments[exp_name]
            
            result = {
                '실험명': exp_name,
                '배치크기': config['batch_size'],
                'Accumulation': config['accumulation'],
                '효과적배치크기': config['effective_batch'],
                '평균GPU메모리(GB)': np.mean(metrics['gpu_memory_usage']) / 1024**3 if metrics['gpu_memory_usage'] else 0,
                '최대GPU메모리(GB)': np.max(metrics['gpu_memory_usage']) / 1024**3 if metrics['gpu_memory_usage'] else 0,
                '평균학습속도(steps/sec)': np.mean(metrics['training_speed']) if metrics['training_speed'] else 0,
                '최종ROUGE점수': metrics['eval_scores'][-1] if metrics['eval_scores'] else 0,
                '최고ROUGE점수': max(metrics['eval_scores']) if metrics['eval_scores'] else 0,
                '총학습시간(분)': (metrics['wall_time'][-1] - metrics['wall_time'][0]) / 60 if len(metrics['wall_time']) > 1 else 0
            }
            results.append(result)
    
    # 데이터프레임 생성
    df = pd.DataFrame(results)
    df = df.sort_values('실험명')
    
    return df, experiments

def visualize_results(df, experiments, output_dir):
    """실험 결과 시각화"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 메모리 사용량 vs 학습 속도
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    scatter = ax.scatter(df['평균GPU메모리(GB)'], df['평균학습속도(steps/sec)'], 
                        s=df['배치크기']*5, alpha=0.6)
    
    for idx, row in df.iterrows():
        ax.annotate(f"BS={row['배치크기']}\nAcc={row['Accumulation']}", 
                   (row['평균GPU메모리(GB)'], row['평균학습속도(steps/sec)']),
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax.set_xlabel('평균 GPU 메모리 사용량 (GB)')
    ax.set_ylabel('평균 학습 속도 (steps/sec)')
    ax.set_title('배치 크기별 메모리 효율성 분석')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'memory_vs_speed.png'), dpi=150)
    plt.close()
    
    # 2. 학습 곡선 비교
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 학습 손실 곡선
    ax = axes[0, 0]
    for exp_name, metrics in experiments.items():
        if metrics['training_loss']:
            ax.plot(metrics['training_loss'], label=exp_name.replace('05_', ''), alpha=0.7)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Training Loss')
    ax.set_title('학습 손실 곡선')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ROUGE 점수 변화
    ax = axes[0, 1]
    for exp_name, metrics in experiments.items():
        if metrics['eval_scores']:
            epochs = list(range(1, len(metrics['eval_scores']) + 1))
            ax.plot(epochs, metrics['eval_scores'], 'o-', label=exp_name.replace('05_', ''), alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('ROUGE Combined F1')
    ax.set_title('검증 성능 변화')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 메모리 사용량 시계열
    ax = axes[1, 0]
    for exp_name, metrics in experiments.items():
        if metrics['gpu_memory_usage']:
            memory_gb = [m / 1024**3 for m in metrics['gpu_memory_usage']]
            ax.plot(memory_gb, label=exp_name.replace('05_', ''), alpha=0.7)
    ax.set_xlabel('Steps')
    ax.set_ylabel('GPU Memory (GB)')
    ax.set_title('GPU 메모리 사용량 변화')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 성능 대비 효율성
    ax = axes[1, 1]
    efficiency = df['최고ROUGE점수'] / df['총학습시간(분)'] * 100  # 분당 ROUGE 향상
    bars = ax.bar(df['실험명'].str.replace('05_', ''), efficiency)
    ax.set_ylabel('효율성 (ROUGE/분 × 100)')
    ax.set_title('학습 효율성 비교')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # 막대 위에 값 표시
    for bar, val in zip(bars, efficiency):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150)
    plt.close()
    
    # 3. 종합 비교 테이블
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # 테이블 데이터 준비
    table_data = df[['실험명', '배치크기', 'Accumulation', '평균GPU메모리(GB)', 
                     '평균학습속도(steps/sec)', '최고ROUGE점수', '총학습시간(분)']].round(3)
    
    table = ax.table(cellText=table_data.values,
                     colLabels=table_data.columns,
                     cellLoc='center',
                     loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # 최적 값 하이라이트
    best_speed_idx = df['평균학습속도(steps/sec)'].idxmax()
    best_score_idx = df['최고ROUGE점수'].idxmax()
    best_efficiency_idx = efficiency.idxmax()
    
    for i in range(len(df)):
        if i == best_speed_idx:
            table[(i+1, 4)].set_facecolor('#90EE90')  # 최고 속도
        if i == best_score_idx:
            table[(i+1, 5)].set_facecolor('#FFB6C1')  # 최고 점수
        if i == best_efficiency_idx:
            table[(i+1, 0)].set_facecolor('#87CEEB')  # 최고 효율
    
    plt.title('배치 크기 최적화 실험 결과 요약', fontsize=14, pad=20)
    plt.savefig(os.path.join(output_dir, 'summary_table.png'), dpi=150, bbox_inches='tight')
    plt.close()

def generate_report(df, output_dir):
    """최종 보고서 생성"""
    report = []
    report.append("# 배치 크기 및 Gradient Accumulation 최적화 실험 결과\n")
    report.append("## 실험 개요")
    report.append("- 목적: 메모리 효율과 학습 성능의 최적점 찾기")
    report.append("- 효과적인 배치 크기를 64로 고정하고 다양한 조합 실험")
    report.append(f"- 총 {len(df)}개 구성 테스트\n")
    
    report.append("## 주요 발견사항\n")
    
    # 최고 성능
    best_score_row = df.loc[df['최고ROUGE점수'].idxmax()]
    report.append(f"### 1. 최고 성능 구성")
    report.append(f"- 실험: {best_score_row['실험명']}")
    report.append(f"- 배치 크기: {best_score_row['배치크기']}, Accumulation: {best_score_row['Accumulation']}")
    report.append(f"- ROUGE F1: {best_score_row['최고ROUGE점수']:.4f}\n")
    
    # 최고 속도
    best_speed_row = df.loc[df['평균학습속도(steps/sec)'].idxmax()]
    report.append(f"### 2. 최고 속도 구성")
    report.append(f"- 실험: {best_speed_row['실험명']}")
    report.append(f"- 배치 크기: {best_speed_row['배치크기']}, Accumulation: {best_speed_row['Accumulation']}")
    report.append(f"- 학습 속도: {best_speed_row['평균학습속도(steps/sec)']:.2f} steps/sec")
    report.append(f"- 메모리 사용: {best_speed_row['평균GPU메모리(GB)']:.2f} GB\n")
    
    # 효율성 분석
    efficiency = df['최고ROUGE점수'] / df['총학습시간(분)'] * 100
    best_eff_idx = efficiency.idxmax()
    best_eff_row = df.loc[best_eff_idx]
    
    report.append(f"### 3. 최고 효율성 구성")
    report.append(f"- 실험: {best_eff_row['실험명']}")
    report.append(f"- 효율성 점수: {efficiency[best_eff_idx]:.2f} (ROUGE/분 × 100)")
    report.append(f"- 학습 시간: {best_eff_row['총학습시간(분)']:.1f}분\n")
    
    report.append("## 권장사항\n")
    report.append("1. **개발/실험 단계**: 중간 배치 + 중간 accumulation (빠른 반복)")
    report.append("2. **최종 학습**: 작은 배치 + 높은 accumulation (최고 성능)")
    report.append("3. **리소스 제한시**: 큰 배치 + accumulation 없음 (메모리 효율적)")
    
    # 보고서 저장
    with open(os.path.join(output_dir, 'batch_optimization_report.md'), 'w') as f:
        f.write('\n'.join(report))
    
    # CSV로도 저장
    df.to_csv(os.path.join(output_dir, 'batch_optimization_results.csv'), index=False)
    
    print("분석 완료!")
    print(f"결과 저장 위치: {output_dir}")

def main():
    """메인 실행 함수"""
    # 경로 설정
    project_root = Path(__file__).parent.parent
    config_dir = project_root / 'config' / 'experiments'
    log_dir = project_root / 'outputs' / 'logs'
    output_dir = project_root / 'outputs' / 'analysis' / 'batch_optimization'
    
    # 분석 실행
    df, experiments = analyze_batch_experiments(config_dir, log_dir)
    
    if not df.empty:
        visualize_results(df, experiments, output_dir)
        generate_report(df, output_dir)
    else:
        print("실험 결과를 찾을 수 없습니다. 실험을 먼저 실행해주세요.")

if __name__ == "__main__":
    main()
