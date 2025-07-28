#!/usr/bin/env python3
"""
Solar API 앙상블 실행 스크립트
"""

import os
import sys
import json
import yaml
import argparse
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import torch

# 프로젝트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

from ensemble.solar_ensemble import WeightedEnsemble, EnsembleConfig
from utils.experiment_utils import ExperimentTracker
import wandb

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """YAML 설정 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def setup_wandb(config: dict):
    """WandB 설정"""
    if config['wandb']['enabled']:
        wandb.init(
            project=config['wandb']['project'],
            name=config['wandb']['name'],
            tags=config['wandb']['tags'],
            notes=config['wandb']['notes'],
            config=config
        )


def get_best_model_path(config: dict) -> str:
    """최고 성능 모델 경로 찾기"""
    base_path = Path(config['base_model']['model_path'])
    
    # 실제 모델 경로 확인
    if base_path.exists():
        return str(base_path)
    
    # 대안 경로들 확인
    alternatives = [
        "outputs/phase2_results/10c_all_optimizations/checkpoint-best",
        "outputs/phase2_results/10b_phase1_plus_backtrans/checkpoint-best",
        "outputs/phase2_results/10a_phase1_plus_token_weight/checkpoint-best",
        "models/baseline/checkpoint-best"
    ]
    
    for alt_path in alternatives:
        if Path(alt_path).exists():
            logger.info(f"Using alternative model path: {alt_path}")
            return alt_path
    
    raise ValueError(f"No valid model found. Checked paths: {[str(base_path)] + alternatives}")


def run_ensemble_experiment(config: dict, mode: dict):
    """앙상블 실험 실행"""
    logger.info(f"Running ensemble mode: {mode['name']} - {mode['description']}")
    
    # API 키 확인
    api_key = os.getenv(config['solar_api']['api_key_env'])
    if not api_key:
        raise ValueError(f"Please set {config['solar_api']['api_key_env']} environment variable")
    
    # 앙상블 설정
    ensemble_config = EnsembleConfig(
        solar_api_key=api_key,
        solar_model=config['solar_api']['model'],
        solar_base_url=config['solar_api']['base_url'],
        fine_tuned_weight=mode.get('weights', mode.get('base_weights', [0.7, 0.3]))[0],
        solar_weight=mode.get('weights', mode.get('base_weights', [0.7, 0.3]))[1],
        dynamic_weights=mode['dynamic_weights'],
        temperature=config['solar_api']['temperature'],
        top_p=config['solar_api']['top_p'],
        max_length=config['solar_api']['max_tokens'],
        rate_limit_per_minute=config['solar_api']['rate_limit_per_minute'],
        max_retries=config['solar_api']['max_retries'],
        retry_delay=config['solar_api']['retry_delay'],
        timeout=config['solar_api']['timeout'],
        batch_size=config['batch_processing']['batch_size'],
        use_async=config['batch_processing']['use_async'],
        use_cache=config['caching']['enabled'],
        cache_dir=config['caching']['cache_dir']
    )
    
    # 모델 경로
    model_path = get_best_model_path(config)
    
    # 디바이스 설정
    device = config['base_model']['device']
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.backends.mps.is_available():
            device = "mps"
    
    # 앙상블 생성
    ensemble = WeightedEnsemble(
        fine_tuned_model_path=model_path,
        ensemble_config=ensemble_config,
        device=device
    )
    
    # Few-shot 예제 로드
    if config['few_shot']['enabled']:
        ensemble.load_few_shot_examples(
            config['few_shot']['source_file'],
            config['few_shot']['num_examples']
        )
    
    # 출력 디렉토리 생성
    output_dir = Path(config['execution']['output_dir']) / mode['name']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 실험 추적
    tracker = ExperimentTracker(str(output_dir))
    experiment_id = tracker.start_experiment(
        name=f"solar_ensemble_{mode['name']}",
        description=mode['description'],
        config={**config, 'mode': mode}
    )
    
    results = {}
    
    try:
        # 검증세트 처리
        logger.info("Processing validation set...")
        val_output = output_dir / "val_results.csv"
        
        val_df = ensemble.process_dataset(
            data_file=config['execution']['dev_file'],
            output_file=str(val_output),
            sample_size=config['execution']['sample_size']
        )
        
        # 평가
        eval_results = ensemble.evaluate(
            str(val_output),
            config['execution']['dev_file']
        )
        
        results['validation'] = eval_results
        
        # WandB 로깅
        if config['wandb']['enabled']:
            wandb.log({
                f"{mode['name']}/val_rouge1": eval_results['ensemble_rouge1'],
                f"{mode['name']}/val_rouge2": eval_results['ensemble_rouge2'],
                f"{mode['name']}/val_rougeL": eval_results['ensemble_rougeL'],
                f"{mode['name']}/val_rouge_avg": eval_results['ensemble_rouge_avg'],
                f"{mode['name']}/improvement": eval_results.get('improvement_percent', 0)
            })
        
        # 성능이 좋으면 테스트 세트 처리
        if eval_results['ensemble_rouge_avg'] > 0.5:  # 50% 이상
            logger.info("Processing test set...")
            test_output = output_dir / "test_results.csv"
            
            test_df = ensemble.process_dataset(
                data_file=config['execution']['test_file'],
                output_file=str(test_output),
                sample_size=None  # 전체 처리
            )
            
            # 제출 파일 생성
            if config['submission']['create_submission_file']:
                submission_df = test_df[['fname', 'ensemble_summary']].copy()
                submission_df.columns = ['fname', 'summary']
                submission_path = output_dir / "submission.csv"
                submission_df.to_csv(submission_path, index=False)
                logger.info(f"Submission file created: {submission_path}")
        
        # 통계 저장
        stats = {
            'mode': mode['name'],
            'val_results': eval_results,
            'api_calls': ensemble.solar_client.request_count,
            'cache_hits': len(ensemble.solar_client.cache) if ensemble_config.use_cache else 0,
            'avg_confidence': val_df['confidence_score'].mean(),
            'weight_distribution': {
                'fine_tuned': val_df['fine_tuned_weight'].describe().to_dict(),
                'solar': val_df['solar_weight'].describe().to_dict()
            } if mode['dynamic_weights'] else None
        }
        
        stats_path = output_dir / "experiment_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        # 실험 완료
        tracker.end_experiment(
            experiment_id,
            status="completed",
            metrics=eval_results
        )
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        tracker.end_experiment(
            experiment_id,
            status="failed",
            error=str(e)
        )
        raise
    
    finally:
        ensemble.close()
    
    return results


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Run Solar API Ensemble")
    parser.add_argument("--config", type=str, 
                       default="config/experiments/11_solar_ensemble.yaml",
                       help="Config file path")
    parser.add_argument("--mode", type=str, default=None,
                       help="Specific mode to run")
    parser.add_argument("--all", action="store_true",
                       help="Run all modes")
    
    args = parser.parse_args()
    
    # 설정 로드
    config = load_config(args.config)
    
    # WandB 설정
    setup_wandb(config)
    
    # 실행할 모드 결정
    if args.all:
        modes = config['experiment']['modes']
    elif args.mode:
        modes = [m for m in config['experiment']['modes'] if m['name'] == args.mode]
        if not modes:
            raise ValueError(f"Mode {args.mode} not found")
    else:
        # 기본: dynamic_weights 모드
        modes = [m for m in config['experiment']['modes'] if m['name'] == 'dynamic_weights']
    
    # 각 모드 실행
    all_results = {}
    
    for mode in modes:
        try:
            results = run_ensemble_experiment(config, mode)
            all_results[mode['name']] = results
        except Exception as e:
            logger.error(f"Failed to run mode {mode['name']}: {str(e)}")
            continue
    
    # 최종 비교 분석
    if len(all_results) > 1:
        logger.info("\n=== Final Comparison ===")
        comparison_data = []
        
        for mode_name, results in all_results.items():
            if 'validation' in results:
                comparison_data.append({
                    'Mode': mode_name,
                    'ROUGE-1': results['validation']['ensemble_rouge1'],
                    'ROUGE-2': results['validation']['ensemble_rouge2'],
                    'ROUGE-L': results['validation']['ensemble_rougeL'],
                    'Average': results['validation']['ensemble_rouge_avg'],
                    'Improvement': results['validation'].get('improvement_percent', 0)
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        # 최고 성능 모드
        best_mode = comparison_df.loc[comparison_df['Average'].idxmax()]
        logger.info(f"\nBest performing mode: {best_mode['Mode']} with ROUGE avg: {best_mode['Average']:.4f}")
    
    # WandB 종료
    if config['wandb']['enabled']:
        wandb.finish()
    
    logger.info("All experiments completed!")


if __name__ == "__main__":
    main()
