"""
WandB Sweep 실행기

6조의 main_sweep_test.py를 참고하여 NLP 대화 요약 프로젝트에 맞게 구현한 Sweep 실행기.
다양한 하이퍼파라미터 조합을 자동으로 실험하고 최적의 설정을 찾습니다.
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import yaml
from datetime import datetime

import wandb
import torch
from functools import partial

# 로컬 모듈 임포트
from trainer import DialogueSummarizationTrainer, create_trainer
from utils import load_config


logger = logging.getLogger(__name__)


class SweepRunner:
    """
    WandB Sweep 실행 및 관리 클래스
    
    ConfigManager와 Trainer를 활용하여 체계적인 하이퍼파라미터 최적화 수행
    """
    
    def __init__(self, base_config_path: str, 
                 sweep_config_name: str,
                 project_name: Optional[str] = None,
                 entity: Optional[str] = None):
        """
        SweepRunner 초기화
        
        Args:
            base_config_path: 기본 설정 파일 경로
            sweep_config_name: Sweep 설정 파일명 (확장자 제외)
            project_name: WandB 프로젝트명
            entity: WandB entity명
        """
        self.base_config_path = Path(base_config_path)
        self.sweep_config_name = sweep_config_name
        
        # 기본 설정 로딩
        self.base_config = load_config(self.base_config_path)
        
        # WandB 설정
        self.project_name = project_name or self.base_config.get('wandb', {}).get('project', 'nlp-dialogue-summarization')
        self.entity = entity or self.base_config.get('wandb', {}).get('entity')
        
        # Sweep 설정 추출
        self.sweep_config = self.base_config.get('meta', {}).get('sweep_config', {})
        
        if not self.sweep_config:
            raise ValueError(f"Sweep configuration not found for: {sweep_config_name}")
        
        # 결과 저장 디렉토리
        self.results_dir = Path(self.base_config['general']['output_dir']) / f"sweep_{sweep_config_name}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"SweepRunner initialized with config: {sweep_config_name}")
    
    def train_function(self):
        """
        단일 Sweep 실행을 위한 학습 함수
        
        WandB가 선택한 하이퍼파라미터로 모델을 학습하고 결과를 로깅
        """
        # WandB run 초기화 (Sweep agent가 자동으로 수행)
        run = wandb.run
        
        if run is None:
            raise RuntimeError("WandB run not initialized")
        
        try:
            # Sweep 파라미터 가져오기
            sweep_params = dict(wandb.config)
            
            logger.info(f"Starting sweep run: {run.id}")
            logger.info(f"Sweep parameters: {sweep_params}")
            
            # 기본 설정에 Sweep 파라미터 병합
            config = self._merge_sweep_params(sweep_params)
            
            # 실험명 생성
            experiment_name = self._generate_experiment_name(sweep_params)
            
            # 트레이너 생성
            trainer = DialogueSummarizationTrainer(
                config=config,
                sweep_mode=True,
                experiment_name=experiment_name
            )
            
            # 컴포넌트 초기화
            trainer.initialize_components()
            
            # 데이터 준비
            datasets = trainer.prepare_data()
            
            # 학습 실행
            result = trainer.train(datasets)
            
            # WandB에 최종 결과 로깅
            wandb.run.summary.update({
                'best_rouge1_f1': result.best_metrics.get('rouge1_f1', 0),
                'best_rouge2_f1': result.best_metrics.get('rouge2_f1', 0),
                'best_rougeL_f1': result.best_metrics.get('rougeL_f1', 0),
                'best_rouge_combined_f1': result.best_metrics.get('rouge_combined_f1', 0),
                'final_loss': result.final_metrics.get('eval_loss', 0),
                'model_path': result.model_path
            })
            
            # 결과 저장
            self._save_sweep_result(run.id, sweep_params, result)
            
            logger.info(f"Sweep run {run.id} completed successfully")
            logger.info(f"Best ROUGE combined F1: {result.best_metrics.get('rouge_combined_f1', 0):.4f}")
            
        except Exception as e:
            logger.error(f"Sweep run failed: {str(e)}")
            wandb.run.summary['status'] = 'failed'
            wandb.run.summary['error'] = str(e)
            raise
    
    def run_sweep(self, count: Optional[int] = None, 
                 sweep_id: Optional[str] = None,
                 resume: bool = False) -> str:
        """
        Sweep 실행
        
        Args:
            count: 실행할 실험 수 (None이면 무제한)
            sweep_id: 기존 Sweep ID (재개 시)
            resume: 기존 Sweep 재개 여부
            
        Returns:
            Sweep ID
        """
        # Sweep 설정 준비
        sweep_config = self._prepare_sweep_config()
        
        # Sweep 생성 또는 재개
        if sweep_id and resume:
            logger.info(f"Resuming sweep: {sweep_id}")
        else:
            sweep_id = wandb.sweep(
                sweep=sweep_config,
                project=self.project_name,
                entity=self.entity
            )
            logger.info(f"Created new sweep: {sweep_id}")
            
            # Sweep 정보 저장
            self._save_sweep_info(sweep_id, sweep_config)
        
        # Agent 실행
        logger.info(f"Starting sweep agent (count={count})...")
        
        try:
            wandb.agent(
                sweep_id=sweep_id,
                function=self.train_function,
                count=count,
                project=self.project_name,
                entity=self.entity
            )
        except KeyboardInterrupt:
            logger.info("Sweep interrupted by user")
        except Exception as e:
            logger.error(f"Sweep failed: {str(e)}")
            raise
        
        # 최종 결과 분석
        self._analyze_sweep_results(sweep_id)
        
        return sweep_id
    
    def _prepare_sweep_config(self) -> Dict[str, Any]:
        """
        WandB Sweep 설정 준비
        
        Returns:
            WandB Sweep 설정 딕셔너리
        """
        # 기본 구조 복사
        sweep_config = self.sweep_config.copy()
        
        # 메트릭 설정 (ROUGE 기반)
        if 'metric' not in sweep_config:
            sweep_config['metric'] = {
                'name': 'best/rouge_combined_f1',
                'goal': 'maximize'
            }
        
        # 파라미터 검증 및 조정
        if 'parameters' in sweep_config:
            params = sweep_config['parameters']
            
            # 배치 크기와 메모리 제약 조건
            if 'encoder_max_len' in params and 'per_device_train_batch_size' in params:
                # 긴 시퀀스에 대한 배치 크기 제한
                if isinstance(params['encoder_max_len'], dict) and params['encoder_max_len'].get('values'):
                    max_seq_len = max(params['encoder_max_len']['values'])
                    if max_seq_len > 1024:
                        # 배치 크기 상한 조정
                        if isinstance(params['per_device_train_batch_size'], dict):
                            if 'max' in params['per_device_train_batch_size']:
                                params['per_device_train_batch_size']['max'] = min(
                                    params['per_device_train_batch_size']['max'], 8
                                )
        
        # 프로그램 설정
        sweep_config['program'] = 'sweep_runner.py'
        
        return sweep_config
    
    def _merge_sweep_params(self, sweep_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sweep 파라미터를 기본 설정에 병합
        
        Args:
            sweep_params: WandB에서 받은 파라미터
            
        Returns:
            병합된 설정
        """
        import copy
        config = copy.deepcopy(self.base_config)
        
        # 간단한 파라미터 매핑
        param_mapping = {
            'learning_rate': ('training', 'learning_rate'),
            'per_device_train_batch_size': ('training', 'per_device_train_batch_size'),
            'per_device_eval_batch_size': ('training', 'per_device_eval_batch_size'),
            'num_train_epochs': ('training', 'num_train_epochs'),
            'warmup_ratio': ('training', 'warmup_ratio'),
            'weight_decay': ('training', 'weight_decay'),
            'encoder_max_len': ('tokenizer', 'encoder_max_len'),
            'decoder_max_len': ('tokenizer', 'decoder_max_len'),
            'num_beams': ('generation', 'num_beams'),
            'length_penalty': ('generation', 'length_penalty'),
        }
        
        for param_name, param_value in sweep_params.items():
            if param_name in param_mapping:
                section, key = param_mapping[param_name]
                if section not in config:
                    config[section] = {}
                config[section][key] = param_value
        
        return config
    
    def _generate_experiment_name(self, sweep_params: Dict[str, Any]) -> str:
        """
        Sweep 파라미터 기반 실험명 생성
        
        Args:
            sweep_params: Sweep 파라미터
            
        Returns:
            생성된 실험명
        """
        # 주요 파라미터 추출
        key_params = []
        
        # 모델 정보
        if 'model_architecture' in sweep_params:
            key_params.append(f"model_{sweep_params['model_architecture']}")
        
        # 학습률
        if 'learning_rate' in sweep_params:
            lr = sweep_params['learning_rate']
            key_params.append(f"lr_{lr:.0e}")
        
        # 배치 크기
        if 'per_device_train_batch_size' in sweep_params:
            bs = sweep_params['per_device_train_batch_size']
            key_params.append(f"bs_{bs}")
        
        # 시퀀스 길이
        if 'encoder_max_len' in sweep_params:
            seq_len = sweep_params['encoder_max_len']
            key_params.append(f"seq_{seq_len}")
        
        # 타임스탬프
        timestamp = datetime.now().strftime("%m%d_%H%M")
        
        experiment_name = f"sweep_{self.sweep_config_name}_{'_'.join(key_params)}_{timestamp}"
        
        return experiment_name
    
    def _save_sweep_info(self, sweep_id: str, sweep_config: Dict[str, Any]):
        """Sweep 정보 저장"""
        info = {
            'sweep_id': sweep_id,
            'sweep_config_name': self.sweep_config_name,
            'sweep_config': sweep_config,
            'base_config_path': str(self.base_config_path),
            'project_name': self.project_name,
            'entity': self.entity,
            'created_at': datetime.now().isoformat()
        }
        
        info_file = self.results_dir / f"sweep_{sweep_id}_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
    
    def _save_sweep_result(self, run_id: str, 
                          sweep_params: Dict[str, Any],
                          result):
        """개별 Sweep 실행 결과 저장"""
        result_data = {
            'run_id': run_id,
            'sweep_params': sweep_params,
            'best_metrics': result.best_metrics,
            'final_metrics': result.final_metrics,
            'model_path': result.model_path,
            'experiment_id': result.experiment_id,
            'timestamp': datetime.now().isoformat()
        }
        
        # 개별 결과 파일
        result_file = self.results_dir / f"run_{run_id}_result.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        # 전체 결과 파일에 추가
        all_results_file = self.results_dir / "all_sweep_results.jsonl"
        with open(all_results_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result_data, ensure_ascii=False) + '\n')
    
    def _analyze_sweep_results(self, sweep_id: str):
        """Sweep 결과 분석 및 요약"""
        logger.info("Analyzing sweep results...")
        
        # 모든 결과 로딩
        all_results_file = self.results_dir / "all_sweep_results.jsonl"
        
        if not all_results_file.exists():
            logger.warning("No results found to analyze")
            return
        
        results = []
        with open(all_results_file, 'r', encoding='utf-8') as f:
            for line in f:
                results.append(json.loads(line))
        
        if not results:
            return
        
        # 최고 성능 찾기
        best_result = max(results, key=lambda x: x['best_metrics'].get('rouge_combined_f1', 0))
        
        # 요약 통계
        rouge_scores = [r['best_metrics'].get('rouge_combined_f1', 0) for r in results]
        
        summary = {
            'sweep_id': sweep_id,
            'total_runs': len(results),
            'best_run_id': best_result['run_id'],
            'best_rouge_combined_f1': best_result['best_metrics'].get('rouge_combined_f1', 0),
            'best_params': best_result['sweep_params'],
            'best_model_path': best_result['model_path'],
            'average_rouge_combined_f1': sum(rouge_scores) / len(rouge_scores),
            'min_rouge_combined_f1': min(rouge_scores),
            'max_rouge_combined_f1': max(rouge_scores)
        }
        
        # 요약 저장
        summary_file = self.results_dir / f"sweep_{sweep_id}_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        # 콘솔 출력
        print("\n" + "="*60)
        print(f"Sweep Analysis Summary - {self.sweep_config_name}")
        print("="*60)
        print(f"Total runs: {summary['total_runs']}")
        print(f"Best ROUGE combined F1: {summary['best_rouge_combined_f1']:.4f}")
        print(f"Average ROUGE combined F1: {summary['average_rouge_combined_f1']:.4f}")
        print(f"Best run ID: {summary['best_run_id']}")
        print("\nBest hyperparameters:")
        for param, value in summary['best_params'].items():
            print(f"  {param}: {value}")
        print(f"\nBest model saved at: {summary['best_model_path']}")
        print("="*60 + "\n")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Run WandB sweep for dialogue summarization")
    
    # 필수 인자
    parser.add_argument('--base-config', type=str, required=True,
                       help='Path to base configuration file')
    parser.add_argument('--sweep-config', type=str, required=True,
                       help='Name of sweep configuration (without .yaml extension)')
    
    # 선택 인자
    parser.add_argument('--count', type=int, default=None,
                       help='Number of sweep runs (default: unlimited)')
    parser.add_argument('--project', type=str, default=None,
                       help='WandB project name')
    parser.add_argument('--entity', type=str, default=None,
                       help='WandB entity name')
    parser.add_argument('--sweep-id', type=str, default=None,
                       help='Existing sweep ID to resume')
    parser.add_argument('--resume', action='store_true',
                       help='Resume existing sweep')
    
    # 로깅 레벨
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # 로깅 설정
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # GPU 사용 가능 여부 확인
    if torch.cuda.is_available():
        logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.warning("No GPU available, training will be slow!")
    
    # SweepRunner 생성 및 실행
    try:
        runner = SweepRunner(
            base_config_path=args.base_config,
            sweep_config_name=args.sweep_config,
            project_name=args.project,
            entity=args.entity
        )
        
        sweep_id = runner.run_sweep(
            count=args.count,
            sweep_id=args.sweep_id,
            resume=args.resume
        )
        
        logger.info(f"Sweep completed: {sweep_id}")
        
    except KeyboardInterrupt:
        logger.info("Sweep interrupted by user")
    except Exception as e:
        logger.error(f"Sweep failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
