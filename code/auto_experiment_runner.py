#!/usr/bin/env python3
"""
자동 실험 실행 시스템 (상대 경로 기준, MPS/CUDA 최적화)

여러 YAML 설정을 순차적으로 자동 실행하는 시스템
- 상대 경로 기준 파일 처리
- MPS/CUDA 디바이스 자동 감지 및 최적화
- 실험 결과 자동 추적 및 분석
"""

import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import subprocess
import logging

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# code 디렉토리의 utils 임포트
# code 디렉토리의 utils 임포트
from utils.path_utils import PathManager, path_manager
from utils.device_utils import get_optimal_device, setup_device_config
from utils.experiment_utils import ExperimentTracker, ModelRegistry
from utils.csv_results_saver import CSVResultsSaver
from utils import load_config


class AutoExperimentRunner:
    """자동 실험 실행 관리자 (상대 경로 기준)"""
    def __init__(self, 
                 base_config_path: str = "config/base_config.yaml",
                 output_dir: str = "outputs/auto_experiments"):
        """
        Args:
            base_config_path: 기본 설정 파일 경로 (상대 경로)
            output_dir: 실험 결과 저장 디렉토리 (상대 경로)
        """
        # 상대 경로 확인
        if Path(base_config_path).is_absolute() or Path(output_dir).is_absolute():
            raise ValueError("모든 경로는 상대 경로여야 합니다")
        
        self.base_config_path = path_manager.resolve_path(base_config_path)
        self.output_dir = path_manager.ensure_dir(output_dir)
        
        # 디바이스 자동 감지
        device_tuple = get_optimal_device()
        if isinstance(device_tuple, tuple):
            self.device = device_tuple[0]  # torch.device 객체
            self.device_info = device_tuple[1] if len(device_tuple) > 1 else None
        else:
            self.device = device_tuple
            self.device_info = None
        
        # 실험 추적 초기화
        self.tracker = ExperimentTracker(f"{output_dir}/experiments")
        print(f"\n🆗 ExperimentTracker 초기화 완료")
        print(f"   log_experiment 메서드 존재: {hasattr(self.tracker, 'log_experiment')}")
        self.registry = ModelRegistry(f"{output_dir}/models")
        
        # CSV 결과 저장기 초기화
        self.csv_saver = CSVResultsSaver(f"{output_dir}/csv_results")
        
        # 로깅 설정
        self.logger = self._setup_logger()
        
        print(f"🚀 자동 실험 실행기 초기화")
        print(f"   디바이스: {self.device}")
        if self.device_info and hasattr(self.device_info, 'device_name'):
            print(f"   GPU 정보: {self.device_info.device_name} ({self.device_info.memory_gb:.1f}GB)")
        print(f"   출력 디렉토리: {output_dir}")
    
    def _setup_logger(self) -> logging.Logger:
        log_file = path_manager.ensure_dir("logs") / "auto_experiments.log"
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # 로그 파일 핸들러
        log_file = path_manager.ensure_dir("logs") / "auto_experiments.log"
        file_handler = logging.FileHandler(log_file)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def run_experiments(self, 
                       experiment_configs: List[str],
                       dry_run: bool = False,
                       continue_on_error: bool = True,
                       one_epoch: bool = False,
                       disable_eval: bool = False) -> Dict[str, Any]:
        """
        여러 실험을 순차적으로 실행
        
        Args:
            experiment_configs: 실험 설정 파일 경로 리스트 (상대 경로)
            dry_run: 실제 실행 없이 설정만 확인
            continue_on_error: 오류 발생 시 다음 실험 계속 진행
            one_epoch: 1에포크만 실행 (빠른 테스트용)
            disable_eval: 평가 비활성화 (1에포크 모드용)
            
        Returns:
            실험 결과 딕셔너리
        """
        results = {}
        
        for i, config_path in enumerate(experiment_configs):
            try:
                print(f"\n{'='*60}")
                print(f"실험 {i+1}/{len(experiment_configs)}: {config_path}")
                print(f"{'='*60}")
                
                # 상대 경로 확인
                if Path(config_path).is_absolute():
                    raise ValueError(f"설정 경로는 상대 경로여야 합니다: {config_path}")
                
                # 설정 로드
                full_config = self._load_and_merge_config(config_path)
                
                # 디바이스 설정 적용
                self._apply_device_config(full_config)
                
                # WandB 환경 설정 (미리 설정하여 trainer.py에서 자동 활용)
                wandb_enabled = self.setup_wandb_environment(full_config)
                
                if dry_run:
                    print("\n[DRY RUN] 설정 내용:")
                    print(json.dumps(full_config, indent=2, ensure_ascii=False))
                    results[config_path] = {"status": "dry_run", "config": full_config}
                    continue
                
                # 실험 실행
                result = self._run_single_experiment(full_config, config_path, one_epoch, disable_eval)
                results[config_path] = result
                
                # 실험 추적 - try-except 블록 추가
                try:
                    self.tracker.log_experiment(
                        experiment_name=Path(config_path).stem,
                        config=full_config,
                        results=result
                    )
                except Exception as e:
                    self.logger.warning(f"실험 로그 기록 실패: {e}")
                    # 로그 실패가 전체 실행을 중단하지 않도록 함
                
                # 실험 간 대기 (GPU 메모리 정리 등)
                if i < len(experiment_configs) - 1:
                    print("\n다음 실험 준비 중...")
                    time.sleep(5)
                    
            except Exception as e:
                self.logger.error(f"실험 실행 중 오류 발생: {config_path}", exc_info=True)
                results[config_path] = {"status": "error", "error": str(e)}
                
                if not continue_on_error:
                    raise
        
        # 전체 결과 요약
        self._print_summary(results)
        
        # 전체 결과를 CSV로 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_csv = self.csv_saver.save_batch_results(
            results=results,
            output_filename=f"experiment_summary_{timestamp}.csv"
        )
        print(f"\n실험 결과 CSV 저장: {summary_csv}")
        
        return results
    
    def _load_and_merge_config(self, config_path: str) -> Dict[str, Any]:
        """기본 설정과 실험 설정을 병합"""
        # 기본 설정 로드
        base_config = load_config(self.base_config_path)
        
        # 실험 설정 로드
        exp_config_path = path_manager.resolve_path(config_path)
        exp_config = load_config(exp_config_path)
        
        # 설정 병합 (실험 설정이 우선)
        merged = self._deep_merge(base_config, exp_config)
        
        return merged
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """딕셔너리 깊은 병합"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def setup_wandb_environment(self, config: Dict[str, Any]) -> bool:
        """
        실험별 WandB 환경 설정
        
        Args:
            config: 실험 설정 딕셔너리
            
        Returns:
            WandB 활성화 여부
        """
        import os
        
        # 항상 WANDB_LOG_MODEL=end 설정 (best model만 저장)
        os.environ["WANDB_LOG_MODEL"] = "end"
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        
        # report_to 설정 확인
        report_to = config.get('training', {}).get('report_to', 'wandb')
        
        if report_to in ['all', 'wandb']:
            print("✅ WandB 활성화: best model artifacts 자동 저장")
            # WandB 활성화를 위한 환경 세팅
            if "WANDB_MODE" in os.environ:
                del os.environ["WANDB_MODE"]
            
            # 실험을 식별할 수 있는 고유한 설정 추가
            experiment_path = config.get('__config_path__', 'unknown')
            if experiment_path and experiment_path != 'unknown':
                # 실험 이름을 WandB config에 추가
                config['wandb_experiment_name'] = Path(experiment_path).stem
            
            return True
        else:
            print(f"⚠️ WandB 비활성화 (report_to={report_to})")
            os.environ["WANDB_MODE"] = "disabled"
            return False
    
    def _apply_device_config(self, config: Dict[str, Any]) -> None:
        """디바이스별 최적화 설정 적용"""
        if not self.device_info:
            return
            return
            
        # 모델 크기 추정
        model_name = config.get('general', {}).get('model_name', '')
        if 'large' in model_name.lower() or 'xl' in model_name.lower():
            model_size = 'large'
        elif 'small' in model_name.lower() or 'tiny' in model_name.lower():
            model_size = 'small'
        else:
            model_size = 'base'
        
        # 최적화 설정 생성
        opt_config = setup_device_config(self.device_info, model_size)
        
        # training 섹션에 적용
        if 'training' not in config:
            config['training'] = {}
        
        # 기존 설정을 유지하면서 디바이스 최적화 설정 추가
        training_config = config['training']
        opt_dict = opt_config.to_dict()
        
        for key, value in opt_dict.items():
            if key not in training_config:
                training_config[key] = value
        config['device'] = str(self.device)
        config['device_info'] = {
            'type': self.device_info.device_type,
            'name': self.device_info.device_name,
            'memory_gb': self.device_info.memory_gb
        } if hasattr(self.device_info, 'device_type') else None
    
    def _run_single_experiment(self, config: Dict[str, Any], config_path: str, one_epoch: bool = False, disable_eval: bool = False) -> Dict[str, Any]:
        """단일 실험 실행"""
        print(f"\n🔧 _run_single_experiment 시작: {config_path}")
        start_time = time.time()
        
        try:
            # 항상 환경 변수 복사
            import os
            env = os.environ.copy()
            
            # WandB 환경 설정
            wandb_enabled = self.setup_wandb_environment(config)
            
            # 한국 시간 기반 실험 ID 생성
            try:
                from utils.experiment_utils import get_korean_time_format
                korean_time = get_korean_time_format('MMDDHHMM')
                experiment_name = config.get('experiment_name', Path(config_path).stem)
                experiment_id = f"{experiment_name}_{korean_time}"
                print(f"🔍 실험 ID: {experiment_id}")
            except ImportError as e:
                print(f"⚠️ 한국 시간 유틸리티 import 실패: {e}")
                experiment_id = Path(config_path).stem
            
            # 1에포크 모드를 위한 환경 변수 설정
            if one_epoch:
                env['FORCE_ONE_EPOCH'] = '1'
                print(f"\n🚀 1에포크 모드로 실행: {Path(config_path).stem}")
            
            # trainer.py 실행
            cmd = [
                sys.executable,
                str(path_manager.resolve_path("code/trainer.py")),
                "--config", config_path
            ]
            
            # 1에포크 모드 옵션 추가
            if one_epoch:
                cmd.append("--one-epoch")
            
            # 평가 비활성화 옵션 추가
            if disable_eval:
                cmd.append("--disable-eval")
            
            print(f"\n실행 명령: {' '.join(cmd)}")
            print(f"현재 디렉토리: {os.getcwd()}")
            print(f"Python 경로: {sys.executable}")
            
            # 프로세스 실행
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env
            )
            
            # 모든 출력을 수집하면서 실시간 표시
            output_lines = []
            for line in process.stdout:
                print(line, end='')
                output_lines.append(line)
            
            # 프로세스 종료 대기
            process.wait()
            
            # 결과 수집
            if process.returncode == 0:
                
                # 🆕 학습 완료 후 test.csv 추론 수행
                print(f"\n📊 Test 추론 시작: {experiment_id}")
                
                try:
                    # 베스트 체크포인트 찾기
                    output_dir = Path(config.get('training', {}).get('output_dir', 'outputs'))
                    checkpoint_dirs = list(output_dir.glob('checkpoint-*'))
                    
                    if checkpoint_dirs:
                        # 가장 최근 체크포인트 선택
                        best_checkpoint = max(checkpoint_dirs, key=lambda p: p.stat().st_mtime)
                        print(f"🎯 베스트 체크포인트: {best_checkpoint}")
                        
                        # post_training_inference 활용
                        try:
                            from post_training_inference import generate_submission_after_training
                            
                            submission_path = generate_submission_after_training(
                                experiment_name=experiment_id,
                                model_path=str(best_checkpoint),
                                config_dict=config
                            )
                            
                            print(f"✅ 제출 파일 생성 완료: {submission_path}")
                            result = self._collect_results(config, Path(config_path).stem)
                            result['submission_path'] = submission_path
                            
                        except ImportError as ie:
                            print(f"⚠️ post_training_inference import 실패: {ie}")
                            # 대안: run_inference.py 직접 사용
                            try:
                                inference_cmd = [
                                    sys.executable,
                                    str(path_manager.resolve_path("code/run_inference.py")),
                                    "--model_path", str(best_checkpoint),
                                    "--input_file", "data/test.csv",
                                    "--output_file", f"outputs/auto_experiments/{experiment_id}_submission.csv",
                                    "--batch_size", "16"
                                ]
                                
                                print(f"🔄 대안 추론 실행: {' '.join(inference_cmd)}")
                                
                                inference_process = subprocess.run(
                                    inference_cmd,
                                    capture_output=True,
                                    text=True,
                                    env=env
                                )
                                
                                if inference_process.returncode == 0:
                                    submission_path = f"outputs/auto_experiments/{experiment_id}_submission.csv"
                                    print(f"✅ 대안 추론 성공: {submission_path}")
                                    result = self._collect_results(config, Path(config_path).stem)
                                    result['submission_path'] = submission_path
                                else:
                                    print(f"❌ 대안 추론 실패: {inference_process.stderr}")
                                    result = self._collect_results(config, Path(config_path).stem)
                                    result['inference_error'] = inference_process.stderr
                                    
                            except Exception as alt_e:
                                print(f"❌ 대안 추론 예외: {alt_e}")
                                result = self._collect_results(config, Path(config_path).stem)
                                result['inference_error'] = str(alt_e)
                                
                    else:
                        print("⚠️ 체크포인트를 찾을 수 없습니다.")
                        result = self._collect_results(config, Path(config_path).stem)
                        result['inference_error'] = "No checkpoint found"
                        
                except Exception as inf_e:
                    print(f"❌ 추론 실행 중 예외: {inf_e}")
                    result = self._collect_results(config, Path(config_path).stem)
                    result['inference_error'] = str(inf_e)
                result['status'] = 'success'
                result['duration'] = time.time() - start_time
            else:
                # 에러 시 출력 내용도 포함
                error_output = '\n'.join(output_lines[-50:])  # 마지막 50줄만
                result = {
                    'status': 'error',
                    'error': f'Training failed with exit code {process.returncode}\n\nLast output:\n{error_output}',
                    'duration': time.time() - start_time
                }
        
        except Exception as e:
            result = {
                'status': 'error',
                'error': str(e),
                'duration': time.time() - start_time
            }
        
        return result
    
    def _collect_results(self, config: Dict[str, Any], experiment_name: str) -> Dict[str, Any]:
        """실험 결과 수집"""
        results = {
            'experiment_name': experiment_name,
            'model_name': config.get('general', {}).get('model_name', 'unknown')
        }
        
        # 결과 파일 경로
        output_dir = Path(config.get('training', {}).get('output_dir', 'outputs'))
        
        # 메트릭 파일 읽기
        metrics_file = output_dir / 'eval_results.json'
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                results['metrics'] = metrics
        
        
        # 베스트 모델 정보
        checkpoint_dirs = list(output_dir.glob('checkpoint-*'))
        if checkpoint_dirs:
            results['best_checkpoint'] = str(max(checkpoint_dirs, key=lambda p: p.stat().st_mtime))
        
        # CSV 결과 저장
        if 'metrics' in results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = self.csv_saver.save_experiment_results(
                experiment_name=experiment_name,
                config=config,
                metrics=results['metrics'],
                timestamp=timestamp
            )
            results['csv_path'] = str(csv_path)
        
        return results
    def _print_summary(self, results: Dict[str, Dict[str, Any]]) -> None:
        """실험 결과 요약 출력"""
        print("\n" + "="*60)
        print("실험 결과 요약")
        print("="*60)
        
        success_count = sum(1 for r in results.values() if r.get('status') == 'success')
        total_count = len(results)
        
        print(f"\n총 실험: {total_count}")
        print(f"성공: {success_count}")
        print(f"실패: {total_count - success_count}")
        
        # 성공한 실험의 메트릭 비교
        print("\n메트릭 비교:")
        print(f"{'실험명':<30} {'ROUGE-1':<10} {'ROUGE-2':<10} {'ROUGE-L':<10}")
        print("-" * 60)
        
        for config_path, result in results.items():
            exp_name = Path(config_path).stem[:30]
            if result.get('status') == 'success' and 'metrics' in result:
                metrics = result['metrics']
                rouge1 = metrics.get('eval_rouge1', 0)
                rouge2 = metrics.get('eval_rouge2', 0)
                rougeL = metrics.get('eval_rougeL', 0)
                print(f"{exp_name:<30} {rouge1:<10.4f} {rouge2:<10.4f} {rougeL:<10.4f}")
            else:
                status = result.get('status', 'unknown')
                print(f"{exp_name:<30} {status}")
        
        # 최고 성능 모델
        best_model = None
        best_score = 0
        
        for config_path, result in results.items():
            if result.get('status') == 'success' and 'metrics' in result:
                metrics = result['metrics']
                score = (metrics.get('eval_rouge1', 0) + 
                        metrics.get('eval_rouge2', 0) + 
                        metrics.get('eval_rougeL', 0)) / 3
                if score > best_score:
                    best_score = score
                    best_model = Path(config_path).stem
        
        if best_model:
            print(f"\n최고 성능 모델: {best_model} (평균 ROUGE: {best_score:.4f})")
    
    def run_single_config(self, config_path: str, dry_run: bool = False) -> Dict[str, Any]:
        """단일 설정 파일로 실험 실행"""
        return self.run_experiments([config_path], dry_run=dry_run)


def main():
    """CLI 진입점"""
    import argparse
    
    parser = argparse.ArgumentParser(description="자동 실험 실행 시스템")
    parser.add_argument(
        '--config', '-c',
        type=str,
        nargs='+',
        help='실험 설정 파일 경로 (상대 경로, 여러 개 가능)'
    )
    parser.add_argument(
        '--base-config',
        type=str,
        default='config/base_config.yaml',
        help='기본 설정 파일 경로 (기본값: config/base_config.yaml)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/auto_experiments',
        help='실험 결과 저장 디렉토리 (기본값: outputs/auto_experiments)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='실제 실행 없이 설정만 확인'
    )
    parser.add_argument(
        '--stop-on-error',
        action='store_true',
        help='오류 발생 시 중단 (기본값: 계속 진행)'
    )
    parser.add_argument(
        '--one-epoch',
        action='store_true',
        help='1에포크만 실행 (빠른 테스트용)'
    )
    parser.add_argument(
        '--disable-eval',
        action='store_true',
        help='평가 비활성화 (1에포크 모드용)'
    )
    
    args = parser.parse_args()
    
    if not args.config:
        # 기본 실험 세트
        default_configs = [
            "config/experiments/01_baseline.yaml",
            "config/experiments/02_simple_augmentation.yaml",
            "config/experiments/03_high_learning_rate.yaml"
        ]
        print(f"설정 파일이 지정되지 않았습니다. 기본 실험을 실행합니다:")
        for config in default_configs:
            print(f"  - {config}")
        args.config = default_configs
    
    # 실행기 초기화
    runner = AutoExperimentRunner(
        base_config_path=args.base_config,
        output_dir=args.output_dir
    )
    
    # 실험 실행
    results = runner.run_experiments(
        experiment_configs=args.config,
        dry_run=args.dry_run,
        continue_on_error=not args.stop_on_error,
        one_epoch=args.one_epoch,
        disable_eval=args.disable_eval
    )
    
    # 결과 저장
    if not args.dry_run:
        result_file = Path(args.output_dir) / f"experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n실험 결과 저장: {result_file}")
    
    # 성공 여부 반환
    success_count = sum(1 for r in results.values() if r.get('status') == 'success')
    return 0 if success_count == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
