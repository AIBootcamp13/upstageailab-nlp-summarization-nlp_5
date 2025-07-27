"""
병렬 Sweep 실행 스크립트

여러 Sweep 설정을 동시에 실행하거나 단일 Sweep을 여러 워커로 병렬 실행
"""

import os
import sys
import argparse
import logging
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

logger = logging.getLogger(__name__)


class ParallelSweepRunner:
    """병렬 Sweep 실행 관리자"""
    
    def __init__(self, base_config_path: str, output_dir: str = "./sweep_results"):
        """
        초기화
        
        Args:
            base_config_path: 기본 설정 파일 경로
            output_dir: 결과 저장 디렉토리
        """
        self.base_config_path = Path(base_config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 실행 로그 디렉토리
        self.log_dir = self.output_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)
    
    def run_single_sweep_worker(self, sweep_id: str, 
                               sweep_config_name: str,
                               worker_id: int,
                               count: int = 5) -> Dict[str, Any]:
        """
        단일 Sweep 워커 실행
        
        Args:
            sweep_id: Sweep ID
            sweep_config_name: Sweep 설정명
            worker_id: 워커 ID
            count: 실행할 실험 수
            
        Returns:
            실행 결과
        """
        log_file = self.log_dir / f"worker_{worker_id}_{sweep_config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        cmd = [
            sys.executable,
            "sweep_runner.py",
            "--base-config", str(self.base_config_path),
            "--sweep-config", sweep_config_name,
            "--sweep-id", sweep_id,
            "--resume",
            "--count", str(count)
        ]
        
        logger.info(f"Starting worker {worker_id} for sweep {sweep_id}")
        
        start_time = time.time()
        
        try:
            with open(log_file, 'w') as f:
                process = subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True
                )
            
            elapsed_time = time.time() - start_time
            
            result = {
                'worker_id': worker_id,
                'sweep_id': sweep_id,
                'sweep_config_name': sweep_config_name,
                'status': 'completed' if process.returncode == 0 else 'failed',
                'return_code': process.returncode,
                'elapsed_time': elapsed_time,
                'log_file': str(log_file)
            }
            
            logger.info(f"Worker {worker_id} finished with status: {result['status']}")
            
        except Exception as e:
            result = {
                'worker_id': worker_id,
                'sweep_id': sweep_id,
                'sweep_config_name': sweep_config_name,
                'status': 'error',
                'error': str(e),
                'elapsed_time': time.time() - start_time,
                'log_file': str(log_file)
            }
            logger.error(f"Worker {worker_id} failed with error: {e}")
        
        return result
    
    def run_parallel_sweep(self, sweep_config_name: str, 
                          num_workers: int = 4,
                          runs_per_worker: int = 5,
                          create_new: bool = True) -> List[Dict[str, Any]]:
        """
        단일 Sweep을 여러 워커로 병렬 실행
        
        Args:
            sweep_config_name: Sweep 설정명
            num_workers: 워커 수
            runs_per_worker: 워커당 실행 수
            create_new: 새 Sweep 생성 여부
            
        Returns:
            워커 실행 결과 리스트
        """
        logger.info(f"Starting parallel sweep: {sweep_config_name} with {num_workers} workers")
        
        # Sweep ID 생성 또는 가져오기
        if create_new:
            sweep_id = self._create_sweep(sweep_config_name)
        else:
            sweep_id = self._get_existing_sweep_id(sweep_config_name)
        
        if not sweep_id:
            raise ValueError(f"Failed to get sweep ID for {sweep_config_name}")
        
        # 병렬 실행
        results = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            
            for worker_id in range(num_workers):
                future = executor.submit(
                    self.run_single_sweep_worker,
                    sweep_id,
                    sweep_config_name,
                    worker_id,
                    runs_per_worker
                )
                futures.append(future)
            
            # 결과 수집
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Worker failed: {e}")
                    results.append({
                        'status': 'error',
                        'error': str(e)
                    })
        
        # 결과 저장
        self._save_parallel_results(sweep_config_name, sweep_id, results)
        
        return results
    
    def run_multiple_sweeps(self, sweep_configs: List[str],
                           runs_per_sweep: int = 20) -> Dict[str, List[Dict[str, Any]]]:
        """
        여러 Sweep 설정을 순차/병렬로 실행
        
        Args:
            sweep_configs: Sweep 설정명 리스트
            runs_per_sweep: Sweep당 총 실행 수
            
        Returns:
            Sweep별 결과 딕셔너리
        """
        all_results = {}
        
        for sweep_config in sweep_configs:
            logger.info(f"Running sweep: {sweep_config}")
            
            try:
                # 각 Sweep을 순차적으로 실행 (메모리 고려)
                results = self.run_single_sweep_serial(
                    sweep_config_name=sweep_config,
                    total_runs=runs_per_sweep
                )
                all_results[sweep_config] = results
                
            except Exception as e:
                logger.error(f"Sweep {sweep_config} failed: {e}")
                all_results[sweep_config] = [{
                    'status': 'error',
                    'error': str(e)
                }]
        
        # 전체 결과 요약
        self._save_multiple_sweep_summary(all_results)
        
        return all_results
    
    def run_single_sweep_serial(self, sweep_config_name: str, 
                               total_runs: int = 20) -> List[Dict[str, Any]]:
        """
        단일 Sweep을 순차적으로 실행
        
        Args:
            sweep_config_name: Sweep 설정명
            total_runs: 총 실행 수
            
        Returns:
            실행 결과
        """
        log_file = self.log_dir / f"sweep_{sweep_config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        cmd = [
            sys.executable,
            "sweep_runner.py",
            "--base-config", str(self.base_config_path),
            "--sweep-config", sweep_config_name,
            "--count", str(total_runs)
        ]
        
        logger.info(f"Starting serial sweep: {sweep_config_name} with {total_runs} runs")
        
        start_time = time.time()
        
        try:
            with open(log_file, 'w') as f:
                process = subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True
                )
            
            elapsed_time = time.time() - start_time
            
            result = [{
                'sweep_config_name': sweep_config_name,
                'status': 'completed' if process.returncode == 0 else 'failed',
                'return_code': process.returncode,
                'elapsed_time': elapsed_time,
                'total_runs': total_runs,
                'log_file': str(log_file)
            }]
            
        except Exception as e:
            result = [{
                'sweep_config_name': sweep_config_name,
                'status': 'error',
                'error': str(e),
                'elapsed_time': time.time() - start_time,
                'log_file': str(log_file)
            }]
        
        return result
    
    def _create_sweep(self, sweep_config_name: str) -> str:
        """새 Sweep 생성"""
        # 임시 스크립트로 Sweep ID만 생성
        create_script = f"""
import wandb
from utils import load_config

# 설정 로딩
config = load_config("{self.base_config_path}")

# Sweep 설정 추출
sweep_config = config['meta']['sweep_config']

# 메트릭 설정
if 'metric' not in sweep_config:
    sweep_config['metric'] = {{
        'name': 'best/rouge_combined_f1',
        'goal': 'maximize'
    }}

# Sweep 생성
sweep_id = wandb.sweep(
    sweep=sweep_config,
    project=config.get('wandb', {{}}).get('project', 'nlp-dialogue-summarization'),
    entity=config.get('wandb', {{}}).get('entity')
)

print(sweep_id)
"""
        
        # 임시 파일로 실행
        temp_script = self.output_dir / "temp_create_sweep.py"
        with open(temp_script, 'w') as f:
            f.write(create_script)
        
        try:
            result = subprocess.run(
                [sys.executable, str(temp_script)],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                sweep_id = result.stdout.strip()
                logger.info(f"Created sweep: {sweep_id}")
                return sweep_id
            else:
                logger.error(f"Failed to create sweep: {result.stderr}")
                return None
                
        finally:
            # 임시 파일 삭제
            if temp_script.exists():
                temp_script.unlink()
    
    def _get_existing_sweep_id(self, sweep_config_name: str) -> Optional[str]:
        """기존 Sweep ID 조회"""
        # 이전 실행 정보에서 찾기
        info_files = list(self.output_dir.glob(f"sweep_*_{sweep_config_name}_info.json"))
        
        if info_files:
            # 가장 최근 파일
            latest_file = max(info_files, key=lambda p: p.stat().st_mtime)
            
            with open(latest_file, 'r') as f:
                info = json.load(f)
                return info.get('sweep_id')
        
        return None
    
    def _save_parallel_results(self, sweep_config_name: str, 
                              sweep_id: str,
                              results: List[Dict[str, Any]]):
        """병렬 실행 결과 저장"""
        summary = {
            'sweep_config_name': sweep_config_name,
            'sweep_id': sweep_id,
            'num_workers': len(results),
            'timestamp': datetime.now().isoformat(),
            'worker_results': results,
            'total_elapsed_time': max(r.get('elapsed_time', 0) for r in results),
            'successful_workers': sum(1 for r in results if r.get('status') == 'completed'),
            'failed_workers': sum(1 for r in results if r.get('status') != 'completed')
        }
        
        summary_file = self.output_dir / f"parallel_{sweep_config_name}_{sweep_id}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Parallel results saved to {summary_file}")
    
    def _save_multiple_sweep_summary(self, all_results: Dict[str, List[Dict[str, Any]]]):
        """여러 Sweep 실행 결과 요약 저장"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'num_sweeps': len(all_results),
            'sweep_summaries': {}
        }
        
        for sweep_name, results in all_results.items():
            sweep_summary = {
                'num_runs': len(results),
                'successful': sum(1 for r in results if r.get('status') == 'completed'),
                'failed': sum(1 for r in results if r.get('status') != 'completed'),
                'total_time': sum(r.get('elapsed_time', 0) for r in results)
            }
            summary['sweep_summaries'][sweep_name] = sweep_summary
        
        summary_file = self.output_dir / f"all_sweeps_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # 콘솔 출력
        print("\n" + "="*60)
        print("All Sweeps Summary")
        print("="*60)
        for sweep_name, sweep_summary in summary['sweep_summaries'].items():
            print(f"\n{sweep_name}:")
            print(f"  Total runs: {sweep_summary['num_runs']}")
            print(f"  Successful: {sweep_summary['successful']}")
            print(f"  Failed: {sweep_summary['failed']}")
            print(f"  Total time: {sweep_summary['total_time']:.2f} seconds")
        print("="*60 + "\n")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Run parallel WandB sweeps")
    
    # 필수 인자
    parser.add_argument('--base-config', type=str, required=True,
                       help='Path to base configuration file')
    
    # 실행 모드
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--single-parallel', type=str,
                           help='Run single sweep in parallel with multiple workers')
    mode_group.add_argument('--multiple-serial', nargs='+',
                           help='Run multiple sweeps serially')
    
    # 옵션
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of parallel workers (for single-parallel mode)')
    parser.add_argument('--runs-per-worker', type=int, default=5,
                       help='Runs per worker (for single-parallel mode)')
    parser.add_argument('--runs-per-sweep', type=int, default=20,
                       help='Total runs per sweep (for multiple-serial mode)')
    parser.add_argument('--output-dir', type=str, default='./sweep_results',
                       help='Output directory for results')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # 로깅 설정
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 실행기 생성
    runner = ParallelSweepRunner(
        base_config_path=args.base_config,
        output_dir=args.output_dir
    )
    
    try:
        if args.single_parallel:
            # 단일 Sweep 병렬 실행
            results = runner.run_parallel_sweep(
                sweep_config_name=args.single_parallel,
                num_workers=args.num_workers,
                runs_per_worker=args.runs_per_worker
            )
            logger.info(f"Parallel sweep completed with {len(results)} workers")
            
        else:
            # 여러 Sweep 순차 실행
            results = runner.run_multiple_sweeps(
                sweep_configs=args.multiple_serial,
                runs_per_sweep=args.runs_per_sweep
            )
            logger.info(f"Completed {len(results)} sweeps")
            
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user")
    except Exception as e:
        logger.error(f"Execution failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
