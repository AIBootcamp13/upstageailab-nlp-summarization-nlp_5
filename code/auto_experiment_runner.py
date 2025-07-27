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

# code 디렉토리를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

from utils.path_utils import PathManager
from utils.device_utils import get_optimal_device, setup_device_config
from utils.experiment_utils import ExperimentTracker, ModelRegistry
from utils.config_manager import ConfigManager

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
        
        self.base_config_path = PathManager.resolve_path(base_config_path)
        self.output_dir = PathManager.ensure_dir(output_dir)
        
        # 디바이스 자동 감지
        self.device = get_optimal_device()
        
        # 실험 추적 초기화
        self.tracker = ExperimentTracker(f"{output_dir}/experiments")
        self.registry = ModelRegistry(f"{output_dir}/models")
        
        # 로깅 설정
        self.logger = self._setup_logger()
        
        print(f"🚀 자동 실험 실행기 초기화")
        print(f"   디바이스: {self.device}")
        print(f"   출력 디렉토리: {output_dir}")
    
    def _setup_logger(self) -> logging.Logger:
        """로거 설정"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # 로그 파일 핸들러
        log_file = PathManager.ensure_dir("logs") / "auto_experiments.log"
        file_handler = logging.FileHandler(log_file)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def discover_experiment_configs(self, config_dir: str = "config/experiments") -> List[Path]:
        """
        실험 설정 파일들을 자동 발견
        
        Args:
            config_dir: 실험 설정 디렉토리 (상대 경로)
            
        Returns:
            발견된 YAML 설정 파일 목록
        """
        if Path(config_dir).is_absolute():
            raise ValueError(f"설정 디렉토리는 상대 경로여야 합니다: {config_dir}")
        
        config_path = PathManager.resolve_path(config_dir)
        
        if not config_path.exists():
            self.logger.warning(f"설정 디렉토리가 없습니다: {config_dir}")
            return []
        
        # YAML 파일 검색
        yaml_files = []
        for pattern in ['*.yaml', '*.yml']:
            yaml_files.extend(config_path.glob(pattern))
        
        # 파일명으로 정렬 (실행 순서 보장)
        yaml_files.sort(key=lambda x: x.name)
        
        self.logger.info(f"발견된 실험 설정: {len(yaml_files)}개")
        for file in yaml_files:
            self.logger.info(f"  - {file.relative_to(PathManager.get_project_root())}")
        
        return yaml_files
    
    def load_experiment_config(self, config_path: Path) -> Dict[str, Any]:
        """
        실험 설정 로딩 및 디바이스 최적화 적용
        
        Args:
            config_path: 설정 파일 절대 경로
            
        Returns:
            디바이스 최적화가 적용된 설정
        """
        # 상대 경로로 변환
        relative_path = config_path.relative_to(PathManager.get_project_root())
        
        # 기본 설정 로딩
        base_manager = ConfigManager(self.base_config_path.relative_to(PathManager.get_project_root()))
        base_config = base_manager.get_config()
        
        # 실험별 설정 로딩
        exp_manager = ConfigManager(str(relative_path))
        exp_config = exp_manager.get_config()
        
        # 설정 병합 (실험 설정이 우선)
        merged_config = base_manager.merge_configs(base_config, exp_config)
        
        # 디바이스 최적화 적용
        optimized_config = setup_device_config(merged_config)
        
        self.logger.info(f"설정 로딩 완료: {relative_path}")
        self.logger.info(f"디바이스 최적화 적용: {self.device}")
        
        return optimized_config
    
    def run_single_experiment(self, 
                            config_path: Path, 
                            experiment_name: Optional[str] = None) -> Dict[str, Any]:
        """
        단일 실험 실행
        
        Args:
            config_path: 설정 파일 경로
            experiment_name: 실험 이름 (None이면 파일명 사용)
            
        Returns:
            실험 결과 요약
        """
        if experiment_name is None:
            experiment_name = config_path.stem
        
        self.logger.info(f"실험 시작: {experiment_name}")
        
        try:
            # 설정 로딩
            config = self.load_experiment_config(config_path)
            
            # 실험 추적 시작
            exp_id = self.tracker.start_experiment(
                name=experiment_name,
                description=f"자동 실험: {config_path.name}",
                config=config,
                device=self.device,
                config_file=str(config_path.relative_to(PathManager.get_project_root()))
            )
            
            # 실제 학습 실행
            result = self._execute_training(config, exp_id)
            
            # 실험 종료
            final_metrics = {
                'best_rouge_combined_f1': result.get('best_rouge_combined_f1', 0),
                'training_time_minutes': result.get('training_time_minutes', 0),
                'total_epochs': result.get('total_epochs', 0)
            }
            
            experiment_summary = self.tracker.end_experiment(
                exp_id, 
                final_metrics, 
                "completed"
            )
            
            # 모델 등록 (성능이 좋은 경우)
            if result.get('best_rouge_combined_f1', 0) > 0.3:  # 임계값
                model_id = self.registry.register_model(
                    name=f"{experiment_name}_model",
                    architecture=config.get('model', {}).get('name', 'unknown'),
                    config=config,
                    performance=final_metrics,
                    model_path=result.get('model_path', ''),
                    experiment_id=exp_id
                )
                result['model_id'] = model_id
            
            self.logger.info(f"실험 완료: {experiment_name} (ROUGE: {final_metrics['best_rouge_combined_f1']:.4f})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"실험 실패: {experiment_name} - {e}")
            
            # 실패한 실험도 추적
            try:
                self.tracker.end_experiment(exp_id, {}, "failed")
            except:
                pass
            
            return {
                'experiment_name': experiment_name,
                'status': 'failed',
                'error': str(e)
            }
    
    def _execute_training(self, config: Dict[str, Any], exp_id: str) -> Dict[str, Any]:
        """
        실제 학습 실행 (trainer.py 호출)
        
        Args:
            config: 실험 설정
            exp_id: 실험 ID
            
        Returns:
            학습 결과
        """
        # 임시 설정 파일 생성
        temp_config_path = self.output_dir / f"temp_config_{exp_id}.yaml"
        
        import yaml
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        try:
            # trainer.py 실행
            cmd = [
                sys.executable,
                str(PathManager.resolve_path("code/trainer.py")),
                "--config", str(temp_config_path.relative_to(PathManager.get_project_root())),
                "--experiment-name", f"auto_exp_{exp_id[:8]}",
                "--device", self.device
            ]
            
            start_time = time.time()
            
            # subprocess로 실행
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=PathManager.get_project_root(),
                timeout=7200  # 2시간 타임아웃
            )
            
            end_time = time.time()
            training_time = (end_time - start_time) / 60  # 분 단위
            
            if result.returncode == 0:
                # 성공 시 결과 파싱
                return self._parse_training_result(result.stdout, training_time)
            else:
                raise RuntimeError(f"Training failed: {result.stderr}")
        
        finally:
            # 임시 파일 정리
            if temp_config_path.exists():
                temp_config_path.unlink()
    
    def _parse_training_result(self, stdout: str, training_time: float) -> Dict[str, Any]:
        """학습 결과 파싱"""
        # 간단한 결과 파싱 (실제로는 더 정교하게 구현)
        result = {
            'training_time_minutes': training_time,
            'status': 'completed'
        }
        
        # ROUGE 점수 추출 (로그에서)
        lines = stdout.split('\n')
        for line in lines:
            if 'rouge_combined_f1' in line.lower():
                try:
                    # 예: "ROUGE Combined F1: 0.4567"
                    score = float(line.split(':')[-1].strip())
                    result['best_rouge_combined_f1'] = score
                except:
                    pass
        
        return result
    
    def run_all_experiments(self, 
                          config_dir: str = "config/experiments",
                          max_parallel: int = 1) -> Dict[str, Any]:
        """
        모든 실험 순차 실행
        
        Args:
            config_dir: 실험 설정 디렉토리 (상대 경로)
            max_parallel: 최대 병렬 실행 수 (현재는 1만 지원)
            
        Returns:
            전체 실험 결과 요약
        """
        print(f"🔍 실험 설정 검색 중: {config_dir}")
        config_files = self.discover_experiment_configs(config_dir)
        
        if not config_files:
            print(f"❌ 실험 설정 파일이 없습니다: {config_dir}")
            return {'status': 'no_configs', 'results': []}
        
        print(f"📋 총 {len(config_files)}개 실험을 순차 실행합니다")
        
        overall_start_time = time.time()
        all_results = []
        
        for i, config_file in enumerate(config_files, 1):
            print(f"\n🚀 실험 {i}/{len(config_files)}: {config_file.name}")
            
            experiment_name = f"{i:02d}_{config_file.stem}"
            result = self.run_single_experiment(config_file, experiment_name)
            all_results.append(result)
            
            # 실험 간 휴식 (리소스 정리)
            if i < len(config_files):
                print("⏱️ 다음 실험 준비 중... (30초 대기)")
                time.sleep(30)
        
        overall_end_time = time.time()
        total_time = (overall_end_time - overall_start_time) / 3600  # 시간 단위
        
        # 전체 결과 요약
        summary = self._generate_experiment_summary(all_results, total_time)
        
        # 결과 저장
        summary_file = self.output_dir / "experiment_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n🎉 모든 실험 완료!")
        print(f"⏱️ 총 소요 시간: {total_time:.2f}시간")
        print(f"📄 결과 요약: {summary_file.relative_to(PathManager.get_project_root())}")
        
        return summary
    
    def _generate_experiment_summary(self, 
                                   results: List[Dict[str, Any]], 
                                   total_time: float) -> Dict[str, Any]:
        """전체 실험 결과 요약 생성"""
        successful_results = [r for r in results if r.get('status') != 'failed']
        failed_results = [r for r in results if r.get('status') == 'failed']
        
        summary = {
            'execution_info': {
                'total_experiments': len(results),
                'successful_experiments': len(successful_results),
                'failed_experiments': len(failed_results),
                'total_time_hours': total_time,
                'device_used': self.device,
                'execution_date': datetime.now().isoformat()
            },
            'performance_summary': {},
            'best_experiment': None,
            'all_results': results
        }
        
        # 성능 요약
        if successful_results:
            rouge_scores = [r.get('best_rouge_combined_f1', 0) for r in successful_results]
            
            summary['performance_summary'] = {
                'best_rouge_score': max(rouge_scores),
                'average_rouge_score': sum(rouge_scores) / len(rouge_scores),
                'worst_rouge_score': min(rouge_scores)
            }
            
            # 최고 성능 실험 찾기
            best_idx = rouge_scores.index(max(rouge_scores))
            summary['best_experiment'] = successful_results[best_idx]
        
        return summary
    
    def create_sample_configs(self, output_dir: str = "config/experiments"):
        """샘플 실험 설정 파일들 생성"""
        if Path(output_dir).is_absolute():
            raise ValueError(f"출력 디렉토리는 상대 경로여야 합니다: {output_dir}")
        
        config_dir = PathManager.ensure_dir(output_dir)
        
        sample_configs = {
            "01_baseline.yaml": {
                "experiment_name": "baseline",
                "model": {"name": "gogamza/kobart-base-v2"},
                "training": {
                    "learning_rate": 0.0001,
                    "per_device_train_batch_size": 8,
                    "num_train_epochs": 5
                }
            },
            "02_high_lr.yaml": {
                "experiment_name": "high_learning_rate",
                "model": {"name": "gogamza/kobart-base-v2"},
                "training": {
                    "learning_rate": 0.0005,
                    "per_device_train_batch_size": 8,
                    "num_train_epochs": 5
                }
            },
            "03_large_batch.yaml": {
                "experiment_name": "large_batch_size",
                "model": {"name": "gogamza/kobart-base-v2"},
                "training": {
                    "learning_rate": 0.0001,
                    "per_device_train_batch_size": 16,
                    "num_train_epochs": 5
                }
            },
            "04_longer_training.yaml": {
                "experiment_name": "longer_training",
                "model": {"name": "gogamza/kobart-base-v2"},
                "training": {
                    "learning_rate": 0.0001,
                    "per_device_train_batch_size": 8,
                    "num_train_epochs": 10
                }
            }
        }
        
        import yaml
        for filename, config in sample_configs.items():
            file_path = config_dir / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
            print(f"✅ 샘플 설정 생성: {file_path.relative_to(PathManager.get_project_root())}")
        
        print(f"\n📁 총 {len(sample_configs)}개 샘플 설정 파일 생성 완료")
        print(f"🚀 실행 방법: python code/auto_experiment_runner.py --run-all")


def main():
    """CLI 인터페이스"""
    import argparse
    
    parser = argparse.ArgumentParser(description="자동 실험 실행 시스템")
    parser.add_argument('--base-config', default="config/base_config.yaml",
                       help='기본 설정 파일 경로 (상대 경로)')
    parser.add_argument('--config-dir', default="config/experiments",
                       help='실험 설정 디렉토리 (상대 경로)')
    parser.add_argument('--output-dir', default="outputs/auto_experiments",
                       help='결과 저장 디렉토리 (상대 경로)')
    parser.add_argument('--run-all', action='store_true',
                       help='모든 실험 순차 실행')
    parser.add_argument('--create-samples', action='store_true',
                       help='샘플 설정 파일들 생성')
    parser.add_argument('--experiment', type=str,
                       help='특정 실험 하나만 실행 (파일명)')
    
    args = parser.parse_args()
    
    try:
        runner = AutoExperimentRunner(
            base_config_path=args.base_config,
            output_dir=args.output_dir
        )
        
        if args.create_samples:
            runner.create_sample_configs(args.config_dir)
        
        elif args.run_all:
            runner.run_all_experiments(args.config_dir)
        
        elif args.experiment:
            config_path = PathManager.resolve_path(f"{args.config_dir}/{args.experiment}")
            if not config_path.exists():
                print(f"❌ 설정 파일을 찾을 수 없습니다: {config_path}")
                return 1
            
            result = runner.run_single_experiment(config_path)
            print(f"실험 결과: {result}")
        
        else:
            print("❌ 실행할 작업을 지정하세요 (--run-all, --create-samples, --experiment)")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"❌ 실행 실패: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
