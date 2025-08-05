"""
CSV 결과 저장 유틸리티

실험 결과를 baseline.ipynb와 동일한 형태의 CSV로 저장
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class CSVResultsSaver:
    """실험 결과를 CSV 형태로 저장하는 클래스"""
    
    def __init__(self, output_dir: str = "outputs/experiment_results"):
        """
        Args:
            output_dir: 결과 저장 디렉토리
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_experiment_results(self, experiment_name: str, config: Dict[str, Any], 
                              metrics: Dict[str, float], timestamp: Optional[str] = None,
                              submission_path: Optional[str] = None) -> Path:
        """
        단일 실험 결과를 CSV로 저장
        
        Args:
            experiment_name: 실험명
            config: 실험 설정
            metrics: 평가 메트릭
            timestamp: 타임스탬프 (없으면 자동 생성)
            submission_path: 제출 파일 경로
            
        Returns:
            저장된 CSV 파일 경로
        """
        if not timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 결과 데이터 구성
        result_data = {
            'experiment_name': experiment_name,
            'timestamp': timestamp,
            'model_name': config.get('general', {}).get('model_name', 'unknown'),
            'num_train_epochs': config.get('training', {}).get('num_train_epochs', 0),
            'learning_rate': config.get('training', {}).get('learning_rate', 0),
            'batch_size': config.get('training', {}).get('per_device_train_batch_size', 0),
            'rouge1': metrics.get('eval_rouge1', 0),
            'rouge2': metrics.get('eval_rouge2', 0),
            'rougeL': metrics.get('eval_rougeL', 0),
            'rouge1_f1': metrics.get('eval_rouge1_f1', 0),
            'rouge2_f1': metrics.get('eval_rouge2_f1', 0),
            'rougeL_f1': metrics.get('eval_rougeL_f1', 0),
            'rouge_combined': metrics.get('eval_rouge_combined', 0),
            'rouge_combined_f1': metrics.get('eval_rouge_combined_f1', 0),
            'eval_loss': metrics.get('eval_loss', 0),
            'config_path': str(config.get('__config_path__', 'N/A')),
            'submission_path': submission_path or 'N/A'  # 제출 파일 경로 추가
        }
        
        # DataFrame 생성 및 저장
        df = pd.DataFrame([result_data])
        csv_path = self.output_dir / f"{experiment_name}_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        
        logger.info(f"실험 결과 저장: {csv_path}")
        return csv_path
    
    def save_batch_results(self, results: Dict[str, Dict[str, Any]], 
                          output_filename: str = "experiment_summary.csv") -> Path:
        """
        여러 실험 결과를 하나의 CSV로 저장
        
        Args:
            results: 실험 결과 딕셔너리 {config_path: result_dict}
            output_filename: 출력 파일명
            
        Returns:
            저장된 CSV 파일 경로
        """
        all_results = []
        
        for config_path, result in results.items():
            if result.get('status') == 'success' and 'metrics' in result:
                metrics = result['metrics']
                experiment_name = Path(config_path).stem
                
                row_data = {
                    'experiment_name': experiment_name,
                    'config_path': config_path,
                    'status': result['status'],
                    'duration_seconds': result.get('duration', 0),
                    'model_name': result.get('model_name', 'unknown'),
                    'rouge1': metrics.get('eval_rouge1', 0),
                    'rouge2': metrics.get('eval_rouge2', 0),
                    'rougeL': metrics.get('eval_rougeL', 0),
                    'rouge1_f1': metrics.get('eval_rouge1_f1', 0),
                    'rouge2_f1': metrics.get('eval_rouge2_f1', 0),
                    'rougeL_f1': metrics.get('eval_rougeL_f1', 0),
                    'rouge_combined': metrics.get('eval_rouge_combined', 0),
                    'rouge_combined_f1': metrics.get('eval_rouge_combined_f1', 0),
                    'eval_loss': metrics.get('eval_loss', 0),
                }
            else:
                # 실패한 실험도 기록
                row_data = {
                    'experiment_name': Path(config_path).stem,
                    'config_path': config_path,
                    'status': result.get('status', 'unknown'),
                    'duration_seconds': result.get('duration', 0),
                    'error': result.get('error', 'N/A')[:200]  # 에러 메시지 일부만
                }
            
            all_results.append(row_data)
        
        # DataFrame 생성 및 저장
        df = pd.DataFrame(all_results)
        
        # 성공한 실험 기준으로 ROUGE 점수로 정렬
        if 'rouge_combined_f1' in df.columns:
            df = df.sort_values('rouge_combined_f1', ascending=False)
        
        csv_path = self.output_dir / output_filename
        df.to_csv(csv_path, index=False)
        
        logger.info(f"전체 실험 결과 저장: {csv_path}")
        return csv_path
    
    def append_to_master_csv(self, result_data: Dict[str, Any], 
                           master_filename: str = "all_experiments.csv"):
        """
        마스터 CSV 파일에 결과 추가
        
        Args:
            result_data: 추가할 결과 데이터
            master_filename: 마스터 CSV 파일명
        """
        master_path = self.output_dir / master_filename
        
        # 기존 파일이 있으면 읽기
        if master_path.exists():
            df = pd.read_csv(master_path)
            new_df = pd.DataFrame([result_data])
            df = pd.concat([df, new_df], ignore_index=True)
        else:
            df = pd.DataFrame([result_data])
        
        # 저장
        df.to_csv(master_path, index=False)
        logger.info(f"마스터 CSV 업데이트: {master_path}")
    
    def create_comparison_table(self, csv_files: List[Path], 
                              output_filename: str = "model_comparison.csv") -> Path:
        """
        여러 CSV 파일의 결과를 비교 테이블로 생성
        
        Args:
            csv_files: 비교할 CSV 파일 경로 리스트
            output_filename: 출력 파일명
            
        Returns:
            비교 테이블 CSV 경로
        """
        all_dfs = []
        
        for csv_file in csv_files:
            if csv_file.exists():
                df = pd.read_csv(csv_file)
                all_dfs.append(df)
        
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            
            # 중복 제거 (실험명 기준)
            if 'experiment_name' in combined_df.columns:
                combined_df = combined_df.drop_duplicates(subset=['experiment_name'], keep='last')
            
            # ROUGE 점수로 정렬
            if 'rouge_combined_f1' in combined_df.columns:
                combined_df = combined_df.sort_values('rouge_combined_f1', ascending=False)
            
            comparison_path = self.output_dir / output_filename
            combined_df.to_csv(comparison_path, index=False)
            
            logger.info(f"비교 테이블 생성: {comparison_path}")
            return comparison_path
        
        return None
