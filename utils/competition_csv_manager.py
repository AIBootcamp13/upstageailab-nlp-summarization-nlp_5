#!/usr/bin/env python3
"""
대회 제출용 CSV 관리 유틸리티

대회 채점용 CSV 파일을 baseline.py와 동일한 형식으로 생성하면서,
다중 실험 지원 및 실험 추적 기능을 제공합니다.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


class CompetitionCSVManager:
    """대회 제출용 CSV 관리자"""
    
    def __init__(self, prediction_base: str = "./prediction"):
        """
        Args:
            prediction_base: 예측 결과 저장 기본 디렉토리
        """
        self.prediction_base = Path(prediction_base)
        self._ensure_directories()
        
    def _ensure_directories(self):
        """필수 디렉토리 생성"""
        self.prediction_base.mkdir(exist_ok=True)
        (self.prediction_base / "history").mkdir(exist_ok=True)
        
    def save_experiment_submission(self, 
                                 experiment_name: str, 
                                 result_df: pd.DataFrame,
                                 config: Dict = None,
                                 metrics: Dict = None,
                                 timestamp: str = None) -> Dict[str, str]:
        """
        실험 결과를 대회 제출 형식으로 저장
        
        Args:
            experiment_name: 실험명
            result_df: 결과 DataFrame (fname, summary 컬럼 필수)
            config: 실험 설정 (선택)
            metrics: 성능 지표 (선택)
            timestamp: 타임스탬프 (없으면 자동 생성)
            
        Returns:
            생성된 파일 경로들
        """
        # 입력 검증
        if 'fname' not in result_df.columns or 'summary' not in result_df.columns:
            raise ValueError("result_df must have 'fname' and 'summary' columns")
        
        # 타임스탬프 생성
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 제출용 DataFrame 준비 (fname, summary만)
        submission_df = result_df[['fname', 'summary']].copy()
        
        # 실험별 폴더 생성
        experiment_folder = f"{experiment_name}_{timestamp}"
        experiment_path = self.prediction_base / experiment_folder
        experiment_path.mkdir(exist_ok=True)
        
        # 1. 실험별 output.csv 저장 (대회 표준 형식)
        output_path = experiment_path / "output.csv"
        submission_df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"📤 실험별 제출 파일 저장: {output_path}")
        
        # 2. 메타데이터 저장
        metadata_path = experiment_path / "experiment_metadata.json"
        self._save_experiment_metadata(
            metadata_path, experiment_name, config, metrics, timestamp
        )
        
        # 3. latest_output.csv 업데이트 (덮어쓰기)
        latest_path = self.prediction_base / "latest_output.csv"
        submission_df.to_csv(latest_path, index=False, encoding='utf-8')
        logger.info(f"📤 최신 제출 파일 업데이트: {latest_path}")
        
        # 4. 히스토리 백업
        history_path = self._save_to_history(submission_df, experiment_name, timestamp)
        
        # 5. 실험 인덱스 업데이트
        self._update_experiment_index(
            experiment_name, experiment_folder, timestamp, metrics
        )
        
        # 6. 생성 요약 출력
        result_paths = {
            'experiment_path': str(output_path),
            'latest_path': str(latest_path),
            'history_path': str(history_path),
            'metadata_path': str(metadata_path),
            'experiment_folder': experiment_folder
        }
        
        self._print_generation_summary(result_paths, len(submission_df))
        
        return result_paths
    
    def _save_experiment_metadata(self, 
                                metadata_path: Path,
                                experiment_name: str,
                                config: Dict,
                                metrics: Dict,
                                timestamp: str):
        """실험 메타데이터 저장"""
        metadata = {
            "experiment_name": experiment_name,
            "timestamp": timestamp,
            "created_at": datetime.now().isoformat(),
            "submission_info": {
                "format": "fname,summary",
                "encoding": "utf-8"
            }
        }
        
        # 설정 정보 추가
        if config:
            metadata["model_name"] = config.get('general', {}).get('model_name', 'unknown')
            metadata["config_summary"] = {
                "learning_rate": config.get('training', {}).get('learning_rate'),
                "batch_size": config.get('training', {}).get('per_device_train_batch_size'),
                "num_epochs": config.get('training', {}).get('num_train_epochs')
            }
        
        # 메트릭 정보 추가
        if metrics:
            metadata["metrics"] = {
                "eval_rouge1_f1": metrics.get('eval_rouge1', 0),
                "eval_rouge2_f1": metrics.get('eval_rouge2', 0),
                "eval_rougeL_f1": metrics.get('eval_rougeL', 0),
                "eval_rouge_combined_f1": (
                    metrics.get('eval_rouge1', 0) + 
                    metrics.get('eval_rouge2', 0) + 
                    metrics.get('eval_rougeL', 0)
                )
            }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📄 메타데이터 저장: {metadata_path}")
    
    def _save_to_history(self, submission_df: pd.DataFrame, 
                        experiment_name: str, timestamp: str) -> str:
        """히스토리 백업"""
        history_filename = f"output_{experiment_name}_{timestamp}.csv"
        history_path = self.prediction_base / "history" / history_filename
        submission_df.to_csv(history_path, index=False, encoding='utf-8')
        logger.info(f"💾 히스토리 백업: {history_path}")
        return str(history_path)
    
    def _update_experiment_index(self, 
                               experiment_name: str,
                               experiment_folder: str,
                               timestamp: str,
                               metrics: Dict = None):
        """실험 인덱스 업데이트"""
        index_path = self.prediction_base / "experiment_index.csv"
        
        # 기존 인덱스 로드 또는 새로 생성
        if index_path.exists():
            index_df = pd.read_csv(index_path)
        else:
            index_df = pd.DataFrame(columns=[
                'experiment_name', 'folder_name', 'timestamp', 
                'submission_file', 'latest_file', 'created_at',
                'rouge_combined', 'rouge1', 'rouge2', 'rougeL'
            ])
        
        # 새 실험 정보 추가
        new_row = {
            'experiment_name': experiment_name,
            'folder_name': experiment_folder,
            'timestamp': timestamp,
            'submission_file': f"./prediction/{experiment_folder}/output.csv",
            'latest_file': "./prediction/latest_output.csv",
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 메트릭 정보 추가
        if metrics:
            new_row['rouge1'] = metrics.get('eval_rouge1', 0)
            new_row['rouge2'] = metrics.get('eval_rouge2', 0)
            new_row['rougeL'] = metrics.get('eval_rougeL', 0)
            new_row['rouge_combined'] = (
                new_row['rouge1'] + new_row['rouge2'] + new_row['rougeL']
            )
        else:
            new_row['rouge1'] = 0
            new_row['rouge2'] = 0
            new_row['rougeL'] = 0
            new_row['rouge_combined'] = 0
        
        # DataFrame에 추가 (concat 사용)
        index_df = pd.concat([index_df, pd.DataFrame([new_row])], ignore_index=True)
        
        # 시간순 정렬 (최신이 위로)
        index_df = index_df.sort_values('created_at', ascending=False)
        
        # 저장
        index_df.to_csv(index_path, index=False, encoding='utf-8')
        logger.info(f"📋 실험 인덱스 업데이트: {index_path}")
    
    def _print_generation_summary(self, result_paths: Dict[str, str], num_samples: int):
        """생성 결과 요약 출력"""
        print("\n✅ 채점용 CSV 파일 생성 완료!")
        print(f"📊 처리된 샘플 수: {num_samples}")
        print("📁 생성된 파일들:")
        print(f"  📤 실험별 제출: {result_paths['experiment_path']}")
        print(f"  📤 최신 제출: {result_paths['latest_path']}")
        print(f"  📋 실험 인덱스: {self.prediction_base}/experiment_index.csv")
        print(f"  💾 히스토리 백업: {result_paths['history_path']}")
        print(f"  📄 메타데이터: {result_paths['metadata_path']}")
    
    def get_latest_experiment(self) -> Optional[Dict]:
        """가장 최근 실험 정보 조회"""
        index_path = self.prediction_base / "experiment_index.csv"
        if not index_path.exists():
            return None
        
        index_df = pd.read_csv(index_path)
        if len(index_df) == 0:
            return None
        
        # 첫 번째 행이 가장 최근 (이미 정렬됨)
        latest = index_df.iloc[0].to_dict()
        return latest
    
    def list_all_experiments(self) -> pd.DataFrame:
        """모든 실험 목록 조회"""
        index_path = self.prediction_base / "experiment_index.csv"
        if not index_path.exists():
            return pd.DataFrame()
        
        return pd.read_csv(index_path)
    
    def get_best_experiment_by_rouge(self) -> Optional[Dict]:
        """ROUGE 점수 기준 최고 성능 실험 조회"""
        index_path = self.prediction_base / "experiment_index.csv"
        if not index_path.exists():
            return None
        
        index_df = pd.read_csv(index_path)
        if len(index_df) == 0:
            return None
        
        # ROUGE combined 점수 기준 정렬
        best_idx = index_df['rouge_combined'].idxmax()
        best = index_df.loc[best_idx].to_dict()
        return best
    
    def cleanup_old_experiments(self, keep_latest: int = 10):
        """오래된 실험 결과 정리"""
        index_path = self.prediction_base / "experiment_index.csv"
        if not index_path.exists():
            return
        
        index_df = pd.read_csv(index_path)
        if len(index_df) <= keep_latest:
            return
        
        # 보관할 실험과 삭제할 실험 분리
        keep_df = index_df.head(keep_latest)
        remove_df = index_df.iloc[keep_latest:]
        
        # 삭제할 실험 폴더 제거
        for _, row in remove_df.iterrows():
            folder_path = self.prediction_base / row['folder_name']
            if folder_path.exists():
                shutil.rmtree(folder_path)
                logger.info(f"🗑️ 오래된 실험 삭제: {folder_path}")
        
        # 인덱스 업데이트
        keep_df.to_csv(index_path, index=False, encoding='utf-8')
        logger.info(f"✨ 최근 {keep_latest}개 실험만 유지")


# 테스트 코드
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # 프로젝트 루트 추가
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    
    # 테스트 데이터 생성
    test_df = pd.DataFrame({
        'fname': [f'TEST_{i:03d}' for i in range(1, 6)],
        'summary': [f'테스트 요약 {i}' for i in range(1, 6)]
    })
    
    # CSV 관리자 테스트
    manager = CompetitionCSVManager()
    
    print("\n=== CSV 관리자 테스트 ===")
    
    # 실험 결과 저장
    result_paths = manager.save_experiment_submission(
        experiment_name="test_experiment",
        result_df=test_df,
        config={
            'general': {'model_name': 'test-model'},
            'training': {
                'learning_rate': 1e-5,
                'per_device_train_batch_size': 16,
                'num_train_epochs': 3
            }
        },
        metrics={
            'eval_rouge1': 0.254,
            'eval_rouge2': 0.095,
            'eval_rougeL': 0.230
        }
    )
    
    print("\n생성된 파일 경로:")
    for key, path in result_paths.items():
        print(f"  {key}: {path}")
    
    # 최근 실험 조회
    print("\n=== 최근 실험 조회 ===")
    latest = manager.get_latest_experiment()
    if latest:
        print(f"최근 실험: {latest['experiment_name']}")
        print(f"제출 파일: {latest['submission_file']}")
        print(f"ROUGE 점수: {latest['rouge_combined']}")
    
    # 전체 실험 목록
    print("\n=== 전체 실험 목록 ===")
    all_experiments = manager.list_all_experiments()
    if not all_experiments.empty:
        print(all_experiments[['experiment_name', 'rouge_combined', 'created_at']].head())
