"""
NLP 대화 요약 프로젝트 - 유틸리티 패키지
설정 관리, 데이터 처리, 메트릭 계산 등 공통 기능 제공
"""

from .data_utils import DataProcessor, TextPreprocessor
from .metrics import MultiReferenceROUGE, RougeCalculator
from .experiment_utils import ExperimentTracker, ModelRegistry
import yaml
from pathlib import Path
from typing import Dict, Any, Union

__version__ = "1.0.0"
__author__ = "NLP Team 5"

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    간단한 YAML 설정 파일 로딩 함수
    
    Args:
        config_path: 설정 파일 경로
        
    Returns:
        로딩된 설정 딕셔너리
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

__all__ = [
    'load_config',
    'DataProcessor',
    'TextPreprocessor',
    'MultiReferenceROUGE',
    'RougeCalculator',
    'ExperimentTracker',
    'ModelRegistry'
]
