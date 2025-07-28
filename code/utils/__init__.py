"""
NLP 대화 요약 프로젝트 - 유틸리티 패키지
설정 관리, 데이터 처리, 메트릭 계산 등 공통 기능 제공
"""

from .data_utils import DataProcessor, TextPreprocessor
from .experiment_utils import ExperimentTracker, ModelRegistry

# Rouge 메트릭은 선택적으로 import (설치되지 않은 경우 대비)
try:
    from .metrics import MultiReferenceROUGE, RougeCalculator
    ROUGE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Rouge 메트릭 모듈을 불러올 수 없습니다: {e}")
    print("👉 'pip install rouge==1.0.1'로 설치해주세요.")
    MultiReferenceROUGE = None
    RougeCalculator = None
    ROUGE_AVAILABLE = False

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
    'ModelRegistry',
    'ROUGE_AVAILABLE'
]
