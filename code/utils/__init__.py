"""
NLP 대화 요약 프로젝트 - 유틸리티 패키지
설정 관리, 데이터 처리, 메트릭 계산 등 공통 기능 제공
"""

from .config_manager import ConfigManager, ConfigValidationError
from .data_utils import DataProcessor, TextPreprocessor
from .metrics import MultiReferenceROUGE, RougeCalculator
from .experiment_utils import ExperimentTracker, ModelRegistry

__version__ = "1.0.0"
__author__ = "NLP Team 5"

__all__ = [
    'ConfigManager',
    'ConfigValidationError', 
    'DataProcessor',
    'TextPreprocessor',
    'MultiReferenceROUGE',
    'RougeCalculator',
    'ExperimentTracker',
    'ModelRegistry'
]
