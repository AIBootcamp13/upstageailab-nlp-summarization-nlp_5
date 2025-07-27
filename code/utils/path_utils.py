"""
PathManager: 크로스 플랫폼 경로 관리 시스템

모든 파일 경로를 상대 경로로 처리하여 프로젝트 이식성을 보장합니다.
Windows, Mac, Linux에서 동일하게 동작합니다.
"""

import os
import sys
from pathlib import Path
from typing import Union, Optional


class PathManager:
    """프로젝트 경로를 관리하는 싱글톤 클래스"""
    
    _instance = None
    _project_root = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._project_root is None:
            self._project_root = self._find_project_root()
    
    @classmethod
    def _find_project_root(cls) -> Path:
        """
        프로젝트 루트 디렉토리를 찾습니다.
        
        프로젝트 루트는 다음 조건을 만족하는 디렉토리입니다:
        - code/, data/, config/ 디렉토리가 존재
        - 또는 .git 디렉토리가 존재
        - 또는 pyproject.toml 파일이 존재
        """
        current = Path.cwd()
        
        # 현재 디렉토리부터 상위로 올라가며 검색
        for parent in [current] + list(current.parents):
            # 프로젝트 루트 식별 조건들
            if all((parent / d).exists() for d in ['code', 'config']):
                return parent
            if (parent / '.git').exists():
                return parent
            if (parent / 'pyproject.toml').exists():
                return parent
        
        # 찾지 못한 경우 현재 디렉토리 반환
        return current
    
    @property
    def project_root(self) -> Path:
        """프로젝트 루트 디렉토리 경로를 반환합니다."""
        return self._project_root
    
    def resolve_path(self, relative_path: Union[str, Path]) -> Path:
        """
        상대 경로를 절대 경로로 변환합니다.
        
        Args:
            relative_path: 프로젝트 루트 기준 상대 경로
            
        Returns:
            절대 경로 Path 객체
        """
        if isinstance(relative_path, str):
            relative_path = Path(relative_path)
        
        # 이미 절대 경로인 경우 그대로 반환
        if relative_path.is_absolute():
            return relative_path
        
        # 프로젝트 루트 기준으로 절대 경로 생성
        return self._project_root / relative_path
    
    def ensure_dir(self, directory: Union[str, Path]) -> Path:
        """
        디렉토리가 존재하지 않으면 생성합니다.
        
        Args:
            directory: 생성할 디렉토리 경로
            
        Returns:
            생성된 디렉토리의 절대 경로
        """
        abs_path = self.resolve_path(directory)
        abs_path.mkdir(parents=True, exist_ok=True)
        return abs_path
    
    def get_data_path(self, filename: Optional[str] = None) -> Path:
        """데이터 디렉토리 경로를 반환합니다."""
        data_dir = self.resolve_path("data")
        if filename:
            return data_dir / filename
        return data_dir
    
    def get_config_path(self, filename: Optional[str] = None) -> Path:
        """설정 파일 디렉토리 경로를 반환합니다."""
        config_dir = self.resolve_path("config")
        if filename:
            return config_dir / filename
        return config_dir
    
    def get_output_path(self, experiment_name: str, filename: Optional[str] = None) -> Path:
        """실험 출력 디렉토리 경로를 반환합니다."""
        output_dir = self.resolve_path(f"outputs/{experiment_name}")
        self.ensure_dir(output_dir)
        if filename:
            return output_dir / filename
        return output_dir
    
    def get_model_path(self, model_name: str, filename: Optional[str] = None) -> Path:
        """모델 저장 디렉토리 경로를 반환합니다."""
        model_dir = self.resolve_path(f"models/{model_name}")
        self.ensure_dir(model_dir)
        if filename:
            return model_dir / filename
        return model_dir
    
    def get_log_path(self, experiment_name: str, filename: Optional[str] = None) -> Path:
        """로그 디렉토리 경로를 반환합니다."""
        log_dir = self.resolve_path(f"logs/{experiment_name}")
        self.ensure_dir(log_dir)
        if filename:
            return log_dir / filename
        return log_dir
    
    def relative_to_root(self, path: Union[str, Path]) -> Path:
        """
        절대 경로를 프로젝트 루트 기준 상대 경로로 변환합니다.
        
        Args:
            path: 변환할 경로
            
        Returns:
            프로젝트 루트 기준 상대 경로
        """
        if isinstance(path, str):
            path = Path(path)
        
        if not path.is_absolute():
            return path
        
        try:
            return path.relative_to(self._project_root)
        except ValueError:
            # 프로젝트 루트 외부의 경로인 경우 그대로 반환
            return path
    
    def __str__(self) -> str:
        return f"PathManager(root={self._project_root})"
    
    def __repr__(self) -> str:
        return str(self)


# 전역 인스턴스 생성
path_manager = PathManager()


# 편의 함수들
def get_project_root() -> Path:
    """프로젝트 루트 디렉토리를 반환합니다."""
    return path_manager.project_root


def resolve_path(relative_path: Union[str, Path]) -> Path:
    """상대 경로를 절대 경로로 변환합니다."""
    return path_manager.resolve_path(relative_path)


def ensure_dir(directory: Union[str, Path]) -> Path:
    """디렉토리가 존재하지 않으면 생성합니다."""
    return path_manager.ensure_dir(directory)
