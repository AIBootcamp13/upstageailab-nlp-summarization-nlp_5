#!/usr/bin/env python3
"""
체크포인트 탐색 유틸리티

학습된 모델의 체크포인트를 정확한 경로에서 찾아주는 유틸리티입니다.
실제 체크포인트 저장 구조: outputs/{experiment_name}_{timestamp}/checkpoints/checkpoint-*
"""

import logging
from pathlib import Path
from typing import Optional, List, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class CheckpointFinder:
    """체크포인트 탐색기"""
    
    def __init__(self, base_output_dir: str = "outputs"):
        """
        Args:
            base_output_dir: 기본 출력 디렉토리 (상대 경로)
        """
        self.base_output_dir = Path(base_output_dir)
        
    def find_latest_checkpoint(self, experiment_id: str = None) -> Optional[Path]:
        """
        가장 최근/최적의 체크포인트를 찾습니다.
        
        Args:
            experiment_id: 특정 실험 ID (없으면 가장 최근 실험 선택)
            
        Returns:
            체크포인트 경로 또는 None
        """
        logger.info(f"🔍 체크포인트 탐색 시작: experiment_id={experiment_id}")
        
        # 1. 실험 디렉토리 찾기
        experiment_dirs = self._find_experiment_directories()
        if not experiment_dirs:
            logger.warning("실험 디렉토리를 찾을 수 없습니다.")
            return None
            
        logger.info(f"📂 발견된 실험 디렉토리: {len(experiment_dirs)}개")
        for exp_dir in experiment_dirs[:3]:  # 처음 3개만 로그
            logger.info(f"  - {exp_dir}")
        
        # 2. 대상 실험 선택
        if experiment_id:
            target_dir = self._find_experiment_by_id(experiment_dirs, experiment_id)
        else:
            # 가장 최근 실험 선택
            target_dir = max(experiment_dirs, key=lambda p: p.stat().st_mtime)
            
        if not target_dir:
            logger.warning(f"실험 디렉토리를 찾을 수 없습니다: {experiment_id}")
            return None
            
        logger.info(f"🎯 대상 실험 디렉토리: {target_dir}")
        
        # 3. 체크포인트 디렉토리 탐색
        checkpoint_dir = target_dir / "checkpoints"
        if not checkpoint_dir.exists():
            logger.warning(f"체크포인트 디렉토리가 없습니다: {checkpoint_dir}")
            return None
            
        logger.info(f"📁 체크포인트 디렉토리: {checkpoint_dir}")
        
        # 4. checkpoint-* 디렉토리 찾기
        checkpoint_dirs = list(checkpoint_dir.glob("checkpoint-*"))
        if not checkpoint_dirs:
            logger.warning(f"체크포인트를 찾을 수 없습니다: {checkpoint_dir}")
            return None
            
        # 체크포인트 이름 로그
        checkpoint_names = [cp.name for cp in checkpoint_dirs]
        logger.info(f"🔢 발견된 체크포인트: {', '.join(sorted(checkpoint_names))}")
        
        # 5. 최적 체크포인트 선택
        best_checkpoint = self._find_best_checkpoint(checkpoint_dirs)
        
        # 6. 유효성 검증
        if self.validate_checkpoint(best_checkpoint):
            logger.info(f"✅ 발견된 체크포인트: {best_checkpoint}")
            return best_checkpoint
        else:
            logger.warning(f"❌ 체크포인트 유효성 검증 실패: {best_checkpoint}")
            return None
    
    def _find_experiment_directories(self) -> List[Path]:
        """outputs/* 패턴으로 실험 디렉토리 찾기"""
        experiment_dirs = []
        
        # outputs/* 패턴으로 모든 실험 디렉토리 찾기
        if self.base_output_dir.exists():
            for item in self.base_output_dir.iterdir():
                if item.is_dir():
                    experiment_dirs.append(item)
        
        # 시간순 정렬 (최신이 먼저)
        experiment_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        return experiment_dirs
    
    def _find_experiment_by_id(self, experiment_dirs: List[Path], experiment_id: str) -> Optional[Path]:
        """실험 ID로 특정 실험 디렉토리 찾기"""
        # 실험 ID가 디렉토리 이름에 포함되어 있는지 확인
        for exp_dir in experiment_dirs:
            # 실험 ID가 디렉토리 이름에 포함되어 있거나
            # 디렉토리 이름이 실험 ID로 끝나는 경우
            if experiment_id in exp_dir.name or exp_dir.name.endswith(experiment_id):
                return exp_dir
                
        # 정확히 매칭되는 것이 없으면 가장 최근 것 반환
        logger.warning(f"실험 ID '{experiment_id}'와 매칭되는 디렉토리를 찾을 수 없어 최신 디렉토리를 사용합니다.")
        return experiment_dirs[0] if experiment_dirs else None
    
    def _find_best_checkpoint(self, checkpoint_dirs: List[Path]) -> Path:
        """가장 좋은 체크포인트 선택 (가장 큰 숫자)"""
        # checkpoint-숫자 형식에서 숫자 추출
        numbered_checkpoints = []
        
        for cp_dir in checkpoint_dirs:
            try:
                # checkpoint-2800 -> 2800
                number = int(cp_dir.name.split('-')[-1])
                numbered_checkpoints.append((number, cp_dir))
            except (ValueError, IndexError):
                logger.warning(f"체크포인트 번호 추출 실패: {cp_dir.name}")
                continue
        
        if not numbered_checkpoints:
            # 숫자 추출이 모두 실패한 경우, 수정 시간 기준
            return max(checkpoint_dirs, key=lambda p: p.stat().st_mtime)
        
        # 가장 큰 번호의 체크포인트 선택
        _, best_checkpoint = max(numbered_checkpoints, key=lambda x: x[0])
        return best_checkpoint
    
    def validate_checkpoint(self, checkpoint_path: Path) -> bool:
        """체크포인트 유효성 검증"""
        if not checkpoint_path or not checkpoint_path.exists():
            return False
        
        # 필수 파일 목록 (safetensors 형식 지원)
        required_files = [
            ("config.json", True),  # (파일명, 필수여부)
            ("tokenizer_config.json", True),
            ("model.safetensors", False),  # safetensors 형식
            ("pytorch_model.bin", False),   # 기존 형식
        ]
        
        # config.json은 필수
        config_exists = (checkpoint_path / "config.json").exists()
        if not config_exists:
            logger.warning(f"config.json이 없습니다: {checkpoint_path}")
            return False
        
        # model 파일은 둘 중 하나만 있으면 됨
        model_exists = (
            (checkpoint_path / "model.safetensors").exists() or
            (checkpoint_path / "pytorch_model.bin").exists()
        )
        
        if not model_exists:
            logger.warning(f"모델 파일이 없습니다: {checkpoint_path}")
            return False
        
        # tokenizer 파일 확인
        tokenizer_exists = (
            (checkpoint_path / "tokenizer_config.json").exists() or
            (checkpoint_path / "tokenizer.json").exists() or
            (checkpoint_path / "spiece.model").exists()  # sentencepiece 모델
        )
        
        if not tokenizer_exists:
            logger.warning(f"토크나이저 파일이 없습니다: {checkpoint_path}")
            return False
        
        logger.info("✅ 체크포인트 유효성 검증 통과")
        return True
    
    def find_all_checkpoints(self) -> List[Tuple[Path, Path]]:
        """모든 실험의 체크포인트 찾기
        
        Returns:
            [(실험_디렉토리, 체크포인트_디렉토리), ...] 리스트
        """
        all_checkpoints = []
        
        experiment_dirs = self._find_experiment_directories()
        for exp_dir in experiment_dirs:
            checkpoint_dir = exp_dir / "checkpoints"
            if checkpoint_dir.exists():
                checkpoints = list(checkpoint_dir.glob("checkpoint-*"))
                if checkpoints:
                    best_checkpoint = self._find_best_checkpoint(checkpoints)
                    if self.validate_checkpoint(best_checkpoint):
                        all_checkpoints.append((exp_dir, best_checkpoint))
        
        return all_checkpoints


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
    
    # 체크포인트 탐색 테스트
    finder = CheckpointFinder()
    
    print("\n=== 체크포인트 탐색 테스트 ===")
    checkpoint = finder.find_latest_checkpoint()
    
    if checkpoint:
        print(f"\n✅ 최신 체크포인트: {checkpoint}")
        print(f"   경로 존재: {checkpoint.exists()}")
        
        # 파일 목록 확인
        print("\n📁 체크포인트 파일 목록:")
        for file in sorted(checkpoint.iterdir()):
            if file.is_file():
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"   - {file.name:30} ({size_mb:>8.2f} MB)")
    else:
        print("\n❌ 체크포인트를 찾을 수 없습니다.")
    
    # 모든 체크포인트 찾기
    print("\n=== 모든 체크포인트 ===")
    all_checkpoints = finder.find_all_checkpoints()
    for exp_dir, cp_dir in all_checkpoints[:5]:  # 최대 5개만 표시
        print(f"📁 {exp_dir.name} → {cp_dir.name}")
