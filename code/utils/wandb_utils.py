"""
WandB 통합 유틸리티

실험별로 고유한 WandB run 설정 및 모델 레지스트리 연동
"""

import os
from datetime import datetime
from typing import Dict, Any, Optional
import wandb
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def setup_wandb_for_experiment(config: Dict[str, Any], 
                             experiment_name: str,
                             sweep_mode: bool = False) -> Dict[str, Any]:
    """
    실험별 WandB 설정 초기화
    
    Args:
        config: 실험 설정
        experiment_name: 실험명
        sweep_mode: Sweep 모드 여부
        
    Returns:
        WandB 설정 정보
    """
    # 한국 시간 기반 타임스탬프
    from utils.experiment_utils import get_korean_time_format
    korean_time = get_korean_time_format('MMDDHHMM')
    
    # WandB 설정 가져오기
    wandb_config = config.get('wandb', {})
    
    # 실험별 고유한 run name 생성
    # 조장님 패턴: b_automodel_{current_time}
    model_name = config.get('model', {}).get('architecture', 'unknown')
    model_short = model_name[:1] if model_name != 'unknown' else 'x'  # 첫 글자만
    run_name = f"{model_short}_{experiment_name}_{korean_time}"
    
    # Job type 설정
    job_type = "sweep" if sweep_mode else "experiment"
    
    # Tags 설정
    tags = wandb_config.get('tags', []).copy()
    tags.extend([
        model_name,
        experiment_name,
        f"date_{datetime.now().strftime('%Y%m%d')}",
        f"time_{korean_time}"
    ])
    
    # Notes 생성
    notes = f"""
실험명: {experiment_name}
모델: {config.get('model', {}).get('checkpoint', 'N/A')}
시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (KST)
학습 에포크: {config.get('training', {}).get('num_train_epochs', 'N/A')}
배치 크기: {config.get('training', {}).get('per_device_train_batch_size', 'N/A')}
학습률: {config.get('training', {}).get('learning_rate', 'N/A')}
"""
    
    # WandB 초기화 설정
    wandb_init_config = {
        'project': wandb_config.get('project', 'nlp-5'),
        'entity': wandb_config.get('entity', 'lyjune37-juneictlab'),
        'name': run_name,
        'notes': notes,
        'tags': tags,
        'group': wandb_config.get('group', experiment_name),
        'job_type': job_type,
        'config': config,
        'save_code': wandb_config.get('save_code', True),
    }
    
    # 환경 변수 설정
    if wandb_config.get('log_model', 'end') == 'end':
        os.environ["WANDB_LOG_MODEL"] = "end"
    
    return wandb_init_config


def log_model_to_wandb(model_path: str, 
                      model_name: str,
                      metrics: Dict[str, float],
                      config: Dict[str, Any],
                      aliases: Optional[list] = None):
    """
    학습된 모델을 WandB Model Registry에 등록
    
    Args:
        model_path: 모델 저장 경로
        model_name: 모델명
        metrics: 성능 메트릭
        config: 모델 설정
        aliases: 모델 별칭 (예: ["best", "latest"])
    """
    if wandb.run is None:
        logger.warning("WandB run이 활성화되지 않아 모델을 등록할 수 없습니다.")
        return
    
    # 모델 크기 확인
    model_size_mb = get_directory_size_mb(model_path)
    size_threshold = config.get('wandb', {}).get('log_model_size_threshold', 2000)  # 기본 2GB
    
    # 모델 크기가 임계값을 초과하면 로컬만 저장
    if model_size_mb > size_threshold:
        logger.warning(f"모델 크기({model_size_mb:.1f}MB)가 임계값({size_threshold}MB)을 초과하여 WandB에 업로드하지 않습니다.")
        logger.info(f"모델은 로컬에만 저장되었습니다: {model_path}")
        # WandB에 메타데이터만 기록
        wandb.run.summary.update({
            "model_saved_locally": True,
            "model_size_mb": model_size_mb,
            "model_path": model_path,
            **metrics
        })
        return
    
    try:
        # 모델 아티팩트 생성
        from utils.experiment_utils import get_korean_time_format
        model_artifact = wandb.Artifact(
            name=model_name,
            type="model",
            description=f"Dialogue summarization model: {model_name}",
            metadata={
                "rouge1_f1": metrics.get('rouge1_f1', 0),
                "rouge2_f1": metrics.get('rouge2_f1', 0),
                "rougeL_f1": metrics.get('rougeL_f1', 0),
                "rouge_combined_f1": metrics.get('rouge_combined_f1', 0),
                "architecture": config.get('model', {}).get('architecture', 'unknown'),
                "base_model": config.get('model', {}).get('checkpoint', 'unknown'),
                "training_epochs": config.get('training', {}).get('num_train_epochs', 0),
                "korean_time": get_korean_time_format('MMDDHHMM'),
                "model_size_mb": model_size_mb
            }
        )
        
        # 모델 파일 추가
        model_artifact.add_dir(model_path)
        
        # 별칭 설정
        if aliases is None:
            aliases = ["latest"]
            # 최고 성능 모델인 경우 best 태그 추가
            if metrics.get('rouge_combined_f1', 0) > 0.3:  # 임계값 조정 가능
                aliases.append("best")
        
        # WandB에 로그
        wandb.run.log_artifact(model_artifact, aliases=aliases)
        
        logger.info(f"모델이 WandB에 등록되었습니다: {model_name} ({model_size_mb:.1f}MB)")
        
    except Exception as e:
        logger.error(f"WandB 모델 등록 실패: {e}")


def get_directory_size_mb(directory_path: str) -> float:
    """
    디렉토리의 전체 크기를 MB 단위로 계산
    
    Args:
        directory_path: 크기를 계산할 디렉토리 경로
        
    Returns:
        디렉토리 크기 (MB)
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    
    return total_size / (1024 * 1024)  # bytes to MB


def finish_wandb_run(summary_metrics: Dict[str, Any] = None):
    """
    WandB run 종료 및 요약 정보 저장
    
    Args:
        summary_metrics: 최종 요약 메트릭
    """
    if wandb.run is not None:
        # 요약 메트릭 업데이트
        if summary_metrics:
            wandb.run.summary.update(summary_metrics)
        
        # run 종료
        wandb.finish()
        logger.info("WandB run이 종료되었습니다.")


def get_wandb_run_url() -> Optional[str]:
    """현재 WandB run의 URL 반환"""
    if wandb.run is not None:
        return wandb.run.get_url()
    return None
