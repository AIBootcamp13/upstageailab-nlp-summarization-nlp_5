#!/usr/bin/env python3
"""
빠른 검증 스크립트 (Quick Test Runner)

1에포크만 실행하여 전체 파이프라인이 에러 없이 동작하는지 빠르게 검증합니다.
"""

import sys
import os
import yaml
import argparse
import logging
from pathlib import Path

# 프로젝트 경로 추가
sys.path.insert(0, str(Path(__file__).parent))

from code.trainer import NMTTrainer, TrainingConfig
from code.utils import load_config

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_quick_test_config(base_config: dict) -> dict:
    """
    빠른 테스트를 위한 설정 생성
    
    Args:
        base_config: 기본 설정 딕셔너리
        
    Returns:
        빠른 테스트용으로 수정된 설정
    """
    # 기본 설정 복사
    quick_config = base_config.copy()
    
    # 빠른 테스트 설정 적용
    training_overrides = {
        'num_train_epochs': 1,  # 1에포크만
        'per_device_train_batch_size': 2,  # 작은 배치 크기
        'per_device_eval_batch_size': 2,
        'logging_steps': 10,  # 자주 로깅
        'eval_steps': 50,  # 자주 평가
        'save_steps': 50,  # 자주 저장
        'warmup_steps': 10,  # 적은 워밍업
        'max_steps': 100,  # 최대 스텝 제한
        'load_best_model_at_end': False,  # 시간 단축
        'evaluation_strategy': 'steps',  # steps 기반 평가
        'save_strategy': 'steps',
        'report_to': [],  # WandB 비활성화 (선택적)
    }
    
    # 토크나이저 설정 조정
    tokenizer_overrides = {
        'encoder_max_len': 256,  # 짧은 입력
        'decoder_max_len': 64,   # 짧은 출력
    }
    
    # 추론 설정 조정
    inference_overrides = {
        'batch_size': 2,
        'generate_max_length': 64,
    }
    
    # 설정 적용
    if 'training' in quick_config:
        quick_config['training'].update(training_overrides)
    else:
        quick_config['training'] = training_overrides
        
    if 'tokenizer' in quick_config:
        quick_config['tokenizer'].update(tokenizer_overrides)
    else:
        quick_config['tokenizer'] = tokenizer_overrides
        
    if 'inference' in quick_config:
        quick_config['inference'].update(inference_overrides)
    else:
        quick_config['inference'] = inference_overrides
    
    # 출력 디렉토리를 quick_test로 변경
    if 'general' in quick_config:
        original_output = quick_config['general'].get('output_dir', './outputs')
        quick_config['general']['output_dir'] = f"{original_output}_quick_test"
    
    return quick_config


def limit_dataset_samples(trainer: NMTTrainer, max_samples: int = 100):
    """
    데이터셋 샘플 수를 제한하여 빠른 테스트 수행
    
    Args:
        trainer: NMTTrainer 인스턴스
        max_samples: 최대 샘플 수
    """
    if hasattr(trainer, 'train_dataset') and trainer.train_dataset:
        original_size = len(trainer.train_dataset)
        if original_size > max_samples:
            # 데이터셋을 max_samples로 제한
            trainer.train_dataset = trainer.train_dataset.select(range(max_samples))
            logger.info(f"🗂️ 훈련 데이터셋 제한: {original_size} -> {max_samples} 샘플")
    
    if hasattr(trainer, 'valid_dataset') and trainer.valid_dataset:
        original_size = len(trainer.valid_dataset)
        eval_samples = min(max_samples // 4, 50)  # 평가용은 더 적게
        if original_size > eval_samples:
            trainer.valid_dataset = trainer.valid_dataset.select(range(eval_samples))
            logger.info(f"🗂️ 검증 데이터셋 제한: {original_size} -> {eval_samples} 샘플")


def run_quick_test(config_path: str, 
                   model_section: str = None,
                   max_samples: int = 100,
                   disable_wandb: bool = True) -> dict:
    """
    빠른 테스트 실행
    
    Args:
        config_path: 설정 파일 경로
        model_section: 사용할 모델 섹션 (예: 'eenzeenee', 'xlsum_mt5')
        max_samples: 최대 훈련 샘플 수
        disable_wandb: WandB 비활성화 여부
        
    Returns:
        테스트 결과 딕셔너리
    """
    try:
        logger.info(f"🚀 빠른 테스트 시작: {config_path}")
        
        # 설정 로드
        base_config = load_config(config_path)
        
        # 특정 모델 섹션 사용
        if model_section and model_section in base_config:
            logger.info(f"📋 모델 섹션 사용: {model_section}")
            config = base_config[model_section]
        else:
            config = base_config
        
        # 빠른 테스트 설정 적용
        quick_config = create_quick_test_config(config)
        
        # WandB 비활성화
        if disable_wandb:
            quick_config.get('training', {})['report_to'] = []
            logger.info("📴 WandB 비활성화됨")
        
        # 트레이너 초기화
        trainer = NMTTrainer(quick_config)
        
        # 모델 및 토크나이저 로드
        logger.info("🤖 모델 로딩 중...")
        trainer.load_model_and_tokenizer()
        logger.info("✅ 모델 로딩 완료")
        
        # 데이터 로드
        logger.info("📊 데이터 로딩 중...")
        trainer.load_data()
        logger.info("✅ 데이터 로딩 완료")
        
        # 데이터셋 샘플 수 제한
        limit_dataset_samples(trainer, max_samples)
        
        # 트레이너 설정
        logger.info("⚙️ 트레이너 설정 중...")
        trainer.setup_trainer()
        logger.info("✅ 트레이너 설정 완료")
        
        # 빠른 훈련 실행
        logger.info("🏃 빠른 훈련 시작 (1 epoch)...")
        train_result = trainer.train()
        logger.info("✅ 훈련 완료")
        
        # 간단한 평가
        logger.info("📈 평가 실행 중...")
        eval_result = trainer.evaluate()
        logger.info("✅ 평가 완료")
        
        # 결과 정리
        result = {
            'status': 'success',
            'model_name': quick_config.get('general', {}).get('model_name', 'unknown'),
            'train_samples': len(trainer.train_dataset) if trainer.train_dataset else 0,
            'eval_samples': len(trainer.valid_dataset) if trainer.valid_dataset else 0,
            'epochs_completed': 1,
            'train_metrics': train_result,
            'eval_metrics': eval_result,
            'config_used': quick_config
        }
        
        # 성능 로깅
        if 'eval_rouge_l' in eval_result:
            rouge_l = eval_result['eval_rouge_l']
            logger.info(f"🎯 ROUGE-L 점수: {rouge_l:.4f}")
        
        logger.info("🎉 빠른 테스트 성공적으로 완료!")
        return result
        
    except Exception as e:
        logger.error(f"❌ 빠른 테스트 실패: {e}")
        return {
            'status': 'failed',
            'error': str(e),
            'model_name': model_section or 'unknown'
        }
    
    finally:
        # 리소스 정리
        if 'trainer' in locals():
            trainer.cleanup()


def main():
    """CLI 인터페이스"""
    parser = argparse.ArgumentParser(description="빠른 검증 테스트 러너")
    parser.add_argument('--config', default='config.yaml',
                       help='설정 파일 경로')
    parser.add_argument('--model-section', 
                       choices=['eenzeenee', 'xlsum_mt5', 'baseline'],
                       help='사용할 모델 섹션')
    parser.add_argument('--max-samples', type=int, default=100,
                       help='최대 훈련 샘플 수')
    parser.add_argument('--disable-wandb', action='store_true', default=True,
                       help='WandB 비활성화')
    parser.add_argument('--verbose', action='store_true',
                       help='상세 로깅 활성화')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 설정 파일 확인
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"설정 파일을 찾을 수 없습니다: {config_path}")
        return 1
    
    # 빠른 테스트 실행
    result = run_quick_test(
        config_path=str(config_path),
        model_section=args.model_section,
        max_samples=args.max_samples,
        disable_wandb=args.disable_wandb
    )
    
    # 결과 출력
    if result['status'] == 'success':
        print(f"\n✅ 빠른 테스트 성공!")
        print(f"📊 모델: {result['model_name']}")
        print(f"📊 훈련 샘플: {result['train_samples']}")
        print(f"📊 평가 샘플: {result['eval_samples']}")
        
        eval_metrics = result.get('eval_metrics', {})
        if 'eval_rouge_l' in eval_metrics:
            print(f"🎯 ROUGE-L: {eval_metrics['eval_rouge_l']:.4f}")
        
        return 0
    else:
        print(f"\n❌ 빠른 테스트 실패: {result['error']}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
