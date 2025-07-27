#!/usr/bin/env python3
"""
백트랜슬레이션 데이터 증강 실행 스크립트

학습 데이터에 백트랜슬레이션을 적용하여 증강된 데이터셋을 생성합니다.
"""

import os
import sys
import argparse
import yaml
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging
from datetime import datetime

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from code.data_augmentation.backtranslation import (
    BackTranslationAugmenter, 
    MultilingualBackTranslation,
    create_backtranslation_augmenter
)
from code.data_augmentation.backtranslation_evaluator import (
    BackTranslationEvaluator,
    analyze_augmentation_distribution
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """설정 파일 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def augment_dataset(
    df: pd.DataFrame,
    config: dict,
    text_column: str = 'dialogue',
    summary_column: str = 'summary'
) -> pd.DataFrame:
    """데이터셋 증강"""
    
    # 백트랜슬레이션 설정
    bt_config = config['data_augmentation']['backtranslation']
    
    # 증강기 생성
    augmenter = create_backtranslation_augmenter(bt_config)
    
    # 증강할 샘플 수 계산
    augmentation_ratio = bt_config.get('augmentation_ratio', 0.5)
    num_samples_to_augment = int(len(df) * augmentation_ratio)
    num_augmentations_per_sample = bt_config.get('num_augmentations_per_sample', 1)
    
    logger.info(f"증강할 샘플 수: {num_samples_to_augment}")
    logger.info(f"샘플당 증강 수: {num_augmentations_per_sample}")
    
    # 랜덤 샘플링
    samples_to_augment = df.sample(n=num_samples_to_augment, random_state=42)
    
    augmented_rows = []
    
    # 원본 데이터 추가
    for idx, row in df.iterrows():
        augmented_rows.append({
            text_column: row[text_column],
            summary_column: row[summary_column],
            'is_augmented': False,
            'augmentation_method': 'original',
            'original_idx': idx
        })
    
    # 백트랜슬레이션 증강
    logger.info("백트랜슬레이션 시작...")
    
    for idx, row in tqdm(samples_to_augment.iterrows(), total=len(samples_to_augment)):
        text = row[text_column]
        summary = row[summary_column]
        
        # 대화 텍스트 증강
        augmented_texts = augmenter.augment(text, num_augmentations_per_sample)
        
        for aug_text in augmented_texts:
            augmented_rows.append({
                text_column: aug_text,
                summary_column: summary,  # 요약은 그대로 유지
                'is_augmented': True,
                'augmentation_method': 'backtranslation',
                'original_idx': idx
            })
    
    # 데이터프레임 생성
    augmented_df = pd.DataFrame(augmented_rows)
    
    # 간단한 증강과 조합 (옵션)
    if bt_config.get('combine_with_simple', False):
        logger.info("간단한 증강 방법과 조합...")
        # 기존 simple_augmentation.py의 방법들을 추가로 적용
        # TODO: 구현 필요
    
    return augmented_df


def evaluate_augmentation_quality(
    original_df: pd.DataFrame,
    augmented_df: pd.DataFrame,
    config: dict,
    output_dir: Path
) -> dict:
    """증강 품질 평가"""
    
    logger.info("증강 품질 평가 시작...")
    
    # 평가기 생성
    evaluator = BackTranslationEvaluator()
    
    # 증강된 샘플만 추출
    augmented_only = augmented_df[augmented_df['is_augmented'] == True]
    
    # 원본과 증강 텍스트 매칭
    originals = []
    augmented_lists = []
    
    for orig_idx in augmented_only['original_idx'].unique():
        # 원본 텍스트
        orig_text = original_df.loc[orig_idx, 'dialogue']
        originals.append(orig_text)
        
        # 해당 원본의 모든 증강 텍스트
        aug_texts = augmented_only[
            augmented_only['original_idx'] == orig_idx
        ]['dialogue'].tolist()
        augmented_lists.append(aug_texts)
    
    # 샘플 평가 (전체가 너무 많으면 일부만)
    sample_size = min(len(originals), 100)
    sample_indices = np.random.choice(len(originals), sample_size, replace=False)
    
    sample_originals = [originals[i] for i in sample_indices]
    sample_augmented = [augmented_lists[i] for i in sample_indices]
    
    # 평가 실행
    metrics = evaluator.evaluate_batch(sample_originals, sample_augmented)
    
    # 보고서 생성
    report_path = output_dir / 'backtranslation_quality_report.txt'
    report = evaluator.generate_quality_report(
        sample_originals, 
        sample_augmented,
        save_path=str(report_path)
    )
    
    logger.info(f"품질 평가 보고서 저장: {report_path}")
    
    # 분포 분석
    distribution_analysis = analyze_augmentation_distribution(
        original_df, 
        augmented_df,
        text_column='dialogue'
    )
    
    metrics.update(distribution_analysis)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='백트랜슬레이션 데이터 증강')
    parser.add_argument('--config', type=str, required=True,
                       help='실험 설정 파일 경로')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='데이터 디렉토리')
    parser.add_argument('--output-dir', type=str, default='./data/augmented',
                       help='출력 디렉토리')
    parser.add_argument('--evaluate', action='store_true',
                       help='증강 품질 평가 수행')
    
    args = parser.parse_args()
    
    # 설정 로드
    config = load_config(args.config)
    
    # 디렉토리 설정
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 실험별 출력 디렉토리
    exp_name = config['experiment_name']
    exp_output_dir = output_dir / exp_name
    exp_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 데이터 로드
    train_file = data_dir / config['data']['train_file']
    logger.info(f"학습 데이터 로드: {train_file}")
    
    train_df = pd.read_csv(train_file)
    logger.info(f"원본 데이터 크기: {len(train_df)}")
    
    # 백트랜슬레이션 증강
    augmented_df = augment_dataset(train_df, config)
    logger.info(f"증강된 데이터 크기: {len(augmented_df)}")
    
    # 저장
    output_file = exp_output_dir / config['data'].get(
        'augmented_train_file', 'train_augmented.csv'
    )
    augmented_df.to_csv(output_file, index=False)
    logger.info(f"증강된 데이터 저장: {output_file}")
    
    # 통계 저장
    stats = {
        'original_size': len(train_df),
        'augmented_size': len(augmented_df),
        'augmentation_ratio': (len(augmented_df) - len(train_df)) / len(train_df),
        'timestamp': datetime.now().isoformat()
    }
    
    # 품질 평가
    if args.evaluate:
        import numpy as np
        quality_metrics = evaluate_augmentation_quality(
            train_df,
            augmented_df,
            config,
            exp_output_dir
        )
        stats.update(quality_metrics)
    
    # 통계 저장
    stats_file = exp_output_dir / 'augmentation_stats.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        import json
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    logger.info(f"증강 통계 저장: {stats_file}")
    logger.info("백트랜슬레이션 증강 완료!")
    
    # 결과 요약 출력
    print("\n=== 백트랜슬레이션 결과 요약 ===")
    print(f"원본 데이터: {stats['original_size']}개")
    print(f"증강된 데이터: {stats['augmented_size']}개")
    print(f"증강 비율: {stats['augmentation_ratio']:.1%}")
    
    if args.evaluate and 'semantic_similarity' in stats:
        print(f"\n품질 메트릭:")
        print(f"  - 의미 유사도: {stats['semantic_similarity']:.3f}")
        print(f"  - 어휘 다양성: {stats['lexical_diversity']:.3f}")
        print(f"  - 특수 토큰 보존율: {stats['special_token_preservation']:.1%}")


if __name__ == "__main__":
    main()
