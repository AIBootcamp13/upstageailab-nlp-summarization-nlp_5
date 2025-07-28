#!/usr/bin/env python
"""
추론 CLI 도구

대회 제출을 위한 독립 실행 가능한 추론 스크립트입니다.
모델과 입력 파일을 받아 요약을 생성하고 제출 형식으로 저장합니다.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

from core.inference import InferenceEngine, InferenceConfig
from utils.path_utils import PathManager, path_manager


def setup_logging(verbose: bool = False):
    """로깅 설정"""
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def parse_arguments():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(
        description='NLP 대화 요약 추론 도구',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 기본 추론 실행
  python run_inference.py --model_path gogamza/kobart-summarization --input_file data/test.csv

  # 배치 크기와 출력 파일 지정
  python run_inference.py --model_path models/best_model --input_file data/test.csv --output_file results/submission.csv --batch_size 16

  # FP16 사용 및 상세 로그
  python run_inference.py --model_path gogamza/kobart-summarization --input_file data/test.csv --fp16 --verbose
        """
    )
    
    # 필수 인자
    parser.add_argument(
        '--model_path', '-m',
        type=str,
        required=True,
        help='모델 경로 (로컬 경로 또는 HuggingFace Hub 모델명)'
    )
    
    parser.add_argument(
        '--input_file', '-i',
        type=str,
        required=True,
        help='입력 CSV 파일 경로'
    )
    
    # 선택 인자
    parser.add_argument(
        '--output_file', '-o',
        type=str,
        default=None,
        help='출력 CSV 파일 경로 (기본값: results/submission_YYYYMMDD_HHMMSS.csv)'
    )
    
    parser.add_argument(
        '--batch_size', '-b',
        type=int,
        default=8,
        help='배치 크기 (기본값: 8, 디바이스별로 자동 조정됨)'
    )
    
    parser.add_argument(
        '--max_source_length',
        type=int,
        default=1024,
        help='최대 입력 길이 (기본값: 1024)'
    )
    
    parser.add_argument(
        '--max_target_length',
        type=int,
        default=256,
        help='최대 출력 길이 (기본값: 256)'
    )
    
    parser.add_argument(
        '--num_beams',
        type=int,
        default=5,
        help='빔 서치 크기 (기본값: 5)'
    )
    
    parser.add_argument(
        '--length_penalty',
        type=float,
        default=1.0,
        help='길이 패널티 (기본값: 1.0)'
    )
    
    parser.add_argument(
        '--dialogue_column',
        type=str,
        default='dialogue',
        help='대화가 포함된 컬럼명 (기본값: dialogue)'
    )
    
    parser.add_argument(
        '--fname_column',
        type=str,
        default='fname',
        help='파일명이 포함된 컬럼명 (기본값: fname)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='사용할 디바이스 (기본값: 자동 감지)'
    )
    
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='FP16 (half precision) 사용'
    )
    
    parser.add_argument(
        '--no_progress',
        action='store_true',
        help='진행률 표시 비활성화'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='상세 로그 출력'
    )
    
    return parser.parse_args()


def validate_input_file(file_path: str, required_columns: list) -> pd.DataFrame:
    """
    입력 파일 검증
    
    Args:
        file_path: 입력 파일 경로
        required_columns: 필수 컬럼 리스트
        
    Returns:
        로드된 DataFrame
    """
    # 경로 해결
    file_path = path_manager.resolve_path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {file_path}")
    
    # CSV 로드
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except Exception as e:
        raise ValueError(f"CSV 파일 로드 실패: {e}")
    
    # 필수 컬럼 확인
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"필수 컬럼이 없습니다: {missing_columns}")
    
    logging.info(f"입력 파일 로드 완료: {len(df)}개 행")
    return df


def main():
    """메인 실행 함수"""
    # 인자 파싱
    args = parse_arguments()
    
    # 로깅 설정
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("=== NLP 대화 요약 추론 시작 ===")
    logger.info(f"모델: {args.model_path}")
    logger.info(f"입력: {args.input_file}")
    
    try:
        # 입력 파일 검증 및 로드
        required_columns = [args.dialogue_column, args.fname_column]
        df = validate_input_file(args.input_file, required_columns)
        
        # 추론 설정 생성
        config = InferenceConfig(
            model_path=args.model_path,
            batch_size=args.batch_size,
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length,
            num_beams=args.num_beams,
            length_penalty=args.length_penalty,
            device=args.device,
            fp16=args.fp16
        )
        
        # 추론 엔진 생성
        logger.info("추론 엔진 초기화 중...")
        engine = InferenceEngine(config)
        
        # 추론 실행
        logger.info(f"추론 시작 (배치 크기: {engine.config.batch_size})...")
        result_df = engine.predict_from_dataframe(
            df,
            dialogue_column=args.dialogue_column,
            output_column='summary',
            show_progress=not args.no_progress
        )
        
        # 출력 파일 경로 설정
        if args.output_file:
            output_path = args.output_file
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"results/submission_{timestamp}.csv"
        
        # 결과 저장
        engine.save_submission(
            result_df,
            output_path,
            fname_column=args.fname_column,
            summary_column='summary'
        )
        
        # 완료 메시지
        logger.info("=== 추론 완료 ===")
        logger.info(f"출력 파일: {path_manager.resolve_path(output_path)}")
        logger.info(f"처리된 대화: {len(result_df)}개")
        
        # 샘플 출력
        if args.verbose and len(result_df) > 0:
            logger.info("\n--- 샘플 결과 (첫 3개) ---")
            for idx, row in result_df.head(3).iterrows():
                logger.info(f"\n[{idx+1}] {row[args.fname_column]}")
                logger.info(f"대화: {row[args.dialogue_column][:100]}...")
                logger.info(f"요약: {row['summary']}")
        
    except KeyboardInterrupt:
        logger.warning("사용자에 의해 중단됨")
        sys.exit(1)
    except Exception as e:
        logger.error(f"오류 발생: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
