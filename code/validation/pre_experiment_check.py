#!/opt/conda/bin/python3
"""
실험 전 검증 스크립트

실험 실행 전에 토큰 호환성과 메모리 요구사항을 검증하고
문제가 있으면 자동으로 설정을 수정합니다.
"""

import sys
import os
import argparse
import yaml
import logging
from pathlib import Path
import json
from typing import Dict, Any

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent.parent  # code/validation -> code -> 프로젝트 루트
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "code"))  # code 디렉토리도 추가

from validation.token_validation import validate_model_tokenizer_compatibility, fix_token_range_issues
from validation.memory_validation import estimate_memory_requirements, auto_fix_memory_config, cleanup_between_experiments


def setup_logging():
    """로깅 설정"""
    # validation_logs 디렉토리 경로를 프로젝트 루트 기준으로 설정
    log_dir = project_root / "validation_logs"
    log_dir.mkdir(exist_ok=True)  # 디렉토리가 없으면 생성
    log_file = log_dir / "pre_experiment_check.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(log_file))
        ]
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """설정 파일 로드"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logging.error(f"설정 파일 로드 실패: {e}")
        sys.exit(1)


def save_config(config: Dict[str, Any], output_path: str):
    """수정된 설정 저장"""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
        logging.info(f"수정된 설정 저장: {output_path}")
    except Exception as e:
        logging.error(f"설정 파일 저장 실패: {e}")


def print_validation_report(validation_results: Dict[str, Any]):
    """검증 결과 리포트 출력"""
    print("\n" + "="*60)
    print("🔍 실험 전 검증 결과")
    print("="*60)
    
    # 토큰 검증 결과
    if "token_validation" in validation_results:
        token_result = validation_results["token_validation"]
        print(f"\n📝 토큰 호환성 검증:")
        print(f"   상태: {'✅ 통과' if token_result.get('overall_valid', False) else '❌ 실패'}")
        
        if token_result.get("vocabulary_validation"):
            vocab = token_result["vocabulary_validation"]
            print(f"   토크나이저 vocab: {vocab.get('tokenizer_vocab_size', 'N/A')}")
            print(f"   모델 vocab: {vocab.get('model_vocab_size', 'N/A')}")
            print(f"   특수 토큰: {vocab.get('special_token_count', 0)}개")
        
        if token_result.get("recommendations"):
            print("   권장사항:")
            for rec in token_result["recommendations"][:3]:
                print(f"     - {rec}")
    
    # 메모리 검증 결과
    if "memory_validation" in validation_results:
        memory_result = validation_results["memory_validation"]
        print(f"\n💾 메모리 요구사항 검증:")
        print(f"   상태: {'✅ 충분' if memory_result.get('memory_sufficient', False) else '⚠️ 부족'}")
        print(f"   예상 사용량: {memory_result.get('estimated_memory_gb', 0):.1f}GB")
        print(f"   사용 가능: {memory_result.get('available_memory_gb', 0):.1f}GB")
        print(f"   사용률: {memory_result.get('memory_utilization_percent', 0):.1f}%")
        
        if memory_result.get("recommendations"):
            print("   권장사항:")
            for rec in memory_result["recommendations"][:3]:
                print(f"     - {rec}")
    
    # 전체 상태
    overall_valid = (
        validation_results.get("token_validation", {}).get("overall_valid", False) and
        validation_results.get("memory_validation", {}).get("memory_sufficient", False)
    )
    
    print(f"\n🎯 전체 검증 결과: {'✅ 실험 실행 가능' if overall_valid else '❌ 문제 해결 필요'}")
    
    if validation_results.get("config_modified", False):
        print("🔧 설정이 자동으로 수정되었습니다")
        print("   수정된 설정 파일을 확인하세요")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="실험 전 검증 실행")
    parser.add_argument("--config", required=True, help="실험 설정 파일 경로")
    parser.add_argument("--output", help="수정된 설정 저장 경로 (선택사항)")
    parser.add_argument("--auto-fix", action="store_true", help="문제 자동 수정")
    parser.add_argument("--cleanup", action="store_true", help="시작 전 GPU 메모리 정리")
    parser.add_argument("--verbose", action="store_true", help="상세 로그 출력")
    
    args = parser.parse_args()
    
    # 로깅 설정
    setup_logging()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger = logging.getLogger(__name__)
    logger.info("🚀 실험 전 검증 시작")
    
    # 로그 디렉토리 생성
    (project_root / "validation_logs").mkdir(exist_ok=True)
    
    # GPU 메모리 정리 (요청시)
    if args.cleanup:
        logger.info("🧹 GPU 메모리 정리 중...")
        cleanup_success = cleanup_between_experiments()
        if cleanup_success:
            logger.info("✅ GPU 메모리 정리 완료")
        else:
            logger.warning("⚠️ GPU 메모리 정리 부분 실패")
    
    # 설정 로드
    config = load_config(args.config)
    model_name = config.get('general', {}).get('model_name', '')
    
    logger.info(f"📋 검증 대상: {model_name}")
    logger.info(f"📁 설정 파일: {args.config}")
    
    validation_results = {}
    config_modified = False
    final_config = config.copy()
    
    try:
        # 1. 토큰 호환성 검증
        logger.info("🔍 토큰 호환성 검증 중...")
        
        if args.auto_fix:
            token_fix_result = fix_token_range_issues(model_name, config)
            validation_results["token_validation"] = token_fix_result["validation_result"]
            
            if token_fix_result["config_modified"]:
                final_config = token_fix_result["fixed_config"]
                config_modified = True
                logger.info("🔧 토큰 관련 설정 자동 수정 완료")
        else:
            token_result = validate_model_tokenizer_compatibility(model_name, config)
            validation_results["token_validation"] = token_result
        
        # 2. 메모리 요구사항 검증
        logger.info("💾 메모리 요구사항 검증 중...")
        
        if args.auto_fix:
            memory_fixed_config, memory_config_modified = auto_fix_memory_config(final_config)
            if memory_config_modified:
                final_config = memory_fixed_config
                config_modified = True
                logger.info("🔧 메모리 관련 설정 자동 수정 완료")
        
        memory_result = estimate_memory_requirements(final_config)
        validation_results["memory_validation"] = memory_result
        validation_results["config_modified"] = config_modified
        
        # 3. 결과 출력
        print_validation_report(validation_results)
        
        # 4. 수정된 설정 저장 (필요시)
        if config_modified:
            if args.output:
                save_config(final_config, args.output)
            else:
                # 기본 저장 경로: 원본 파일명에 _fixed 추가
                config_path = Path(args.config)
                output_path = config_path.parent / f"{config_path.stem}_fixed{config_path.suffix}"
                save_config(final_config, str(output_path))
        
        # 5. 검증 결과 저장
        result_path = project_root / "validation_logs" / "last_validation_result.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(validation_results, f, indent=2, ensure_ascii=False, default=str)
        
        # 6. 종료 코드 결정
        overall_valid = (
            validation_results.get("token_validation", {}).get("overall_valid", False) and
            validation_results.get("memory_validation", {}).get("memory_sufficient", False)
        )
        
        if overall_valid:
            logger.info("✅ 모든 검증 통과 - 실험 실행 가능")
            sys.exit(0)
        else:
            logger.error("❌ 검증 실패 - 문제 해결 후 재시도")
            if not args.auto_fix:
                logger.info("💡 --auto-fix 옵션을 사용하여 자동 수정을 시도해보세요")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"검증 중 예외 발생: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
