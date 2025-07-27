#!/usr/bin/env python3
"""
제출 파일 검증 스크립트
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json


def validate_submission(submission_path: str, sample_path: str) -> dict:
    """제출 파일 검증"""
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'statistics': {}
    }
    
    # 파일 존재 확인
    if not Path(submission_path).exists():
        results['valid'] = False
        results['errors'].append(f"Submission file not found: {submission_path}")
        return results
    
    if not Path(sample_path).exists():
        results['valid'] = False
        results['errors'].append(f"Sample file not found: {sample_path}")
        return results
    
    # 파일 로드
    try:
        submission_df = pd.read_csv(submission_path)
        sample_df = pd.read_csv(sample_path)
    except Exception as e:
        results['valid'] = False
        results['errors'].append(f"Error loading files: {str(e)}")
        return results
    
    # 1. 컬럼 확인
    expected_columns = list(sample_df.columns)
    actual_columns = list(submission_df.columns)
    
    if actual_columns != expected_columns:
        results['valid'] = False
        results['errors'].append(
            f"Column mismatch. Expected: {expected_columns}, Got: {actual_columns}"
        )
    
    # 2. 행 수 확인
    expected_rows = len(sample_df)
    actual_rows = len(submission_df)
    
    if actual_rows != expected_rows:
        results['valid'] = False
        results['errors'].append(
            f"Row count mismatch. Expected: {expected_rows}, Got: {actual_rows}"
        )
    
    # 3. fname 일치 확인
    if 'fname' in submission_df.columns and 'fname' in sample_df.columns:
        if not submission_df['fname'].equals(sample_df['fname']):
            results['valid'] = False
            results['errors'].append("fname values do not match the sample")
        
        # fname 중복 확인
        duplicates = submission_df['fname'].duplicated().sum()
        if duplicates > 0:
            results['valid'] = False
            results['errors'].append(f"Found {duplicates} duplicate fname entries")
    
    # 4. summary 컴럼 검증
    if 'summary' in submission_df.columns:
        # 널 값 확인
        null_count = submission_df['summary'].isnull().sum()
        if null_count > 0:
            results['valid'] = False
            results['errors'].append(f"Found {null_count} null summaries")
        
        # 빈 문자열 확인
        empty_count = (submission_df['summary'] == '').sum()
        if empty_count > 0:
            results['valid'] = False
            results['errors'].append(f"Found {empty_count} empty summaries")
        
        # 길이 통계
        summary_lengths = submission_df['summary'].str.len()
        results['statistics']['summary_length'] = {
            'mean': float(summary_lengths.mean()),
            'std': float(summary_lengths.std()),
            'min': int(summary_lengths.min()),
            'max': int(summary_lengths.max()),
            'median': float(summary_lengths.median())
        }
        
        # 너무 짧은 요약 경고
        very_short = (summary_lengths < 10).sum()
        if very_short > 0:
            results['warnings'].append(
                f"Found {very_short} very short summaries (< 10 chars)"
            )
        
        # 너무 긴 요약 경고
        very_long = (summary_lengths > 500).sum()
        if very_long > 0:
            results['warnings'].append(
                f"Found {very_long} very long summaries (> 500 chars)"
            )
        
        # 특수 토큰 통계
        special_tokens = [
            '#Person1#', '#Person2#', '#Person3#',
            '#PhoneNumber#', '#Address#', '#Email#',
            '#DateOfBirth#', '#SSN#', '#CardNumber#'
        ]
        
        token_stats = {}
        for token in special_tokens:
            count = submission_df['summary'].str.contains(
                token, regex=False, na=False
            ).sum()
            percentage = (count / len(submission_df)) * 100
            token_stats[token] = {
                'count': int(count),
                'percentage': float(percentage)
            }
        
        results['statistics']['special_tokens'] = token_stats
        
        # 화자 토큰이 전혀 없는 경우 경고
        person_tokens = ['#Person1#', '#Person2#', '#Person3#']
        has_any_person = submission_df['summary'].apply(
            lambda x: any(token in str(x) for token in person_tokens)
        ).sum()
        
        if has_any_person < len(submission_df) * 0.5:
            results['warnings'].append(
                f"Only {has_any_person}/{len(submission_df)} summaries "
                f"contain speaker tokens (expected > 50%)"
            )
    
    # 5. 파일 크기 확인
    file_size = Path(submission_path).stat().st_size / (1024 * 1024)  # MB
    results['statistics']['file_size_mb'] = round(file_size, 2)
    
    if file_size > 50:
        results['warnings'].append(f"Large file size: {file_size:.2f} MB")
    
    # 6. 인코딩 확인
    try:
        with open(submission_path, 'r', encoding='utf-8') as f:
            f.read()
    except UnicodeDecodeError:
        results['warnings'].append("File may not be UTF-8 encoded")
    
    return results


def print_validation_results(results: dict):
    """검증 결과 출력"""
    print("\n" + "="*60)
    print("SUBMISSION VALIDATION REPORT")
    print("="*60 + "\n")
    
    # 전체 상태
    if results['valid']:
        print("✅ VALIDATION PASSED")
    else:
        print("❌ VALIDATION FAILED")
    
    # 에러
    if results['errors']:
        print("\n🚨 ERRORS:")
        for error in results['errors']:
            print(f"  - {error}")
    
    # 경고
    if results['warnings']:
        print("\n⚠️  WARNINGS:")
        for warning in results['warnings']:
            print(f"  - {warning}")
    
    # 통계
    if results['statistics']:
        print("\n📊 STATISTICS:")
        
        # 요약 길이
        if 'summary_length' in results['statistics']:
            stats = results['statistics']['summary_length']
            print(f"\n  Summary Length:")
            print(f"    - Mean: {stats['mean']:.1f} chars")
            print(f"    - Std: {stats['std']:.1f}")
            print(f"    - Min: {stats['min']}")
            print(f"    - Max: {stats['max']}")
            print(f"    - Median: {stats['median']:.1f}")
        
        # 특수 토큰
        if 'special_tokens' in results['statistics']:
            print(f"\n  Special Tokens:")
            tokens = results['statistics']['special_tokens']
            for token, stats in sorted(tokens.items(), 
                                      key=lambda x: x[1]['count'], 
                                      reverse=True):
                if stats['count'] > 0:
                    print(f"    - {token}: {stats['count']} "
                          f"({stats['percentage']:.1f}%)")
        
        # 파일 크기
        if 'file_size_mb' in results['statistics']:
            print(f"\n  File Size: {results['statistics']['file_size_mb']} MB")
    
    print("\n" + "="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Validate submission file")
    parser.add_argument("--submission", type=str, required=True,
                       help="Path to submission file")
    parser.add_argument("--sample", type=str, required=True,
                       help="Path to sample submission file")
    parser.add_argument("--output", type=str, default=None,
                       help="Save validation results as JSON")
    
    args = parser.parse_args()
    
    # 검증 실행
    results = validate_submission(args.submission, args.sample)
    
    # 결과 출력
    print_validation_results(results)
    
    # JSON 저장 (선택사항)
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Validation results saved to: {args.output}")
    
    # 종료 코드
    exit(0 if results['valid'] else 1)


if __name__ == "__main__":
    main()
