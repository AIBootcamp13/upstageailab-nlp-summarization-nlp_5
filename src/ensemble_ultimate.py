#!/usr/bin/env python
"""
🏆 Ultimate Ensemble Strategy
여러 모델 결과를 조합하여 최고 성능 달성
"""

import pandas as pd
import numpy as np
from rouge import Rouge
import os
import sys
import yaml
from collections import Counter
import re

sys.path.append("/data/ephemeral/home/nlp-5/lyj")

def load_submissions(paths):
    """여러 submission 파일 로드"""
    submissions = {}
    for name, path in paths.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            submissions[name] = df
            print(f"✅ Loaded {name}: {len(df)} samples")
        else:
            print(f"❌ Missing {name}: {path}")
    return submissions

def ensemble_by_length_and_rouge(submissions):
    """길이와 품질 기반 앙상블"""
    result = []
    
    for i in range(len(list(submissions.values())[0])):
        candidates = []
        
        for name, df in submissions.items():
            summary = df.iloc[i]['summary']
            candidates.append({
                'summary': summary,
                'length': len(summary),
                'source': name
            })
        
        # 길이 기준으로 필터링 (너무 짧거나 긴 것 제외)
        lengths = [c['length'] for c in candidates]
        mean_len = np.mean(lengths)
        std_len = np.std(lengths)
        
        filtered = [c for c in candidates 
                   if abs(c['length'] - mean_len) <= 1.5 * std_len]
        
        if not filtered:
            filtered = candidates
        
        # 가장 적절한 길이의 요약 선택 (median 길이에 가까운 것)
        target_length = np.median([c['length'] for c in filtered])
        best_candidate = min(filtered, 
                           key=lambda x: abs(x['length'] - target_length))
        
        result.append(best_candidate['summary'])
    
    return result

def ensemble_by_voting(submissions):
    """투표 기반 앙상블 (키워드 중심)"""
    result = []
    
    for i in range(len(list(submissions.values())[0])):
        summaries = [df.iloc[i]['summary'] for df in submissions.values()]
        
        # 각 요약에서 핵심 단어 추출
        all_words = []
        for summary in summaries:
            words = re.findall(r'#Person\d+#|[가-힣]+', summary)
            all_words.extend(words)
        
        # 가장 빈도가 높은 단어들을 포함한 요약 선택
        word_counts = Counter(all_words)
        common_words = set([word for word, count in word_counts.most_common(10)])
        
        best_summary = summaries[0]
        best_score = 0
        
        for summary in summaries:
            summary_words = set(re.findall(r'#Person\d+#|[가-힣]+', summary))
            overlap = len(summary_words & common_words)
            if overlap > best_score:
                best_score = overlap
                best_summary = summary
        
        result.append(best_summary)
    
    return result

def create_ultimate_ensemble():
    """최종 앙상블 실행"""
    
    # 가능한 모든 결과 파일 경로
    submission_paths = {
        'optimized': './outputs/exp_optimized_lyj/submission_lyj.csv/result1',
        'quick_boost': './outputs/exp_quick_boost_lyj/submission_quick.csv/result1',
        'ultimate': './outputs/exp_ultimate_lyj/submission_ultimate.csv/result1',
        'final_boost': './outputs/exp_final_boost_lyj/submission_final.csv/result1',
    }
    
    # 파일 로드
    submissions = load_submissions(submission_paths)
    
    if len(submissions) < 2:
        print("❌ 앙상블할 파일이 부족합니다")
        return None
    
    print(f"🔥 {len(submissions)} 모델로 앙상블 시작")
    
    # 기본 템플릿 (첫 번째 파일 구조 사용)
    base_df = list(submissions.values())[0].copy()
    
    # 방법 1: 길이 기반 앙상블
    ensemble_1 = ensemble_by_length_and_rouge(submissions)
    
    # 방법 2: 투표 기반 앙상블  
    ensemble_2 = ensemble_by_voting(submissions)
    
    # 최종 결과 저장
    results = []
    
    # 길이 기반 앙상블
    result_df_1 = base_df.copy()
    result_df_1['summary'] = ensemble_1
    output_path_1 = './outputs/ensemble_length_based.csv'
    result_df_1.to_csv(output_path_1, index=False)
    results.append(('Length-based Ensemble', output_path_1))
    
    # 투표 기반 앙상블
    result_df_2 = base_df.copy()
    result_df_2['summary'] = ensemble_2
    output_path_2 = './outputs/ensemble_voting_based.csv'
    result_df_2.to_csv(output_path_2, index=False)
    results.append(('Voting-based Ensemble', output_path_2))
    
    print("🎉 앙상블 완료!")
    for name, path in results:
        print(f"  📁 {name}: {path}")
        
    return results

if __name__ == "__main__":
    create_ultimate_ensemble()
