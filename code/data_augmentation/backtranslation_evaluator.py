"""
백트랜슬레이션 품질 평가 모듈

생성된 백트랜슬레이션의 품질을 다양한 메트릭으로 평가합니다.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import pandas as pd
from collections import Counter
import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch


class BackTranslationEvaluator:
    """백트랜슬레이션 품질 평가 클래스"""
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Args:
            model_name: 의미 유사도 계산을 위한 문장 임베딩 모델
        """
        self.sentence_model = SentenceTransformer(model_name)
        
    def evaluate_batch(
        self,
        originals: List[str],
        augmented: List[List[str]]
    ) -> Dict[str, float]:
        """
        배치 단위로 백트랜슬레이션 품질 평가
        
        Args:
            originals: 원본 텍스트 리스트
            augmented: 각 원본에 대한 증강 텍스트 리스트
            
        Returns:
            metrics: 평가 메트릭 딕셔너리
        """
        metrics = {
            'semantic_similarity': [],
            'lexical_diversity': [],
            'length_preservation': [],
            'special_token_preservation': [],
            'grammaticality_score': []
        }
        
        for orig, aug_list in zip(originals, augmented):
            for aug in aug_list:
                # 의미 유사도
                sim = self.compute_semantic_similarity(orig, aug)
                metrics['semantic_similarity'].append(sim)
                
                # 어휘 다양성
                div = self.compute_lexical_diversity(orig, aug)
                metrics['lexical_diversity'].append(div)
                
                # 길이 보존율
                len_pres = self.compute_length_preservation(orig, aug)
                metrics['length_preservation'].append(len_pres)
                
                # 특수 토큰 보존
                token_pres = self.check_special_token_preservation(orig, aug)
                metrics['special_token_preservation'].append(token_pres)
                
                # 문법성 점수 (간단한 휴리스틱)
                gram_score = self.compute_grammaticality_score(aug)
                metrics['grammaticality_score'].append(gram_score)
        
        # 평균 계산
        avg_metrics = {
            key: np.mean(values) if values else 0.0
            for key, values in metrics.items()
        }
        
        # 추가 통계
        avg_metrics.update({
            'semantic_similarity_std': np.std(metrics['semantic_similarity']) if metrics['semantic_similarity'] else 0.0,
            'diversity_score': self.compute_overall_diversity(augmented),
            'augmentation_success_rate': self.compute_success_rate(originals, augmented)
        })
        
        return avg_metrics
    
    def compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """의미적 유사도 계산"""
        embeddings = self.sentence_model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
    
    def compute_lexical_diversity(self, original: str, augmented: str) -> float:
        """어휘 다양성 계산 (원본과 다른 단어 비율)"""
        orig_words = set(original.lower().split())
        aug_words = set(augmented.lower().split())
        
        # 특수 토큰 제외
        special_token_pattern = r'#\w+#'
        orig_words = {w for w in orig_words if not re.match(special_token_pattern, w)}
        aug_words = {w for w in aug_words if not re.match(special_token_pattern, w)}
        
        if not orig_words:
            return 0.0
        
        different_words = aug_words - orig_words
        diversity = len(different_words) / len(orig_words)
        
        return min(diversity, 1.0)  # 최대 1.0
    
    def compute_length_preservation(self, original: str, augmented: str) -> float:
        """길이 보존율 계산"""
        orig_len = len(original.split())
        aug_len = len(augmented.split())
        
        if orig_len == 0:
            return 0.0
        
        ratio = aug_len / orig_len
        # 0.7 ~ 1.3 범위를 1.0으로 매핑
        if 0.7 <= ratio <= 1.3:
            return 1.0
        elif ratio < 0.7:
            return ratio / 0.7
        else:
            return 1.3 / ratio
    
    def check_special_token_preservation(self, original: str, augmented: str) -> float:
        """특수 토큰 보존 확인"""
        special_token_pattern = r'#\w+#'
        
        orig_tokens = set(re.findall(special_token_pattern, original))
        aug_tokens = set(re.findall(special_token_pattern, augmented))
        
        if not orig_tokens:
            return 1.0  # 원본에 특수 토큰이 없으면 보존된 것으로 간주
        
        # 모든 토큰이 보존되었는지 확인
        preserved = orig_tokens == aug_tokens
        return 1.0 if preserved else 0.0
    
    def compute_grammaticality_score(self, text: str) -> float:
        """문법성 점수 계산 (휴리스틱)"""
        score = 1.0
        
        # 기본적인 문법 체크
        # 1. 문장 부호 체크
        if not re.search(r'[.!?]$', text.strip()):
            score -= 0.1
        
        # 2. 괄호 짝 맞춤
        if text.count('(') != text.count(')'):
            score -= 0.2
        
        # 3. 따옴표 짝 맞춤
        if text.count('"') % 2 != 0:
            score -= 0.1
        
        # 4. 연속된 공백
        if re.search(r'\s{2,}', text):
            score -= 0.1
        
        # 5. 문장 시작 대문자 (영어의 경우)
        sentences = re.split(r'[.!?]\s+', text)
        for sent in sentences:
            if sent and sent[0].isalpha() and sent[0].islower():
                score -= 0.05
        
        return max(score, 0.0)
    
    def compute_overall_diversity(self, augmented_sets: List[List[str]]) -> float:
        """전체 증강 데이터의 다양성 계산"""
        all_augmented = []
        for aug_list in augmented_sets:
            all_augmented.extend(aug_list)
        
        if len(all_augmented) < 2:
            return 0.0
        
        # 모든 증강 텍스트 간의 평균 유사도
        embeddings = self.sentence_model.encode(all_augmented)
        
        total_similarity = 0
        count = 0
        
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                total_similarity += sim
                count += 1
        
        if count == 0:
            return 0.0
        
        avg_similarity = total_similarity / count
        diversity = 1.0 - avg_similarity  # 유사도가 낮을수록 다양성이 높음
        
        return diversity
    
    def compute_success_rate(self, originals: List[str], augmented: List[List[str]]) -> float:
        """증강 성공률 계산"""
        total = len(originals)
        success = sum(1 for aug_list in augmented if len(aug_list) > 0)
        
        return success / total if total > 0 else 0.0
    
    def generate_quality_report(
        self,
        originals: List[str],
        augmented: List[List[str]],
        save_path: Optional[str] = None
    ) -> str:
        """
        품질 평가 보고서 생성
        
        Args:
            originals: 원본 텍스트
            augmented: 증강된 텍스트
            save_path: 보고서 저장 경로
            
        Returns:
            report: 보고서 문자열
        """
        metrics = self.evaluate_batch(originals, augmented)
        
        report = []
        report.append("=== 백트랜슬레이션 품질 평가 보고서 ===\n")
        
        report.append("## 전체 요약")
        report.append(f"- 평가 샘플 수: {len(originals)}")
        report.append(f"- 증강 성공률: {metrics['augmentation_success_rate']:.2%}")
        report.append(f"- 평균 의미 유사도: {metrics['semantic_similarity']:.3f} (±{metrics['semantic_similarity_std']:.3f})")
        report.append(f"- 전체 다양성 점수: {metrics['diversity_score']:.3f}")
        report.append("")
        
        report.append("## 세부 메트릭")
        report.append(f"- 어휘 다양성: {metrics['lexical_diversity']:.3f}")
        report.append(f"- 길이 보존율: {metrics['length_preservation']:.3f}")
        report.append(f"- 특수 토큰 보존율: {metrics['special_token_preservation']:.2%}")
        report.append(f"- 문법성 점수: {metrics['grammaticality_score']:.3f}")
        report.append("")
        
        # 품질 등급 부여
        overall_score = (
            metrics['semantic_similarity'] * 0.3 +
            metrics['lexical_diversity'] * 0.2 +
            metrics['length_preservation'] * 0.2 +
            metrics['special_token_preservation'] * 0.2 +
            metrics['grammaticality_score'] * 0.1
        )
        
        if overall_score >= 0.8:
            grade = "A (우수)"
        elif overall_score >= 0.7:
            grade = "B (양호)"
        elif overall_score >= 0.6:
            grade = "C (보통)"
        else:
            grade = "D (개선 필요)"
        
        report.append(f"## 종합 평가")
        report.append(f"- 종합 점수: {overall_score:.3f}")
        report.append(f"- 품질 등급: {grade}")
        report.append("")
        
        # 샘플 예시
        report.append("## 증강 예시 (상위 3개)")
        for i in range(min(3, len(originals))):
            if augmented[i]:
                report.append(f"\n### 예시 {i+1}")
                report.append(f"원본: {originals[i][:100]}...")
                report.append(f"증강: {augmented[i][0][:100]}...")
                sim = self.compute_semantic_similarity(originals[i], augmented[i][0])
                report.append(f"유사도: {sim:.3f}")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
        
        return report_text


def analyze_augmentation_distribution(
    df_original: pd.DataFrame,
    df_augmented: pd.DataFrame,
    text_column: str = 'dialogue'
) -> Dict[str, any]:
    """증강 데이터의 분포 분석"""
    
    analysis = {}
    
    # 증강 비율
    augmentation_ratio = (len(df_augmented) - len(df_original)) / len(df_original)
    analysis['augmentation_ratio'] = augmentation_ratio
    
    # 길이 분포 비교
    orig_lengths = df_original[text_column].str.split().str.len()
    aug_lengths = df_augmented[text_column].str.split().str.len()
    
    analysis['length_stats'] = {
        'original_mean': orig_lengths.mean(),
        'original_std': orig_lengths.std(),
        'augmented_mean': aug_lengths.mean(),
        'augmented_std': aug_lengths.std()
    }
    
    # 특수 토큰 분포
    special_token_pattern = r'#\w+#'
    
    orig_token_counts = df_original[text_column].str.findall(special_token_pattern).str.len()
    aug_token_counts = df_augmented[text_column].str.findall(special_token_pattern).str.len()
    
    analysis['special_token_stats'] = {
        'original_mean': orig_token_counts.mean(),
        'augmented_mean': aug_token_counts.mean(),
        'preservation_rate': (aug_token_counts > 0).mean()
    }
    
    return analysis


if __name__ == "__main__":
    # 테스트
    evaluator = BackTranslationEvaluator()
    
    # 샘플 데이터
    originals = [
        "#Person1#: 오늘 회의는 3시에 시작합니다. #Person2#: 알겠습니다.",
        "#Person1#: 프로젝트 진행 상황은 어떻습니까? #Person2#: 순조롭게 진행 중입니다."
    ]
    
    augmented = [
        ["#Person1#: 회의가 오늘 오후 3시에 시작됩니다. #Person2#: 네, 알겠습니다."],
        ["#Person1#: 프로젝트가 어떻게 진행되고 있나요? #Person2#: 잘 진행되고 있습니다."]
    ]
    
    # 평가
    metrics = evaluator.evaluate_batch(originals, augmented)
    print("평가 메트릭:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.3f}")
    
    # 보고서 생성
    report = evaluator.generate_quality_report(originals, augmented)
    print("\n" + report)
