"""
메트릭 계산 유틸리티

NLP 대화 요약 프로젝트를 위한 평가 메트릭 계산 기능을 제공합니다.
특히 대회 특성인 Multi-reference ROUGE 평가를 지원합니다.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
from rouge import Rouge
from collections import defaultdict
import re
from dataclasses import dataclass


@dataclass
class RougeScore:
    """ROUGE 점수 결과 클래스"""
    precision: float
    recall: float
    f1: float
    
    def to_dict(self) -> Dict[str, float]:
        """딕셔너리 형태로 변환"""
        return {
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1
        }


@dataclass
class EvaluationResult:
    """평가 결과 클래스"""
    rouge1: RougeScore
    rouge2: RougeScore
    rougeL: RougeScore
    rouge_combined_f1: float  # WandB Sweep 최적화 목표
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 형태로 변환"""
        return {
            'rouge1': self.rouge1.to_dict(),
            'rouge2': self.rouge2.to_dict(),
            'rougeL': self.rougeL.to_dict(),
            'rouge_combined_f1': self.rouge_combined_f1,
            'rouge1_f1': self.rouge1.f1,
            'rouge2_f1': self.rouge2.f1,
            'rougeL_f1': self.rougeL.f1
        }


class KoreanTextProcessor:
    """
    한국어 텍스트 처리기
    
    ROUGE 계산을 위한 한국어 텍스트 토큰화 및 정규화를 담당합니다.
    """
    
    def __init__(self):
        """KoreanTextProcessor 초기화"""
        self.logger = logging.getLogger(__name__)
        self._compile_patterns()
    
    def _compile_patterns(self):
        """정규 표현식 패턴 컴파일"""
        # 특수 문자 제거 패턴
        self.special_chars_pattern = re.compile(r'[^\w\s가-힣]')
        
        # 연속 공백 패턴
        self.whitespace_pattern = re.compile(r'\s+')
        
        # 영숫자 패턴
        self.alphanumeric_pattern = re.compile(r'[a-zA-Z0-9]+')
    
    def tokenize(self, text: str, use_morpheme: bool = False) -> List[str]:
        """
        한국어 텍스트 토큰화
        
        Args:
            text: 입력 텍스트
            use_morpheme: 형태소 분석 사용 여부 (현재는 어절 단위)
            
        Returns:
            토큰 리스트
        """
        if not text or not isinstance(text, str):
            return []
        
        # 기본 정규화
        text = text.strip().lower()
        
        # 특수 문자 제거 (한글, 영숫자, 공백만 유지)
        text = self.special_chars_pattern.sub(' ', text)
        
        # 연속 공백 정규화
        text = self.whitespace_pattern.sub(' ', text).strip()
        
        if not text:
            return []
        
        if use_morpheme:
            # 향후 konlpy 등을 활용한 형태소 분석 구현 예정
            # 현재는 어절 단위 분할 사용
            return self._split_by_eojeol(text)
        else:
            return self._split_by_eojeol(text)
    
    def _split_by_eojeol(self, text: str) -> List[str]:
        """
        어절 단위 분할
        
        Args:
            text: 입력 텍스트
            
        Returns:
            어절 토큰 리스트
        """
        tokens = text.split()
        return [token for token in tokens if token.strip()]
    
    def normalize_text(self, text: str) -> str:
        """
        텍스트 정규화
        
        Args:
            text: 입력 텍스트
            
        Returns:
            정규화된 텍스트
        """
        if not text or not isinstance(text, str):
            return ""
        
        # 기본 정규화
        text = text.strip().lower()
        
        # 특수 문자 및 연속 공백 처리
        text = self.special_chars_pattern.sub(' ', text)
        text = self.whitespace_pattern.sub(' ', text).strip()
        
        return text


class RougeCalculator:
    """
    ROUGE 점수 계산기
    
    단일 참조 및 다중 참조 ROUGE 점수 계산을 지원합니다.
    """
    
    def __init__(self, use_korean_tokenizer: bool = True, 
                 use_stemmer: bool = False):
        """
        RougeCalculator 초기화
        
        Args:
            use_korean_tokenizer: 한국어 토크나이저 사용 여부
            use_stemmer: 스테머 사용 여부
        """
        self.use_korean_tokenizer = use_korean_tokenizer
        self.use_stemmer = use_stemmer
        self.logger = logging.getLogger(__name__)
        
        # ROUGE 객체 초기화
        self.rouge = Rouge()
        
        # 한국어 텍스트 처리기
        if self.use_korean_tokenizer:
            self.korean_processor = KoreanTextProcessor()
    
    def calculate_single_reference(self, prediction: str, 
                                 reference: str) -> EvaluationResult:
        """
        단일 참조 ROUGE 점수 계산
        
        Args:
            prediction: 예측 요약문
            reference: 참조 요약문
            
        Returns:
            평가 결과
        """
        # 텍스트 전처리
        pred_processed = self._preprocess_text(prediction)
        ref_processed = self._preprocess_text(reference)
        
        # 빈 텍스트 처리
        if not pred_processed or not ref_processed:
            return self._create_zero_score()
        
        try:
            # ROUGE 점수 계산
            scores = self.rouge.get_scores(pred_processed, ref_processed)[0]
            
            # 결과 변환
            rouge1 = RougeScore(
                precision=scores['rouge-1']['p'],
                recall=scores['rouge-1']['r'],
                f1=scores['rouge-1']['f']
            )
            rouge2 = RougeScore(
                precision=scores['rouge-2']['p'],
                recall=scores['rouge-2']['r'],
                f1=scores['rouge-2']['f']
            )
            rougeL = RougeScore(
                precision=scores['rouge-l']['p'],
                recall=scores['rouge-l']['r'],
                f1=scores['rouge-l']['f']
            )
            
            # 결합 점수 계산 (WandB Sweep 목표)
            rouge_combined_f1 = rouge1.f1 + rouge2.f1 + rougeL.f1
            
            return EvaluationResult(
                rouge1=rouge1,
                rouge2=rouge2,
                rougeL=rougeL,
                rouge_combined_f1=rouge_combined_f1
            )
            
        except Exception as e:
            self.logger.warning(f"ROUGE calculation failed: {e}")
            return self._create_zero_score()
    
    def calculate_multi_reference(self, prediction: str, 
                                references: List[str]) -> EvaluationResult:
        """
        다중 참조 ROUGE 점수 계산 (대회 특성)
        
        각 참조와의 점수를 계산하여 최고 점수를 반환합니다.
        
        Args:
            prediction: 예측 요약문
            references: 참조 요약문 리스트 (보통 3개)
            
        Returns:
            최고 점수 평가 결과
        """
        if not references:
            return self._create_zero_score()
        
        # 각 참조와의 점수 계산
        scores = []
        for ref in references:
            score = self.calculate_single_reference(prediction, ref)
            scores.append(score)
        
        # 각 메트릭별로 최고 점수 선택
        best_rouge1_f1 = max(score.rouge1.f1 for score in scores)
        best_rouge2_f1 = max(score.rouge2.f1 for score in scores)
        best_rougeL_f1 = max(score.rougeL.f1 for score in scores)
        
        # 최고 점수를 가진 결과들에서 precision, recall 추출
        best_rouge1_idx = max(range(len(scores)), key=lambda i: scores[i].rouge1.f1)
        best_rouge2_idx = max(range(len(scores)), key=lambda i: scores[i].rouge2.f1)
        best_rougeL_idx = max(range(len(scores)), key=lambda i: scores[i].rougeL.f1)
        
        # 최종 결과 구성
        rouge1 = RougeScore(
            precision=scores[best_rouge1_idx].rouge1.precision,
            recall=scores[best_rouge1_idx].rouge1.recall,
            f1=best_rouge1_f1
        )
        rouge2 = RougeScore(
            precision=scores[best_rouge2_idx].rouge2.precision,
            recall=scores[best_rouge2_idx].rouge2.recall,
            f1=best_rouge2_f1
        )
        rougeL = RougeScore(
            precision=scores[best_rougeL_idx].rougeL.precision,
            recall=scores[best_rougeL_idx].rougeL.recall,
            f1=best_rougeL_f1
        )
        
        # 결합 점수 (대회 최종 점수)
        rouge_combined_f1 = best_rouge1_f1 + best_rouge2_f1 + best_rougeL_f1
        
        return EvaluationResult(
            rouge1=rouge1,
            rouge2=rouge2,
            rougeL=rougeL,
            rouge_combined_f1=rouge_combined_f1
        )
    
    def _preprocess_text(self, text: str) -> str:
        """
        ROUGE 계산용 텍스트 전처리
        
        Args:
            text: 입력 텍스트
            
        Returns:
            전처리된 텍스트
        """
        if not text or not isinstance(text, str):
            return ""
        
        if self.use_korean_tokenizer:
            # 한국어 토크나이저 사용
            tokens = self.korean_processor.tokenize(text)
            return ' '.join(tokens)
        else:
            # 기본 전처리
            return text.strip().lower()
    
    def _create_zero_score(self) -> EvaluationResult:
        """
        0점 결과 생성
        
        Returns:
            모든 점수가 0인 평가 결과
        """
        zero_rouge = RougeScore(precision=0.0, recall=0.0, f1=0.0)
        return EvaluationResult(
            rouge1=zero_rouge,
            rouge2=zero_rouge,
            rougeL=zero_rouge,
            rouge_combined_f1=0.0
        )


class MultiReferenceROUGE:
    """
    Multi-reference ROUGE 평가기 (HuggingFace Trainer 호환)
    
    대회 특성인 3개 참조 요약문을 지원하는 평가 클래스입니다.
    HuggingFace Trainer의 compute_metrics 함수와 호환됩니다.
    """
    
    def __init__(self, tokenizer, use_korean_tokenizer: bool = True):
        """
        MultiReferenceROUGE 초기화
        
        Args:
            tokenizer: 토크나이저 (토큰 디코딩용)
            use_korean_tokenizer: 한국어 토크나이저 사용 여부
        """
        self.tokenizer = tokenizer
        self.rouge_calculator = RougeCalculator(use_korean_tokenizer=use_korean_tokenizer)
        self.logger = logging.getLogger(__name__)
    
    def compute_metrics(self, eval_preds) -> Dict[str, float]:
        """
        HuggingFace Trainer 호환 메트릭 계산 함수
        
        Args:
            eval_preds: (predictions, labels) 튜플
            
        Returns:
            메트릭 딕셔너리
        """
        predictions, labels = eval_preds
        
        # 토큰 디코딩
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        # 예측 텍스트 디코딩
        pred_texts = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        
        # 레이블 텍스트 디코딩 (-100 토큰 제외)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        label_texts = self.tokenizer.batch_decode(
            labels, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        
        # ROUGE 점수 계산
        rouge_scores = []
        for pred, ref in zip(pred_texts, label_texts):
            # 현재는 단일 참조로 계산 (다중 참조는 별도 처리 필요)
            score = self.rouge_calculator.calculate_single_reference(pred, ref)
            rouge_scores.append(score)
        
        # 평균 점수 계산
        avg_rouge1_f1 = np.mean([score.rouge1.f1 for score in rouge_scores])
        avg_rouge2_f1 = np.mean([score.rouge2.f1 for score in rouge_scores])
        avg_rougeL_f1 = np.mean([score.rougeL.f1 for score in rouge_scores])
        avg_combined_f1 = avg_rouge1_f1 + avg_rouge2_f1 + avg_rougeL_f1
        
        return {
            'rouge1_f1': avg_rouge1_f1,
            'rouge2_f1': avg_rouge2_f1,
            'rougeL_f1': avg_rougeL_f1,
            'rouge_combined_f1': avg_combined_f1,  # WandB Sweep 목표
            'rouge1_precision': np.mean([score.rouge1.precision for score in rouge_scores]),
            'rouge1_recall': np.mean([score.rouge1.recall for score in rouge_scores]),
            'rouge2_precision': np.mean([score.rouge2.precision for score in rouge_scores]),
            'rouge2_recall': np.mean([score.rouge2.recall for score in rouge_scores]),
            'rougeL_precision': np.mean([score.rougeL.precision for score in rouge_scores]),
            'rougeL_recall': np.mean([score.rougeL.recall for score in rouge_scores])
        }
    
    def evaluate_with_multiple_references(self, predictions: List[str], 
                                        multiple_references: List[List[str]]) -> Dict[str, float]:
        """
        다중 참조 평가 (대회 전용)
        
        Args:
            predictions: 예측 요약문 리스트
            multiple_references: 각 예측에 대한 다중 참조 리스트
            
        Returns:
            평균 메트릭 딕셔너리
        """
        if len(predictions) != len(multiple_references):
            raise ValueError("Predictions and references length mismatch")
        
        rouge_scores = []
        for pred, refs in zip(predictions, multiple_references):
            # 다중 참조 ROUGE 계산
            score = self.rouge_calculator.calculate_multi_reference(pred, refs)
            rouge_scores.append(score)
        
        # 평균 점수 계산
        avg_rouge1_f1 = np.mean([score.rouge1.f1 for score in rouge_scores])
        avg_rouge2_f1 = np.mean([score.rouge2.f1 for score in rouge_scores])
        avg_rougeL_f1 = np.mean([score.rougeL.f1 for score in rouge_scores])
        avg_combined_f1 = avg_rouge1_f1 + avg_rouge2_f1 + avg_rougeL_f1
        
        result = {
            'rouge1_f1': avg_rouge1_f1,
            'rouge2_f1': avg_rouge2_f1,
            'rougeL_f1': avg_rougeL_f1,
            'rouge_combined_f1': avg_combined_f1,
            'rouge1_precision': np.mean([score.rouge1.precision for score in rouge_scores]),
            'rouge1_recall': np.mean([score.rouge1.recall for score in rouge_scores]),
            'rouge2_precision': np.mean([score.rouge2.precision for score in rouge_scores]),
            'rouge2_recall': np.mean([score.rouge2.recall for score in rouge_scores]),
            'rougeL_precision': np.mean([score.rougeL.precision for score in rouge_scores]),
            'rougeL_recall': np.mean([score.rougeL.recall for score in rouge_scores])
        }
        
        self.logger.info("Multi-reference ROUGE Evaluation Results:")
        self.logger.info(f"  ROUGE-1 F1: {result['rouge1_f1']:.4f}")
        self.logger.info(f"  ROUGE-2 F1: {result['rouge2_f1']:.4f}")
        self.logger.info(f"  ROUGE-L F1: {result['rougeL_f1']:.4f}")
        self.logger.info(f"  Combined F1: {result['rouge_combined_f1']:.4f}")
        
        return result


class MetricTracker:
    """
    메트릭 추적기
    
    학습 과정에서 메트릭을 추적하고 최고 점수를 기록합니다.
    """
    
    def __init__(self):
        """MetricTracker 초기화"""
        self.history = defaultdict(list)
        self.best_scores = {}
        self.best_epoch = {}
        self.logger = logging.getLogger(__name__)
    
    def update(self, metrics: Dict[str, float], epoch: int):
        """
        메트릭 업데이트
        
        Args:
            metrics: 메트릭 딕셔너리
            epoch: 현재 에폭
        """
        for metric_name, value in metrics.items():
            self.history[metric_name].append(value)
            
            # 최고 점수 업데이트
            if metric_name not in self.best_scores or value > self.best_scores[metric_name]:
                self.best_scores[metric_name] = value
                self.best_epoch[metric_name] = epoch
                self.logger.info(f"New best {metric_name}: {value:.4f} at epoch {epoch}")
    
    def get_best_scores(self) -> Dict[str, float]:
        """
        최고 점수 반환
        
        Returns:
            최고 점수 딕셔너리
        """
        return self.best_scores.copy()
    
    def get_history(self, metric_name: str) -> List[float]:
        """
        특정 메트릭 히스토리 반환
        
        Args:
            metric_name: 메트릭명
            
        Returns:
            메트릭 히스토리 리스트
        """
        return self.history[metric_name].copy()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        요약 정보 반환
        
        Returns:
            요약 딕셔너리
        """
        summary = {
            'best_scores': self.best_scores,
            'best_epochs': self.best_epoch,
            'total_epochs': len(self.history.get('rouge_combined_f1', [])),
            'final_scores': {name: history[-1] if history else 0.0 
                           for name, history in self.history.items()}
        }
        return summary


# 편의 함수들
def calculate_rouge_scores(predictions: List[str], 
                         references: List[str],
                         use_korean_tokenizer: bool = True) -> Dict[str, float]:
    """
    ROUGE 점수 계산 편의 함수
    
    Args:
        predictions: 예측 요약문 리스트
        references: 참조 요약문 리스트
        use_korean_tokenizer: 한국어 토크나이저 사용 여부
        
    Returns:
        평균 ROUGE 점수 딕셔너리
    """
    calculator = RougeCalculator(use_korean_tokenizer=use_korean_tokenizer)
    
    scores = []
    for pred, ref in zip(predictions, references):
        score = calculator.calculate_single_reference(pred, ref)
        scores.append(score)
    
    return {
        'rouge1_f1': np.mean([score.rouge1.f1 for score in scores]),
        'rouge2_f1': np.mean([score.rouge2.f1 for score in scores]),
        'rougeL_f1': np.mean([score.rougeL.f1 for score in scores]),
        'rouge_combined_f1': np.mean([score.rouge_combined_f1 for score in scores])
    }


def create_compute_metrics_fn(tokenizer, use_korean_tokenizer: bool = True):
    """
    HuggingFace Trainer용 compute_metrics 함수 생성
    
    Args:
        tokenizer: 토크나이저
        use_korean_tokenizer: 한국어 토크나이저 사용 여부
        
    Returns:
        compute_metrics 함수
    """
    evaluator = MultiReferenceROUGE(tokenizer, use_korean_tokenizer)
    return evaluator.compute_metrics


def compute_metrics_for_trainer(tokenizer, 
                               use_korean_tokenizer: bool = True,
                               multi_reference: bool = False):
    """
    HuggingFace Trainer용 compute_metrics 함수 생성 (개선된 버전)
    
    Args:
        tokenizer: 토크나이저
        use_korean_tokenizer: 한국어 토크나이저 사용 여부
        multi_reference: 다중 참조 평가 사용 여부
        
    Returns:
        compute_metrics 함수
    """
    evaluator = MultiReferenceROUGE(tokenizer, use_korean_tokenizer)
    
    if multi_reference:
        # 다중 참조를 지원하는 compute_metrics 함수
        def compute_metrics_multi(eval_preds):
            predictions, labels = eval_preds
            
            # 토큰 디코딩
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            
            # 예측 텍스트 디코딩
            pred_texts = tokenizer.batch_decode(
                predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            
            # 레이블이 List[List[str]] 형태인 경우 (다중 참조)
            if isinstance(labels, list) and all(isinstance(l, list) for l in labels):
                # 이미 디코딩된 텍스트 리스트
                return evaluator.evaluate_with_multiple_references(pred_texts, labels)
            else:
                # 토큰 형태의 레이블 처리
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                label_texts = tokenizer.batch_decode(
                    labels, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                
                # 단일 참조로 처리
                return evaluator.compute_metrics((predictions, labels))
        
        return compute_metrics_multi
    else:
        # 기존 단일 참조 compute_metrics
        return evaluator.compute_metrics


def evaluate_competition_format(predictions_file: str,
                              ground_truth_file: str,
                              use_korean_tokenizer: bool = True) -> Dict[str, float]:
    """
    대회 형식의 평가 수행
    
    Args:
        predictions_file: 예측 파일 경로 (fname, summary)
        ground_truth_file: 정답 파일 경로 (fname, summary1, summary2, summary3)
        use_korean_tokenizer: 한국어 토크나이저 사용 여부
        
    Returns:
        평가 결과 딕셔너리
    """
    import pandas as pd
    
    # 파일 로드
    pred_df = pd.read_csv(predictions_file)
    truth_df = pd.read_csv(ground_truth_file)
    
    # fname으로 병합
    merged_df = pred_df.merge(truth_df, on='fname')
    
    # RougeCalculator 생성
    calculator = RougeCalculator(use_korean_tokenizer=use_korean_tokenizer)
    
    # 각 샘플에 대해 multi-reference ROUGE 계산
    rouge_scores = []
    for _, row in merged_df.iterrows():
        prediction = row['summary']
        references = [
            row.get('summary1', ''),
            row.get('summary2', ''),
            row.get('summary3', '')
        ]
        # 빈 참조 제거
        references = [ref for ref in references if ref and isinstance(ref, str)]
        
        score = calculator.calculate_multi_reference(prediction, references)
        rouge_scores.append(score)
    
    # 평균 계산
    avg_rouge1_f1 = np.mean([score.rouge1.f1 for score in rouge_scores])
    avg_rouge2_f1 = np.mean([score.rouge2.f1 for score in rouge_scores])
    avg_rougeL_f1 = np.mean([score.rougeL.f1 for score in rouge_scores])
    avg_combined_f1 = avg_rouge1_f1 + avg_rouge2_f1 + avg_rougeL_f1
    
    result = {
        'rouge1_f1': avg_rouge1_f1,
        'rouge2_f1': avg_rouge2_f1,
        'rougeL_f1': avg_rougeL_f1,
        'rouge_combined_f1': avg_combined_f1,
        'num_samples': len(rouge_scores)
    }
    
    print(f"\n=== 대회 형식 평가 결과 ===")
    print(f"샘플 수: {result['num_samples']}")
    print(f"ROUGE-1 F1: {result['rouge1_f1']:.4f}")
    print(f"ROUGE-2 F1: {result['rouge2_f1']:.4f}")
    print(f"ROUGE-L F1: {result['rougeL_f1']:.4f}")
    print(f"Combined F1: {result['rouge_combined_f1']:.4f}")
    
    return result
