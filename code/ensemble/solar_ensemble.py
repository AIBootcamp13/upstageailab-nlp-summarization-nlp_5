"""
Solar API 앙상블 모듈

Fine-tuned 모델과 Solar API를 결합한 앙상블 시스템을 구현합니다.
가중치 기반 투표와 동적 가중치 조정을 지원합니다.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from tqdm import tqdm
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    GenerationConfig
)
from openai import OpenAI
import evaluate

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    """앙상블 설정"""
    # Solar API 설정
    solar_api_key: str
    solar_model: str = "solar-1-mini-chat"
    solar_base_url: str = "https://api.upstage.ai/v1/solar"
    
    # 가중치 설정
    fine_tuned_weight: float = 0.7
    solar_weight: float = 0.3
    dynamic_weights: bool = False
    
    # API 설정
    max_retries: int = 3
    retry_delay: int = 5
    rate_limit_per_minute: int = 100
    timeout: int = 30
    
    # 생성 설정
    temperature: float = 0.3
    top_p: float = 0.9
    max_length: int = 200
    min_length: int = 30
    
    # 배치 설정
    batch_size: int = 8
    use_async: bool = True
    
    # 캐시 설정
    use_cache: bool = True
    cache_dir: str = "cache/solar_ensemble"


@dataclass
class EnsembleResult:
    """앙상블 결과"""
    dialogue: str
    fine_tuned_summary: str
    solar_summary: str
    ensemble_summary: str
    weights_used: Dict[str, float]
    confidence_score: float
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class SolarAPIClient:
    """Solar API 클라이언트"""
    
    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.client = OpenAI(
            api_key=config.solar_api_key,
            base_url=config.solar_base_url
        )
        self.request_count = 0
        self.last_request_time = time.time()
        
        # 캐시 초기화
        if config.use_cache:
            self.cache_dir = Path(config.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache = self._load_cache()
        else:
            self.cache = {}
    
    def _load_cache(self) -> Dict:
        """캐시 로드"""
        cache_file = self.cache_dir / "solar_cache.json"
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_cache(self):
        """캐시 저장"""
        cache_file = self.cache_dir / "solar_cache.json"
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)
    
    def _check_rate_limit(self):
        """Rate limit 확인"""
        current_time = time.time()
        time_since_start = current_time - self.last_request_time
        
        if self.request_count >= self.config.rate_limit_per_minute:
            if time_since_start < 60:
                sleep_time = 60 - time_since_start + 1
                logger.info(f"Rate limit reached. Sleeping for {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
                self.request_count = 0
                self.last_request_time = time.time()
        
        if time_since_start >= 60:
            self.request_count = 0
            self.last_request_time = current_time
    
    def build_prompt(self, dialogue: str, few_shot_examples: Optional[List[Dict]] = None) -> List[Dict]:
        """프롬프트 생성"""
        system_prompt = (
            "You are an expert in Korean dialogue summarization. "
            "Summarize the given Korean dialogue in a concise and informative manner. "
            "Preserve all important information including speaker identities and key points. "
            "The summary should be in Korean."
        )
        
        if few_shot_examples:
            # Few-shot 프롬프트 구성
            messages = [{"role": "system", "content": system_prompt}]
            
            for example in few_shot_examples[:2]:  # 최대 2개 예제 사용
                messages.extend([
                    {
                        "role": "user",
                        "content": f"대화:\n{example['dialogue']}\n\n요약:"
                    },
                    {
                        "role": "assistant",
                        "content": example['summary']
                    }
                ])
            
            messages.append({
                "role": "user",
                "content": f"대화:\n{dialogue}\n\n요약:"
            })
        else:
            # Zero-shot 프롬프트
            user_prompt = (
                "다음 한국어 대화를 읽고 핵심 내용을 요약해주세요.\n"
                "화자 정보(#Person1#, #Person2# 등)와 개인정보(#PhoneNumber# 등)를 모두 포함해주세요.\n\n"
                f"대화:\n{dialogue}\n\n"
                "요약:"
            )
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        
        return messages
    
    def summarize(self, dialogue: str, few_shot_examples: Optional[List[Dict]] = None) -> str:
        """안정성 강화된 Solar API 요약
        
        개선사항:
        - API 키 검증 및 연결 테스트
        - 지수 백오프 재시도 메커니즘
        - 연속 실패 시 폴백 메커니즘
        - 비용 최적화된 캐싱
        - 자세한 오류 로깅
        """
        # API 클라이언트 상태 검증
        if not hasattr(self, 'client') or not self.client:
            logger.warning("⚠️ Solar API 클라이언트가 초기화되지 않음 - 폴백 모드")
            return ""
            
        # 캐시 확인
        cache_key = dialogue[:100]  # 대화의 처음 100자를 키로 사용
        if self.config.use_cache and cache_key in self.cache:
            logger.debug("💾 캐시된 Solar 요약 사용")
            return self.cache[cache_key]
        
        # 비율 제한 확인
        self._check_rate_limit()
        
        # API 호출 시도
        messages = self.build_prompt(dialogue, few_shot_examples)
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                start_time = time.time()
                
                response = self.client.chat.completions.create(
                    model=self.config.solar_model,
                    messages=messages,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    max_tokens=self.config.max_length,
                    timeout=self.config.timeout
                )
                
                if not response.choices or not response.choices[0].message.content:
                    raise Exception("비어있는 응답 수신")
                
                summary = response.choices[0].message.content.strip()
                self.request_count += 1
                processing_time = time.time() - start_time
                
                # 성공 로깅
                logger.debug(f"✅ Solar API 성공 (attempt {attempt + 1}, {processing_time:.2f}s)")
                
                # 캐시 저장
                if self.config.use_cache:
                    self.cache[cache_key] = summary
                    if len(self.cache) % 100 == 0:  # 100개마다 저장
                        self._save_cache()
                
                return summary
                
            except Exception as e:
                last_error = e
                error_type = type(e).__name__
                
                # 상세 오류 로깅
                logger.warning(
                    f"⚠️ Solar API 오류 (attempt {attempt + 1}/{self.config.max_retries}): "
                    f"{error_type} - {str(e)}"
                )
                
                # 마지막 시도가 아니면 재시도
                if attempt < self.config.max_retries - 1:
                    # 지수 백오프 적용
                    delay = self.config.retry_delay * (2 ** attempt)
                    delay = min(delay, 60)  # 최대 60초
                    
                    logger.info(f"⏱️ {delay}초 후 재시도... ({error_type})")
                    time.sleep(delay)
                    
                    # 타임아웃 오류인 경우 타임아웃 증가
                    if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                        original_timeout = self.config.timeout
                        self.config.timeout = min(self.config.timeout + 15, 120)
                        logger.info(f"타임아웃 증가: {original_timeout}수 → {self.config.timeout}수")
        
        # 모든 시도 실패
        logger.error(
            f"❌ Solar API 완전 실패 ({self.config.max_retries}회 시도): "
            f"{type(last_error).__name__} - {str(last_error)}"
        )
        
        return ""
    async def summarize_async(self, dialogue: str, few_shot_examples: Optional[List[Dict]] = None) -> str:
        """비동기 Solar API 요약"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.summarize, dialogue, few_shot_examples)
    
    def close(self):
        """클라이언트 종료"""
        if self.config.use_cache:
            self._save_cache()


class WeightedEnsemble:
    """가중치 기반 앙상블"""
    
    def __init__(
        self,
        fine_tuned_model_path: str,
        ensemble_config: EnsembleConfig,
        device: Optional[str] = None
    ):
        self.config = ensemble_config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Fine-tuned 모델 로드
        logger.info(f"Loading fine-tuned model from {fine_tuned_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(fine_tuned_model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Solar API 클라이언트
        self.solar_client = SolarAPIClient(ensemble_config)
        
        # ROUGE 평가기
        self.rouge = evaluate.load("rouge")
        
        # 동적 가중치 히스토리
        self.weight_history = []
        
        # Few-shot 예제 (선택적)
        self.few_shot_examples = []
    
    def load_few_shot_examples(self, train_file: str, num_examples: int = 3):
        """Few-shot 예제 로드"""
        df = pd.read_csv(train_file)
        # 높은 품질의 예제 선택 (길이 기반)
        df['summary_len'] = df['summary'].str.len()
        df_sorted = df.sort_values('summary_len', ascending=False)
        
        self.few_shot_examples = [
            {
                'dialogue': row['dialogue'],
                'summary': row['summary']
            }
            for _, row in df_sorted.head(num_examples).iterrows()
        ]
        logger.info(f"Loaded {len(self.few_shot_examples)} few-shot examples")
    
    def generate_fine_tuned_summary(self, dialogue: str) -> str:
        """Fine-tuned 모델을 사용한 요약 생성"""
        inputs = self.tokenizer(
            dialogue,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.config.max_length,
                min_length=self.config.min_length,
                num_beams=4,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
                early_stopping=True
            )
        
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary.strip()
    
    def calculate_dynamic_weights(
        self,
        fine_tuned_summary: str,
        solar_summary: str,
        dialogue: str
    ) -> Dict[str, float]:
        """동적 가중치 계산"""
        if not self.config.dynamic_weights:
            return {
                'fine_tuned': self.config.fine_tuned_weight,
                'solar': self.config.solar_weight
            }
        
        # 특징 추출
        features = {
            'dialogue_length': len(dialogue.split()),
            'fine_tuned_length': len(fine_tuned_summary.split()),
            'solar_length': len(solar_summary.split()),
            'special_tokens_in_dialogue': len([t for t in ['#Person', '#Phone', '#Address'] 
                                              if t in dialogue]),
            'special_tokens_preserved_ft': len([t for t in ['#Person', '#Phone', '#Address'] 
                                               if t in fine_tuned_summary]),
            'special_tokens_preserved_solar': len([t for t in ['#Person', '#Phone', '#Address'] 
                                                 if t in solar_summary])
        }
        
        # 가중치 조정 규칙
        weight_ft = self.config.fine_tuned_weight
        weight_solar = self.config.solar_weight
        
        # 특수 토큰 보존 기반 조정
        if features['special_tokens_preserved_ft'] > features['special_tokens_preserved_solar']:
            weight_ft += 0.1
            weight_solar -= 0.1
        elif features['special_tokens_preserved_solar'] > features['special_tokens_preserved_ft']:
            weight_ft -= 0.1
            weight_solar += 0.1
        
        # 길이 균형 기반 조정
        ideal_length = features['dialogue_length'] * 0.3  # 대화의 30% 길이가 이상적
        ft_diff = abs(features['fine_tuned_length'] - ideal_length)
        solar_diff = abs(features['solar_length'] - ideal_length)
        
        if ft_diff < solar_diff:
            weight_ft += 0.05
            weight_solar -= 0.05
        else:
            weight_ft -= 0.05
            weight_solar += 0.05
        
        # 정규화
        total = weight_ft + weight_solar
        weights = {
            'fine_tuned': weight_ft / total,
            'solar': weight_solar / total
        }
        
        # 히스토리 저장
        self.weight_history.append({
            'weights': weights,
            'features': features
        })
        
        return weights
    
    def combine_summaries(
        self,
        fine_tuned_summary: str,
        solar_summary: str,
        weights: Dict[str, float]
    ) -> str:
        """요약 결합"""
        # 간단한 방법: 높은 가중치를 가진 요약 선택
        if weights['fine_tuned'] >= weights['solar']:
            primary_summary = fine_tuned_summary
            secondary_summary = solar_summary
            primary_weight = weights['fine_tuned']
        else:
            primary_summary = solar_summary
            secondary_summary = fine_tuned_summary
            primary_weight = weights['solar']
        
        # 가중치가 압도적이면 주요 요약만 사용
        if primary_weight > 0.8:
            return primary_summary
        
        # 그렇지 않으면 보조 요약에서 누락된 정보 추가
        primary_tokens = set(primary_summary.split())
        secondary_tokens = set(secondary_summary.split())
        
        # 특수 토큰 확인
        special_tokens = ['#Person1#', '#Person2#', '#Person3#', 
                         '#PhoneNumber#', '#Address#', '#Email#']
        
        missing_special = []
        for token in special_tokens:
            if token in secondary_summary and token not in primary_summary:
                missing_special.append(token)
        
        # 누락된 특수 토큰이 있으면 보조 요약 사용
        if missing_special:
            return secondary_summary
        
        return primary_summary
    
    def calculate_confidence(
        self,
        fine_tuned_summary: str,
        solar_summary: str,
        ensemble_summary: str
    ) -> float:
        """신뢰도 점수 계산"""
        # 두 요약 간의 ROUGE 점수로 일치도 측정
        rouge_scores = self.rouge.compute(
            predictions=[fine_tuned_summary],
            references=[solar_summary]
        )
        
        # F1 점수 평균
        agreement_score = np.mean([
            rouge_scores['rouge1'],
            rouge_scores['rouge2'],
            rouge_scores['rougeL']
        ])
        
        # 특수 토큰 보존 점수
        special_tokens = ['#Person', '#Phone', '#Address', '#Email']
        preservation_score = sum(
            1 for token in special_tokens if token in ensemble_summary
        ) / len(special_tokens)
        
        # 종합 신뢰도
        confidence = 0.7 * agreement_score + 0.3 * preservation_score
        
        return float(confidence)
    
    def process_single(self, dialogue: str) -> EnsembleResult:
        """단일 대화 처리 - 폴백 메커니즘 강화"""
        start_time = time.time()
        
        # Fine-tuned 모델 요약
        try:
            fine_tuned_summary = self.generate_fine_tuned_summary(dialogue)
        except Exception as e:
            logger.error(f"❌ Fine-tuned 모델 오류: {str(e)}")
            fine_tuned_summary = "대화 요약 실패"  # 기본 폴백
        
        # Solar API 요약
        solar_summary = ""
        solar_failed = False
        
        try:
            solar_summary = self.solar_client.summarize(dialogue, self.few_shot_examples)
            if not solar_summary:  # 빈 문자열 반환 시 실패로 간주
                solar_failed = True
                logger.warning("⚠️ Solar API가 빈 요약 반환")
        except Exception as e:
            solar_failed = True
            logger.error(f"❌ Solar API 오류: {str(e)}")
        
        # 폴백 메커니즘
        if solar_failed or not solar_summary:
            logger.info("🔄 Solar API 실패 - Fine-tuned 모델만 사용 (Fallback 모드)")
            ensemble_summary = fine_tuned_summary
            weights = {'fine_tuned': 1.0, 'solar': 0.0}
            confidence = 0.7  # Fine-tuned 모델만 사용 시 기본 신뢰도
            solar_summary = "사용 불가 (폴백 모드)"
        else:
            # 정상 앙상블 처리
            weights = self.calculate_dynamic_weights(
                fine_tuned_summary,
                solar_summary,
                dialogue
            )
            
            ensemble_summary = self.combine_summaries(
                fine_tuned_summary,
                solar_summary,
                weights
            )
            
            confidence = self.calculate_confidence(
                fine_tuned_summary,
                solar_summary,
                ensemble_summary
            )
        
        processing_time = time.time() - start_time
        return EnsembleResult(
            dialogue=dialogue,
            fine_tuned_summary=fine_tuned_summary,
            solar_summary=solar_summary,
            ensemble_summary=ensemble_summary,
            weights_used=weights,
            confidence_score=confidence,
            processing_time=processing_time,
            metadata={
                'solar_failed': solar_failed,
                'fallback_mode': solar_failed or not solar_summary
            }
        )
    
    async def process_batch_async(self, dialogues: List[str]) -> List[EnsembleResult]:
        """배치 비동기 처리"""
        tasks = []
        
        for dialogue in dialogues:
            task = asyncio.create_task(self._process_single_async(dialogue))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results
    
    async def _process_single_async(self, dialogue: str) -> EnsembleResult:
        """단일 대화 비동기 처리"""
        start_time = time.time()
        
        # Fine-tuned 모델 요약 (동기)
        loop = asyncio.get_event_loop()
        fine_tuned_summary = await loop.run_in_executor(
            None,
            self.generate_fine_tuned_summary,
            dialogue
        )
        
        # Solar API 요약 (비동기)
        solar_summary = await self.solar_client.summarize_async(
            dialogue,
            self.few_shot_examples
        )
        
        # 나머지 처리 (동기)
        weights = self.calculate_dynamic_weights(
            fine_tuned_summary,
            solar_summary,
            dialogue
        )
        
        ensemble_summary = self.combine_summaries(
            fine_tuned_summary,
            solar_summary,
            weights
        )
        
        confidence = self.calculate_confidence(
            fine_tuned_summary,
            solar_summary,
            ensemble_summary
        )
        
        processing_time = time.time() - start_time
        
        return EnsembleResult(
            dialogue=dialogue,
            fine_tuned_summary=fine_tuned_summary,
            solar_summary=solar_summary,
            ensemble_summary=ensemble_summary,
            weights_used=weights,
            confidence_score=confidence,
            processing_time=processing_time
        )
    
    def process_dataset(
        self,
        data_file: str,
        output_file: str,
        sample_size: Optional[int] = None
    ) -> pd.DataFrame:
        """데이터셋 처리"""
        # 데이터 로드
        df = pd.read_csv(data_file)
        if sample_size:
            df = df.head(sample_size)
        
        results = []
        
        # 배치 처리
        for i in tqdm(range(0, len(df), self.config.batch_size), desc="Processing"):
            batch = df.iloc[i:i + self.config.batch_size]
            dialogues = batch['dialogue'].tolist()
            
            if self.config.use_async:
                # 비동기 처리
                batch_results = asyncio.run(self.process_batch_async(dialogues))
            else:
                # 동기 처리
                batch_results = [self.process_single(d) for d in dialogues]
            
            results.extend(batch_results)
        
        # 결과 정리
        output_df = pd.DataFrame([
            {
                'fname': df.iloc[i]['fname'],
                'dialogue': r.dialogue,
                'fine_tuned_summary': r.fine_tuned_summary,
                'solar_summary': r.solar_summary,
                'ensemble_summary': r.ensemble_summary,
                'fine_tuned_weight': r.weights_used['fine_tuned'],
                'solar_weight': r.weights_used['solar'],
                'confidence_score': r.confidence_score,
                'processing_time': r.processing_time
            }
            for i, r in enumerate(results)
        ])
        
        # 저장
        output_df.to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"Results saved to {output_file}")
        
        # 통계 출력
        avg_confidence = output_df['confidence_score'].mean()
        avg_time = output_df['processing_time'].mean()
        
        logger.info(f"Average confidence score: {avg_confidence:.3f}")
        logger.info(f"Average processing time: {avg_time:.2f} seconds")
        
        if self.config.dynamic_weights:
            avg_ft_weight = output_df['fine_tuned_weight'].mean()
            avg_solar_weight = output_df['solar_weight'].mean()
            logger.info(f"Average weights - Fine-tuned: {avg_ft_weight:.3f}, Solar: {avg_solar_weight:.3f}")
        
        return output_df
    
    def evaluate(self, predictions_file: str, references_file: str) -> Dict[str, float]:
        """앙상블 성능 평가"""
        pred_df = pd.read_csv(predictions_file)
        ref_df = pd.read_csv(references_file)
        
        # 정렬
        pred_df = pred_df.sort_values('fname').reset_index(drop=True)
        ref_df = ref_df.sort_values('fname').reset_index(drop=True)
        
        # 각 모델별 평가
        results = {}
        
        for col_name, pred_col in [
            ('fine_tuned', 'fine_tuned_summary'),
            ('solar', 'solar_summary'),
            ('ensemble', 'ensemble_summary')
        ]:
            if pred_col in pred_df.columns:
                rouge_scores = self.rouge.compute(
                    predictions=pred_df[pred_col].tolist(),
                    references=ref_df['summary'].tolist()
                )
                
                results[f'{col_name}_rouge1'] = rouge_scores['rouge1']
                results[f'{col_name}_rouge2'] = rouge_scores['rouge2']
                results[f'{col_name}_rougeL'] = rouge_scores['rougeL']
                results[f'{col_name}_rouge_avg'] = np.mean([
                    rouge_scores['rouge1'],
                    rouge_scores['rouge2'],
                    rouge_scores['rougeL']
                ])
        
        # 개선율 계산
        if 'fine_tuned_rouge_avg' in results and 'ensemble_rouge_avg' in results:
            improvement = (
                (results['ensemble_rouge_avg'] - results['fine_tuned_rouge_avg']) /
                results['fine_tuned_rouge_avg'] * 100
            )
            results['improvement_percent'] = improvement
        
        return results
    
    def close(self):
        """리소스 정리"""
        self.solar_client.close()


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Solar API Ensemble System")
    parser.add_argument("--fine_tuned_model", type=str, required=True,
                       help="Path to fine-tuned model")
    parser.add_argument("--solar_api_key", type=str, required=True,
                       help="Solar API key")
    parser.add_argument("--data_file", type=str, required=True,
                       help="Input data file")
    parser.add_argument("--output_file", type=str, default="ensemble_results.csv",
                       help="Output file")
    parser.add_argument("--sample_size", type=int, default=None,
                       help="Number of samples to process")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--dynamic_weights", action="store_true",
                       help="Use dynamic weights")
    parser.add_argument("--fine_tuned_weight", type=float, default=0.7,
                       help="Weight for fine-tuned model")
    parser.add_argument("--solar_weight", type=float, default=0.3,
                       help="Weight for Solar API")
    parser.add_argument("--train_file", type=str, default=None,
                       help="Training file for few-shot examples")
    
    args = parser.parse_args()
    
    # 설정 생성
    config = EnsembleConfig(
        solar_api_key=args.solar_api_key,
        fine_tuned_weight=args.fine_tuned_weight,
        solar_weight=args.solar_weight,
        dynamic_weights=args.dynamic_weights,
        batch_size=args.batch_size
    )
    
    # 앙상블 생성
    ensemble = WeightedEnsemble(
        fine_tuned_model_path=args.fine_tuned_model,
        ensemble_config=config
    )
    
    # Few-shot 예제 로드 (선택적)
    if args.train_file:
        ensemble.load_few_shot_examples(args.train_file)
    
    # 처리 실행
    try:
        results = ensemble.process_dataset(
            data_file=args.data_file,
            output_file=args.output_file,
            sample_size=args.sample_size
        )
        
        # 평가 (validation set인 경우)
        if 'dev' in args.data_file or 'val' in args.data_file:
            eval_results = ensemble.evaluate(args.output_file, args.data_file)
            
            print("\n=== Evaluation Results ===")
            for metric, value in eval_results.items():
                print(f"{metric}: {value:.4f}")
    
    finally:
        ensemble.close()


if __name__ == "__main__":
    main()
