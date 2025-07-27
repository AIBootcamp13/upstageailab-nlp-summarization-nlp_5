"""
Inference with Postprocessing Integration

This module integrates postprocessing into the inference pipeline
"""

import logging
from typing import List, Dict, Optional, Tuple
import torch
from pathlib import Path
import time
import json

from postprocessing import RuleBasedPostProcessor

logger = logging.getLogger(__name__)


class PostprocessedInference:
    """Inference pipeline with integrated postprocessing"""
    
    def __init__(self, model, tokenizer, config: Dict):
        """
        Initialize inference with postprocessing
        
        Args:
            model: Trained model
            tokenizer: Tokenizer
            config: Configuration with postprocessing settings
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # 후처리기 초기화 (활성화된 경우)
        postproc_config = config.get('postprocessing', {})
        if postproc_config.get('enabled', False):
            self.postprocessor = RuleBasedPostProcessor(postproc_config)
            logger.info("Postprocessing enabled for inference")
        else:
            self.postprocessor = None
            logger.info("Postprocessing disabled")
        
        # 생성 설정
        self.generation_config = config.get('generation', {})
        self.inference_config = config.get('inference', {})
    
    def generate_summaries(self, dialogues: List[str], batch_size: int = None) -> List[str]:
        """
        Generate summaries for a batch of dialogues
        
        Args:
            dialogues: List of dialogue texts
            batch_size: Batch size for inference
            
        Returns:
            List of generated summaries
        """
        if batch_size is None:
            batch_size = self.inference_config.get('batch_size', 16)
        
        all_summaries = []
        
        # 배치 단위로 처리
        for i in range(0, len(dialogues), batch_size):
            batch_dialogues = dialogues[i:i + batch_size]
            
            # 토큰화
            inputs = self.tokenizer(
                batch_dialogues,
                max_length=self.tokenizer.model_max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            # 디바이스로 이동
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
                self.model.cuda()
            
            # 생성
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.generation_config.get('max_length', 200),
                    num_beams=self.generation_config.get('num_beams', 4),
                    no_repeat_ngram_size=self.generation_config.get('no_repeat_ngram_size', 2),
                    early_stopping=self.generation_config.get('early_stopping', True),
                    length_penalty=self.generation_config.get('length_penalty', 1.0)
                )
            
            # 디코딩
            batch_summaries = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # 지정된 경우 추가 토큰 제거
            remove_tokens = self.inference_config.get('remove_tokens', [])
            if remove_tokens:
                for j, summary in enumerate(batch_summaries):
                    for token in remove_tokens:
                        summary = summary.replace(token, "")
                    batch_summaries[j] = summary.strip()
            
            all_summaries.extend(batch_summaries)
        
        return all_summaries
    
    def apply_postprocessing(self, summaries: List[str], dialogues: List[str]) -> Tuple[List[str], Dict]:
        """
        Apply postprocessing to generated summaries
        
        Args:
            summaries: Generated summaries
            dialogues: Original dialogues
            
        Returns:
            Tuple of (processed summaries, statistics)
        """
        if not self.postprocessor:
            return summaries, {}
        
        logger.info("Applying postprocessing to %d summaries", len(summaries))
        start_time = time.time()
        
        # 요약 처리
        processed_summaries = self.postprocessor.process_batch(summaries, dialogues)
        
        # 통계 계산
        stats = {
            'num_changed': 0,
            'avg_length_change': 0,
            'duplicate_removals': 0,
            'token_restorations': 0,
            'processing_time': time.time() - start_time
        }
        
        total_length_change = 0
        for orig, proc in zip(summaries, processed_summaries):
            if orig != proc:
                stats['num_changed'] += 1
                
                # 변경 사항 확인
                if len(proc.split()) < len(orig.split()):
                    stats['duplicate_removals'] += 1
                
                # 복원된 토큰 확인
                for token in ['#PhoneNumber#', '#Address#', '#Email#']:
                    if token in proc and token not in orig:
                        stats['token_restorations'] += 1
                
                total_length_change += len(proc.split()) - len(orig.split())
        
        if stats['num_changed'] > 0:
            stats['avg_length_change'] = total_length_change / stats['num_changed']
        
        stats['change_percentage'] = (stats['num_changed'] / len(summaries)) * 100
        
        logger.info("Postprocessing complete: %.1f%% of summaries modified", 
                   stats['change_percentage'])
        
        return processed_summaries, stats
    
    def inference_with_postprocessing(self, dialogues: List[str]) -> Tuple[List[str], Dict]:
        """
        Complete inference pipeline with postprocessing
        
        Args:
            dialogues: List of dialogue texts
            
        Returns:
            Tuple of (final summaries, statistics)
        """
        # 초기 요약 생성
        logger.info("Generating summaries for %d dialogues", len(dialogues))
        summaries = self.generate_summaries(dialogues)
        
        # 활성화된 경우 후처리 적용
        if self.postprocessor and self.inference_config.get('apply_postprocessing', True):
            summaries, postproc_stats = self.apply_postprocessing(summaries, dialogues)
        else:
            postproc_stats = {}
        
        # 전반적인 통계
        stats = {
            'num_dialogues': len(dialogues),
            'num_summaries': len(summaries),
            'postprocessing': postproc_stats
        }
        
        return summaries, stats


def create_postprocessing_report(
    original_summaries: List[str],
    processed_summaries: List[str],
    dialogues: List[str],
    output_path: str
):
    """
    Create a detailed report of postprocessing changes
    
    Args:
        original_summaries: Summaries before postprocessing
        processed_summaries: Summaries after postprocessing
        dialogues: Original dialogues
        output_path: Path to save the report
    """
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_samples': len(original_summaries),
        'total_changed': 0,
        'changes': []
    }
    
    for i, (orig, proc, dial) in enumerate(zip(original_summaries, processed_summaries, dialogues)):
        if orig != proc:
            report['total_changed'] += 1
            
            change = {
                'index': i,
                'dialogue_preview': dial[:200] + '...' if len(dial) > 200 else dial,
                'original_summary': orig,
                'processed_summary': proc,
                'changes_made': []
            }
            
            # 변경 사항 식별
            if len(proc.split()) < len(orig.split()):
                change['changes_made'].append('duplicate_removal')
            
            if len(proc.split()) > len(orig.split()):
                change['changes_made'].append('length_expansion')
            elif len(proc.split()) < len(orig.split()) - 5:
                change['changes_made'].append('length_truncation')
            
            # 토큰 복원 확인
            for token in ['#PhoneNumber#', '#Address#', '#Email#']:
                if token in proc and token not in orig:
                    change['changes_made'].append(f'token_restored_{token}')
            
            report['changes'].append(change)
    
    report['change_percentage'] = (report['total_changed'] / report['total_samples']) * 100
    
    # 보고서 저장
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    logger.info("Postprocessing report saved to %s", output_path)
    
    return report


# 사용 예제
if __name__ == "__main__":
    # 추론에서 후처리를 사용하는 방법의 시연
    print("Postprocessing Integration Example")
    print("="*50)
    
    # 예시 설정
    config = {
        'postprocessing': {
            'enabled': True,
            'duplicate_removal': {'enabled': True},
            'length_optimization': {
                'enabled': True,
                'min_tokens': 50,
                'max_tokens': 100
            },
            'token_validation': {
                'enabled': True,
                'copy_missing_tokens': True
            }
        },
        'generation': {
            'max_length': 200,
            'num_beams': 4,
            'no_repeat_ngram_size': 2
        },
        'inference': {
            'batch_size': 16,
            'apply_postprocessing': True,
            'remove_tokens': ['<s>', '</s>', '<pad>']
        }
    }
    
    print("Configuration loaded")
    print("Postprocessing pipeline includes:")
    print("- Duplicate removal")
    print("- Length optimization (50-100 tokens)")
    print("- Special token validation")
    print("\nReady for inference with postprocessing!")
