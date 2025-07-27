#!/usr/bin/env python3
"""
성능 벤치마크 및 통합 테스트 스크립트

조장님의 최신 기술 스택 통합 후 성능 개선 효과를 정량적으로 측정합니다.
- 메모리 사용량 비교 (기존 vs QLoRA)
- 학습 속도 벤치마크
- 모델 로딩 시간 측정
- GPU/MPS 활용률 분석
"""

import sys
import os
import time
import torch
import psutil
import yaml
import tracemalloc
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging
from datetime import datetime

# 프로젝트 경로 추가
sys.path.append(str(Path(__file__).parent.parent / 'code'))

from trainer import DialogueSummarizationTrainer
from core.inference import InferenceEngine
from utils.config_manager import ConfigManager
from utils.memory_monitor import MemoryMonitor


class PerformanceBenchmark:
    """성능 벤치마크 실행기"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.results = {}
        self.logger = self._setup_logger()
        
        # 메모리 모니터 초기화
        self.memory_monitor = MemoryMonitor()
        
        # 설정 로드
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """현재 메모리 사용량 반환"""
        memory = psutil.virtual_memory()
        result = {
            'cpu': memory.used,
            'cpu_percent': memory.percent
        }
        
        if torch.cuda.is_available():
            result['gpu'] = torch.cuda.memory_allocated(0)
            result['gpu_reserved'] = torch.cuda.memory_reserved(0)
        elif torch.backends.mps.is_available():
            # MPS는 시스템 메모리 공유
            result['gpu'] = 0
            result['gpu_reserved'] = 0
        
        return result
    
    def _setup_logger(self):
        """로거 설정"""
        logger = logging.getLogger('PerformanceBenchmark')
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def benchmark_environment(self) -> Dict[str, Any]:
        """환경 정보 벤치마크"""
        self.logger.info("=== 환경 정보 벤치마크 ===")
        
        env_info = {
            'timestamp': datetime.now().isoformat(),
            'python_version': sys.version,
            'torch_version': torch.__version__,
            'device_info': {},
            'memory_info': {},
            'cpu_info': {}
        }
        
        # 디바이스 정보
        if torch.cuda.is_available():
            env_info['device_info'] = {
                'type': 'CUDA',
                'device_count': torch.cuda.device_count(),
                'device_name': torch.cuda.get_device_name(0),
                'memory_total': torch.cuda.get_device_properties(0).total_memory / 1024**3
            }
        elif torch.backends.mps.is_available():
            env_info['device_info'] = {
                'type': 'MPS',
                'device_count': 1,
                'device_name': 'Apple Silicon',
                'memory_shared': True
            }
        else:
            env_info['device_info'] = {
                'type': 'CPU',
                'device_count': 1
            }
        
        # 시스템 메모리 정보
        memory = psutil.virtual_memory()
        env_info['memory_info'] = {
            'total_gb': memory.total / 1024**3,
            'available_gb': memory.available / 1024**3,
            'percent_used': memory.percent
        }
        
        # CPU 정보
        env_info['cpu_info'] = {
            'physical_cores': psutil.cpu_count(logical=False),
            'logical_cores': psutil.cpu_count(logical=True),
            'frequency_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else 'Unknown'
        }
        
        self.logger.info(f"디바이스: {env_info['device_info']['type']}")
        self.logger.info(f"메모리: {env_info['memory_info']['total_gb']:.1f}GB 총용량")
        
        return env_info
    
    def benchmark_model_loading(self) -> Dict[str, Any]:
        """모델 로딩 성능 벤치마크"""
        self.logger.info("=== 모델 로딩 벤치마크 ===")
        
        results = {
            'standard_loading': {},
            'qlora_loading': {},
            'memory_comparison': {}
        }
        
        # 1. 표준 모델 로딩 (QLoRA 비활성화)
        self.logger.info("1. 표준 모델 로딩 테스트...")
        
        # 임시로 QLoRA 비활성화
        original_qlora = self.config['qlora']['use_qlora']
        self.config['qlora']['use_qlora'] = False
        
        tracemalloc.start()
        start_time = time.time()
        memory_before = self._get_memory_usage()
        
        try:
            # trainer 초기화 (실제 모델 로딩)
            trainer = DialogueSummarizationTrainer(self.config)
            
            load_time = time.time() - start_time
            memory_after = self._get_memory_usage()
            current, peak = tracemalloc.get_traced_memory()
            
            results['standard_loading'] = {
                'load_time_seconds': load_time,
                'memory_before_mb': memory_before['cpu'] / 1024**2,
                'memory_after_mb': memory_after['cpu'] / 1024**2,
                'memory_increase_mb': (memory_after['cpu'] - memory_before['cpu']) / 1024**2,
                'peak_memory_mb': peak / 1024**2,
                'success': True
            }
            
            self.logger.info(f"표준 로딩: {load_time:.2f}초, 메모리 증가: {results['standard_loading']['memory_increase_mb']:.1f}MB")
            
            # 메모리 정리
            del trainer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            self.logger.error(f"표준 모델 로딩 실패: {e}")
            results['standard_loading'] = {'success': False, 'error': str(e)}
        
        tracemalloc.stop()
        
        # 2. QLoRA 모델 로딩
        self.logger.info("2. QLoRA 모델 로딩 테스트...")
        
        # QLoRA 재활성화
        self.config['qlora']['use_qlora'] = original_qlora
        
        tracemalloc.start()
        start_time = time.time()
        memory_before = self._get_memory_usage()
        
        try:
            trainer = DialogueSummarizationTrainer(self.config)
            
            load_time = time.time() - start_time
            memory_after = self._get_memory_usage()
            current, peak = tracemalloc.get_traced_memory()
            
            results['qlora_loading'] = {
                'load_time_seconds': load_time,
                'memory_before_mb': memory_before['cpu'] / 1024**2,
                'memory_after_mb': memory_after['cpu'] / 1024**2,
                'memory_increase_mb': (memory_after['cpu'] - memory_before['cpu']) / 1024**2,
                'peak_memory_mb': peak / 1024**2,
                'success': True
            }
            
            self.logger.info(f"QLoRA 로딩: {load_time:.2f}초, 메모리 증가: {results['qlora_loading']['memory_increase_mb']:.1f}MB")
            
            # 메모리 정리
            del trainer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            self.logger.error(f"QLoRA 모델 로딩 실패: {e}")
            results['qlora_loading'] = {'success': False, 'error': str(e)}
        
        tracemalloc.stop()
        
        # 3. 메모리 효율성 비교
        if results['standard_loading'].get('success') and results['qlora_loading'].get('success'):
            standard_mem = results['standard_loading']['memory_increase_mb']
            qlora_mem = results['qlora_loading']['memory_increase_mb']
            
            memory_saving = ((standard_mem - qlora_mem) / standard_mem) * 100 if standard_mem > 0 else 0
            
            results['memory_comparison'] = {
                'standard_memory_mb': standard_mem,
                'qlora_memory_mb': qlora_mem,
                'memory_saving_percent': memory_saving,
                'memory_saving_mb': standard_mem - qlora_mem
            }
            
            self.logger.info(f"메모리 절약: {memory_saving:.1f}% ({standard_mem - qlora_mem:.1f}MB 절약)")
        
        return results
    
    def benchmark_inference_speed(self) -> Dict[str, Any]:
        """추론 속도 벤치마크"""
        self.logger.info("=== 추론 속도 벤치마크 ===")
        
        # 테스트 데이터 생성
        test_dialogues = [
            "안녕하세요. 오늘 날씨가 참 좋네요. 산책하기 딱 좋은 날씨입니다.",
            "회의 시간을 조정해야 할 것 같습니다. 다음 주 화요일 오후 2시는 어떠세요?",
            "프로젝트 진행 상황을 공유드리겠습니다. 현재 80% 정도 완료되었습니다.",
            "점심 메뉴 추천 부탁드립니다. 한식, 중식, 일식 중에서 골라주세요.",
            "내일 출장 일정 확인 부탁드립니다. 기차표 예약도 필요합니다."
        ]
        
        results = {
            'single_inference': {},
            'batch_inference': {},
            'throughput_analysis': {}
        }
        
        try:
            # InferenceEngine 초기화
            inference_engine = InferenceEngine(self.config)
            
            # 1. 단일 추론 성능
            self.logger.info("1. 단일 추론 성능 측정...")
            
            single_times = []
            for dialogue in test_dialogues:
                start_time = time.time()
                summary = inference_engine.predict(dialogue)
                inference_time = time.time() - start_time
                single_times.append(inference_time)
            
            results['single_inference'] = {
                'avg_time_seconds': sum(single_times) / len(single_times),
                'min_time_seconds': min(single_times),
                'max_time_seconds': max(single_times),
                'total_samples': len(single_times)
            }
            
            # 2. 배치 추론 성능
            self.logger.info("2. 배치 추론 성능 측정...")
            
            batch_sizes = [1, 2, 4]
            batch_results = {}
            
            for batch_size in batch_sizes:
                if len(test_dialogues) >= batch_size:
                    batch_data = test_dialogues[:batch_size]
                    
                    start_time = time.time()
                    summaries = inference_engine.predict_batch(batch_data, batch_size=batch_size)
                    batch_time = time.time() - start_time
                    
                    batch_results[f'batch_{batch_size}'] = {
                        'total_time_seconds': batch_time,
                        'time_per_sample': batch_time / batch_size,
                        'throughput_samples_per_sec': batch_size / batch_time
                    }
            
            results['batch_inference'] = batch_results
            
            # 3. 처리량 분석
            single_throughput = 1 / results['single_inference']['avg_time_seconds']
            best_batch_throughput = max([
                batch_results[key]['throughput_samples_per_sec'] 
                for key in batch_results.keys()
            ])
            
            results['throughput_analysis'] = {
                'single_throughput_samples_per_sec': single_throughput,
                'best_batch_throughput_samples_per_sec': best_batch_throughput,
                'throughput_improvement_percent': ((best_batch_throughput - single_throughput) / single_throughput) * 100
            }
            
            self.logger.info(f"단일 처리량: {single_throughput:.2f} samples/sec")
            self.logger.info(f"최적 배치 처리량: {best_batch_throughput:.2f} samples/sec")
            
        except Exception as e:
            self.logger.error(f"추론 속도 벤치마크 실패: {e}")
            results = {'success': False, 'error': str(e)}
        
        return results
    
    def benchmark_configuration_impact(self) -> Dict[str, Any]:
        """설정 최적화 효과 벤치마크"""
        self.logger.info("=== 설정 최적화 효과 분석 ===")
        
        results = {
            'configuration_analysis': {},
            'optimization_score': 0,
            'recommendations': []
        }
        
        # 주요 최적화 설정 확인
        tokenizer_config = self.config.get('tokenizer', {})
        training_config = self.config.get('training', {})
        qlora_config = self.config.get('qlora', {})
        
        optimizations = {
            'decoder_max_len_200': tokenizer_config.get('decoder_max_len') == 200,
            'eval_strategy_steps': training_config.get('eval_strategy') == 'steps',
            'gradient_checkpointing': training_config.get('gradient_checkpointing') == True,
            'torch_empty_cache_steps': training_config.get('torch_empty_cache_steps') == 10,
            'qlora_enabled': qlora_config.get('use_qlora') == True,
            'lora_rank_optimized': qlora_config.get('lora_rank') == 16
        }
        
        # 최적화 점수 계산
        score = sum(optimizations.values()) / len(optimizations) * 100
        results['optimization_score'] = score
        
        # 설정 분석
        results['configuration_analysis'] = {
            'applied_optimizations': optimizations,
            'total_optimizations': len(optimizations),
            'applied_count': sum(optimizations.values()),
            'score_percentage': score
        }
        
        # 권장사항 생성
        recommendations = []
        if not optimizations['decoder_max_len_200']:
            recommendations.append("decoder_max_len을 200으로 설정하여 더 상세한 요약 생성")
        if not optimizations['eval_strategy_steps']:
            recommendations.append("eval_strategy를 'steps'로 변경하여 세밀한 모니터링")
        if not optimizations['gradient_checkpointing']:
            recommendations.append("gradient_checkpointing 활성화로 메모리 절약")
        if not optimizations['qlora_enabled']:
            recommendations.append("QLoRA 활성화로 메모리 효율성 향상")
        
        results['recommendations'] = recommendations
        
        self.logger.info(f"최적화 점수: {score:.1f}% ({sum(optimizations.values())}/{len(optimizations)})")
        
        return results
    
    def run_full_benchmark(self) -> Dict[str, Any]:
        """전체 벤치마크 실행"""
        self.logger.info("🚀 성능 벤치마크 시작")
        
        # 모든 벤치마크 실행
        benchmark_results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'config_path': self.config_path,
                'benchmark_version': '1.0'
            },
            'environment': self.benchmark_environment(),
            'model_loading': self.benchmark_model_loading(),
            'inference_speed': self.benchmark_inference_speed(),
            'configuration_impact': self.benchmark_configuration_impact()
        }
        
        # 결과 요약
        self._print_summary(benchmark_results)
        
        return benchmark_results
    
    def _print_summary(self, results: Dict[str, Any]):
        """벤치마크 결과 요약 출력"""
        print("\n" + "="*60)
        print("🎯 벤치마크 결과 요약")
        print("="*60)
        
        # 환경 정보
        env = results['environment']
        print(f"🖥️  환경: {env['device_info']['type']}")
        print(f"💾 메모리: {env['memory_info']['total_gb']:.1f}GB")
        
        # 모델 로딩 성능
        if 'memory_comparison' in results['model_loading']:
            mem_comp = results['model_loading']['memory_comparison']
            print(f"🚀 메모리 절약: {mem_comp['memory_saving_percent']:.1f}%")
        
        # 최적화 점수
        config_score = results['configuration_impact']['optimization_score']
        print(f"⚙️  최적화 점수: {config_score:.1f}%")
        
        # 권장사항
        recommendations = results['configuration_impact']['recommendations']
        if recommendations:
            print(f"💡 권장사항: {len(recommendations)}개")
            for rec in recommendations[:2]:  # 처음 2개만
                print(f"   - {rec}")
        else:
            print("✅ 모든 최적화 완료!")
        
        print("="*60)


def main():
    """메인 실행 함수"""
    benchmark = PerformanceBenchmark()
    results = benchmark.run_full_benchmark()
    
    # 결과를 JSON 파일로 저장
    output_file = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    import json
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 상세 결과가 저장되었습니다: {output_file}")


if __name__ == "__main__":
    main()
