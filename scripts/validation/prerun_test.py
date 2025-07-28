#!/usr/bin/env python3
"""
사전 실행 테스트 스크립트
실제 실험 실행 전에 주요 기능들을 빠르게 테스트합니다.
"""

import sys
import os
from pathlib import Path
import time
import traceback
import json
import tempfile
from typing import Dict, List, Tuple, Optional

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent if '__file__' in globals() else Path.cwd()
sys.path.insert(0, str(project_root))

class PrerunTester:
    """사전 실행 테스터"""
    
    def __init__(self):
        self.test_results = {}
        self.errors = []
        
    def run_all_tests(self) -> bool:
        """모든 테스트 실행"""
        print("=" * 60)
        print("사전 실행 테스트 시작")
        print("=" * 60)
        
        tests = [
            ("설정 파일 로딩", self.test_config_loading),
            ("데이터 로딩", self.test_data_loading),
            ("모델 초기화", self.test_model_init),
            ("토크나이저", self.test_tokenizer),
            ("학습 단계", self.test_training_step),
            ("실험 추적", self.test_experiment_tracking),
            ("디바이스 감지", self.test_device_detection),
            ("메트릭 계산", self.test_metrics),
        ]
        
        all_passed = True
        
        for test_name, test_func in tests:
            print(f"\n[테스트] {test_name}")
            print("-" * 40)
            
            try:
                start_time = time.time()
                passed, details = test_func()
                elapsed = time.time() - start_time
                
                self.test_results[test_name] = {
                    "passed": passed,
                    "details": details,
                    "elapsed_seconds": elapsed
                }
                
                if passed:
                    print(f"✓ 성공 ({elapsed:.2f}초)")
                else:
                    print(f"✗ 실패")
                    all_passed = False
                    
            except Exception as e:
                print(f"✗ 오류 발생: {type(e).__name__}: {str(e)}")
                self.test_results[test_name] = {
                    "passed": False,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
                all_passed = False
        
        self._print_summary(all_passed)
        return all_passed
    
    def test_config_loading(self) -> Tuple[bool, Dict]:
        """설정 파일 로딩 테스트"""
        try:
            from code.utils import load_config
            
            # 기본 설정 로딩
            # 기본 설정 로딩
            base_config = load_config("config/base_config.yaml")
            
            # 필수 키 확인
            required_keys = ['model', 'training', 'tokenizer', 'data']
            missing = [k for k in required_keys if k not in base_config]
            
            if missing:
                print(f"  필수 키 누락: {missing}")
                return False, {"missing_keys": missing}
            
            print(f"  기본 설정 로딩 성공")
            
            # 실험 설정 확인
            exp_dir = Path("config/experiments")
            if exp_dir.exists():
                exp_configs = list(exp_dir.glob("*.yaml")) + list(exp_dir.glob("*.yml"))
                print(f"  실험 설정 파일: {len(exp_configs)}개")
            
            return True, {"config_keys": list(base_config.keys())}
            
        except Exception as e:
            return False, {"error": str(e)}
    
    def test_data_loading(self) -> Tuple[bool, Dict]:
        """데이터 로딩 테스트"""
        try:
            import pandas as pd
            from code.utils.data_utils import DataProcessor
            
            # 샘플 데이터 생성
            sample_data = pd.DataFrame({
                'id': ['test1', 'test2'],
                'dialogue': [
                    "#Person1#: 안녕하세요? #Person2#: 네, 안녕하세요!",
                    "#Person1#: 오늘 날씨가 좋네요. #Person2#: 정말 그렇네요."
                ],
                'summary': [
                    "두 사람이 인사를 나눴다.",
                    "날씨에 대해 대화를 나눴다."
                ]
            })
            
            # 임시 파일 생성
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                sample_data.to_csv(f, index=False)
                temp_path = f.name
            
            try:
                # 더미 토크나이저로 테스트
                class DummyTokenizer:
                    def __call__(self, text, **kwargs):
                        return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
                
                processor = DataProcessor(
                    tokenizer=DummyTokenizer(),
                    config={'tokenizer': {'encoder_max_len': 512, 'decoder_max_len': 100}}
                )
                
                # 데이터 로딩
                data = processor.load_data(temp_path)
                print(f"  데이터 로딩: {len(data)}개 샘플")
                
                # 데이터 처리
                processed = processor.process_data(data, is_training=True)
                print(f"  데이터 처리 완료")
                
                return True, {"sample_count": len(data)}
                
            finally:
                # 임시 파일 삭제
                os.unlink(temp_path)
                
        except Exception as e:
            return False, {"error": str(e)}
    
    def test_model_init(self) -> Tuple[bool, Dict]:
        """모델 초기화 테스트 (실제 로딩 없이)"""
        try:
            from transformers import AutoConfig
            
            # 모델 설정만 확인 (실제 가중치는 로딩하지 않음)
            model_name = "gogamza/kobart-base-v2"
            config = AutoConfig.from_pretrained(model_name)
            
            print(f"  모델 아키텍처: {config.model_type}")
            print(f"  히든 크기: {config.hidden_size}")
            print(f"  레이어 수: {config.num_hidden_layers}")
            
            return True, {
                "model_type": config.model_type,
                "hidden_size": config.hidden_size,
                "num_layers": config.num_hidden_layers
            }
            
        except Exception as e:
            return False, {"error": str(e)}
    
    def test_tokenizer(self) -> Tuple[bool, Dict]:
        """토크나이저 테스트"""
        try:
            from transformers import AutoTokenizer
            
            tokenizer = AutoTokenizer.from_pretrained("gogamza/kobart-base-v2")
            
            # 테스트 텍스트
            test_text = "#Person1#: 안녕하세요? #Person2#: 네, 반갑습니다!"
            
            # 인코딩
            encoded = tokenizer(
                test_text,
                max_length=128,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            
            # 디코딩
            decoded = tokenizer.decode(encoded['input_ids'][0], skip_special_tokens=True)
            
            print(f"  원본: {test_text}")
            print(f"  토큰 수: {len(encoded['input_ids'][0])}")
            print(f"  디코딩: {decoded[:50]}...")
            
            return True, {
                "vocab_size": tokenizer.vocab_size,
                "special_tokens": list(tokenizer.special_tokens_map.keys())
            }
            
        except Exception as e:
            return False, {"error": str(e)}
    
    def test_training_step(self) -> Tuple[bool, Dict]:
        """학습 단계 시뮬레이션"""
        try:
            import torch
            import torch.nn as nn
            
            # 간단한 더미 모델
            class DummyModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(10, 1)
                
                def forward(self, x):
                    return self.linear(x)
            
            model = DummyModel()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # 더미 데이터
            x = torch.randn(4, 10)
            y = torch.randn(4, 1)
            
            # Forward pass
            output = model(x)
            loss = nn.MSELoss()(output, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            print(f"  더미 학습 단계 성공")
            print(f"  Loss: {loss.item():.4f}")
            
            return True, {"loss": loss.item()}
            
        except Exception as e:
            return False, {"error": str(e)}
    
    def test_experiment_tracking(self) -> Tuple[bool, Dict]:
        """실험 추적 시스템 테스트"""
        try:
            from code.utils.experiment_utils import ExperimentTracker
            
            # 임시 디렉토리에서 테스트
            with tempfile.TemporaryDirectory() as temp_dir:
                tracker = ExperimentTracker(temp_dir)
                
                # 실험 시작
                exp_id = tracker.start_experiment(
                    name="test_experiment",
                    description="테스트 실험",
                    config={"test": True}
                )
                
                print(f"  실험 ID: {exp_id}")
                
                # 메트릭 로깅
                tracker.log_metrics({"loss": 0.5, "accuracy": 0.8}, step=1)
                
                # 실험 종료
                summary = tracker.end_experiment(
                    exp_id, 
                    {"final_score": 0.9}, 
                    "completed"
                )
                
                print(f"  실험 추적 성공")
                
                return True, {"experiment_id": exp_id}
                
        except Exception as e:
            return False, {"error": str(e)}
    
    def test_device_detection(self) -> Tuple[bool, Dict]:
        """디바이스 감지 테스트"""
        try:
            from code.utils.device_utils import get_optimal_device
            
            device, device_info = get_optimal_device()
            print(f"  감지된 디바이스: {device} ({device_info})")
            
            # PyTorch 디바이스 정보
            import torch
            cuda_available = torch.cuda.is_available()
            cuda_count = torch.cuda.device_count() if cuda_available else 0
            
            details = {
                "device": str(device),
                "device_info": str(device_info),
                "cuda_available": cuda_available,
                "cuda_count": cuda_count
            }
            
            if cuda_available:
                print(f"  CUDA 디바이스: {cuda_count}개")
                for i in range(cuda_count):
                    print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
            
            return True, details
            
        except Exception as e:
            return False, {"error": str(e)}
    
    def test_metrics(self) -> Tuple[bool, Dict]:
        """메트릭 계산 테스트"""
        try:
            from code.utils.metrics import RougeCalculator
            
            # 더미 토크나이저
            class DummyTokenizer:
                def __init__(self):
                    self.eos_token = '</s>'
            
            calculator = RougeCalculator(
                tokenizer=DummyTokenizer(),
                use_stemmer=True,
                tokenize_korean=True
            )
            
            # 테스트 데이터
            predictions = ["날씨가 좋다고 대화를 나눴다."]
            references = ["두 사람이 날씨에 대해 이야기했다."]
            
            # 더미 토크나이저
            scores = calculator.compute_metrics(predictions, references)
            
            print(f"  ROUGE-1: {scores.get('rouge1', 0):.4f}")
            print(f"  ROUGE-2: {scores.get('rouge2', 0):.4f}")
            print(f"  ROUGE-L: {scores.get('rougeL', 0):.4f}")
            
            return True, scores
            
        except Exception as e:
            return False, {"error": str(e)}
    
    def _print_summary(self, all_passed: bool):
        """테스트 요약 출력"""
        print("\n" + "=" * 60)
        print("테스트 요약")
        print("=" * 60)
        
        passed_count = sum(1 for r in self.test_results.values() if r.get('passed', False))
        total_count = len(self.test_results)
        
        print(f"총 테스트: {total_count}개")
        print(f"성공: {passed_count}개")
        print(f"실패: {total_count - passed_count}개")
        
        if all_passed:
            print("\n✓ 모든 테스트를 통과했습니다!")
            print("실험을 안전하게 실행할 수 있습니다.")
        else:
            print("\n✗ 일부 테스트가 실패했습니다.")
            print("실패한 테스트:")
            for test_name, result in self.test_results.items():
                if not result.get('passed', False):
                    error = result.get('error', result.get('details', 'Unknown error'))
                    print(f"  - {test_name}: {error}")

def run_quick_test():
    """빠른 테스트 실행 (최소한의 검증)"""
    print("빠른 사전 테스트 실행 중...")
    
    try:
        # 필수 모듈 import 테스트
        import torch
        import transformers
        import datasets
        import pandas
        import yaml
        
        print("✓ 필수 라이브러리 import 성공")
        
        # 프로젝트 구조 확인
        required_files = [
            "code/trainer.py",
            "code/auto_experiment_runner.py", 
            "config/base_config.yaml",
            "requirements.txt"
        ]
        
        missing = [f for f in required_files if not Path(f).exists()]
        if missing:
            print(f"✗ 필수 파일 누락: {missing}")
            return False
        
        print("✓ 프로젝트 구조 확인 완료")
        
        # PyTorch 디바이스 확인
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✓ PyTorch 디바이스: {device}")
        
        return True
        
    except ImportError as e:
        print(f"✗ 필수 라이브러리 누락: {e}")
        return False
    except Exception as e:
        print(f"✗ 오류 발생: {e}")
        return False

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="사전 실행 테스트")
    parser.add_argument('--quick', action='store_true',
                       help='빠른 테스트만 실행')
    parser.add_argument('--save-report', action='store_true',
                       help='테스트 결과를 파일로 저장')
    
    args = parser.parse_args()
    
    if args.quick:
        # 빠른 테스트
        success = run_quick_test()
        sys.exit(0 if success else 1)
    else:
        # 전체 테스트
        tester = PrerunTester()
        all_passed = tester.run_all_tests()
        
        # 결과 저장
        if args.save_report:
            report_file = f"prerun_test_report_{int(time.time())}.json"
            with open(report_file, 'w') as f:
                json.dump({
                    "all_passed": all_passed,
                    "test_results": tester.test_results,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }, f, indent=2)
            print(f"\n테스트 보고서 저장됨: {report_file}")
        
        sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
