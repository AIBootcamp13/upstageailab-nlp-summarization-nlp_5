#!/usr/bin/env python3
"""
송규헌님 요청사항 구현 검증 스크립트
다양한 모델 지원 및 unsloth 적용 관련 기능을 검증합니다.
"""

import sys
import os
from pathlib import Path
import json
import traceback
from typing import Dict, List, Tuple, Optional
import yaml

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# ANSI 색상 코드
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_status(message: str, status: str = "INFO"):
    """상태 메시지 출력"""
    if status == "SUCCESS":
        print(f"{Colors.GREEN}✓{Colors.RESET} {message}")
    elif status == "WARNING":
        print(f"{Colors.YELLOW}⚠{Colors.RESET} {message}")
    elif status == "ERROR":
        print(f"{Colors.RED}✗{Colors.RESET} {message}")
    else:
        print(f"{Colors.BLUE}ℹ{Colors.RESET} {message}")

def print_header(title: str):
    """섹션 헤더 출력"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title:^60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")


class MultiModelValidator:
    """다양한 모델 지원 검증 클래스"""
    
    def __init__(self):
        self.supported_models = {
            'bart': ['facebook/bart-base', 'facebook/bart-large'],
            't5': ['t5-small', 't5-base', 't5-large'],
            'mt5': ['google/mt5-small', 'google/mt5-base'],
            'flan-t5': ['google/flan-t5-base', 'google/flan-t5-large'],
            'kogpt2': ['skt/kogpt2-base-v2'],
            'kobart': ['gogamza/kobart-base-v2', 'gogamza/kobart-summarization']
        }
        self.errors = []
        self.warnings = []
    
    def validate_all(self) -> bool:
        """모든 검증 수행"""
        print_header("송규헌님 요청사항 구현 검증")
        
        all_passed = True
        
        # 1. trainer.py 코드 검증
        if not self.check_trainer_code():
            all_passed = False
        
        # 2. 다양한 모델 설정 파일 검증
        if not self.check_model_configs():
            all_passed = False
        
        # 3. 모델별 전처리 함수 검증
        if not self.check_model_preprocessing():
            all_passed = False
        
        # 4. unsloth 지원 검증
        if not self.check_unsloth_support():
            all_passed = False
        
        # 5. 실행 스크립트 검증
        if not self.check_execution_scripts():
            all_passed = False
        
        # 결과 요약
        self.print_summary()
        
        return all_passed
    
    def check_trainer_code(self) -> bool:
        """trainer.py 코드 구조 검증"""
        print_header("1. trainer.py 코드 검증")
        
        trainer_path = project_root / "code" / "trainer.py"
        if not trainer_path.exists():
            print_status("trainer.py 파일이 없습니다", "ERROR")
            self.errors.append("trainer.py 미존재")
            return False
        
        try:
            with open(trainer_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 필수 함수/클래스 확인
            required_components = {
                'AutoModelForSeq2SeqLM': '다양한 Seq2Seq 모델 지원',
                'AutoModelForCausalLM': '다양한 Causal LM 지원',
                '_load_model_with_unsloth': 'unsloth 모델 로딩',
                '_load_model_with_qlora': 'QLoRA 모델 로딩',
                '_preprocess_for_model': '모델별 전처리',
                '_get_model_specific_config': '모델별 설정'
            }
            
            missing = []
            for component, desc in required_components.items():
                if component not in content:
                    missing.append(f"{component} ({desc})")
                else:
                    print_status(f"{component} 구현 확인됨", "SUCCESS")
            
            if missing:
                print_status(f"누락된 구성요소: {', '.join(missing)}", "ERROR")
                self.errors.extend(missing)
                return False
            
            # 아키텍처 지원 확인
            architectures = ['bart', 't5', 'mt5', 'flan-t5', 'kogpt2', 'kobart']
            supported = []
            for arch in architectures:
                if f"'{arch}'" in content or f'"{arch}"' in content:
                    supported.append(arch)
            
            print_status(f"지원 아키텍처: {', '.join(supported)}", "INFO")
            
            return True
            
        except Exception as e:
            print_status(f"trainer.py 검증 실패: {e}", "ERROR")
            self.errors.append(f"trainer.py 검증 오류: {str(e)}")
            return False
    
    def check_model_configs(self) -> bool:
        """모델별 설정 파일 검증"""
        print_header("2. 모델 설정 파일 검증")
        
        config_dir = project_root / "config" / "model_configs"
        if not config_dir.exists():
            print_status("model_configs 디렉토리가 없습니다", "ERROR")
            self.errors.append("model_configs 디렉토리 미존재")
            return False
        
        expected_configs = [
            'bart_base.yaml',
            't5_base.yaml',
            'mt5_base.yaml',
            'flan_t5_base.yaml',
            'kogpt2.yaml',
            'kobart_unsloth.yaml'
        ]
        
        all_valid = True
        for config_name in expected_configs:
            config_path = config_dir / config_name
            if not config_path.exists():
                print_status(f"{config_name} 파일 없음", "ERROR")
                self.errors.append(f"{config_name} 미존재")
                all_valid = False
                continue
            
            # YAML 파싱 테스트
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                # 필수 섹션 확인
                required_sections = ['model', 'training', 'tokenizer', 'generation']
                missing_sections = [s for s in required_sections if s not in config]
                
                if missing_sections:
                    print_status(f"{config_name}: 누락된 섹션 - {missing_sections}", "WARNING")
                    self.warnings.append(f"{config_name}: {missing_sections} 섹션 누락")
                else:
                    print_status(f"{config_name} 유효함", "SUCCESS")
                    
                    # unsloth 설정 확인
                    if 'unsloth' in config_name and 'qlora' in config:
                        if config['qlora'].get('use_unsloth'):
                            print_status(f"  → unsloth 활성화됨", "INFO")
                
            except Exception as e:
                print_status(f"{config_name} 파싱 실패: {e}", "ERROR")
                self.errors.append(f"{config_name} 파싱 오류")
                all_valid = False
        
        return all_valid
    
    def check_model_preprocessing(self) -> bool:
        """모델별 전처리 함수 검증"""
        print_header("3. 모델별 전처리 검증")
        
        try:
            # DataProcessor import 테스트
            from code.utils.data_utils import DataProcessor
            print_status("DataProcessor import 성공", "SUCCESS")
            
            # 모델별 전처리 시뮬레이션
            test_examples = {
                'input': ['대화 예시 텍스트'],
                'target': ['요약 예시 텍스트']
            }
            
            # trainer.py에서 _preprocess_for_model 함수 확인
            trainer_path = project_root / "code" / "trainer.py"
            with open(trainer_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # T5 prefix 처리 확인
            if 'summarize:' in content:
                print_status("T5 prefix 처리 구현 확인됨", "SUCCESS")
            else:
                print_status("T5 prefix 처리 미구현", "WARNING")
                self.warnings.append("T5 prefix 처리 미확인")
            
            # GPT TL;DR 처리 확인
            if 'TL;DR:' in content:
                print_status("GPT TL;DR 처리 구현 확인됨", "SUCCESS")
            else:
                print_status("GPT TL;DR 처리 미구현", "WARNING")
                self.warnings.append("GPT TL;DR 처리 미확인")
            
            return True
            
        except Exception as e:
            print_status(f"전처리 검증 실패: {e}", "ERROR")
            self.errors.append(f"전처리 검증 오류: {str(e)}")
            return False
    
    def check_unsloth_support(self) -> bool:
        """unsloth 라이브러리 지원 검증"""
        print_header("4. unsloth 지원 검증")
        
        # unsloth 설치 확인
        try:
            import unsloth
            print_status("unsloth 라이브러리 설치됨", "SUCCESS")
            print_status(f"  → 버전: {unsloth.__version__}", "INFO")
            unsloth_available = True
        except ImportError:
            print_status("unsloth 라이브러리 미설치", "WARNING")
            self.warnings.append("unsloth 미설치 (선택사항)")
            unsloth_available = False
        except NotImplementedError as e:
            print_status("unsloth 라이브러리 설치됨 (GPU 비호환 환경)", "WARNING")
            print_status(f"  → {str(e)}", "INFO")
            self.warnings.append("unsloth GPU 비호환 환경 (Apple Silicon)")
            unsloth_available = False
        except Exception as e:
            print_status(f"unsloth 확인 실패: {e}", "WARNING")
            self.warnings.append(f"unsloth 확인 오류: {str(e)}")
            unsloth_available = False
        
        # PyTorch 버전 확인
        try:
            import torch
            torch_version = torch.__version__
            major, minor = map(int, torch_version.split('.')[:2])
            
            if major >= 2 and minor >= 4:
                print_status(f"PyTorch {torch_version} - unsloth 호환", "SUCCESS")
            else:
                print_status(f"PyTorch {torch_version} - unsloth 비호환 (2.4+ 필요)", "WARNING")
                self.warnings.append("PyTorch 2.4+ 필요")
        except Exception as e:
            print_status(f"PyTorch 버전 확인 실패: {e}", "ERROR")
        
        # 설치 스크립트 확인
        install_script = project_root / "install_unsloth.sh"
        if install_script.exists():
            print_status("unsloth 설치 스크립트 존재", "SUCCESS")
            if os.access(install_script, os.X_OK):
                print_status("  → 실행 권한 있음", "SUCCESS")
            else:
                print_status("  → 실행 권한 없음", "WARNING")
        else:
            print_status("install_unsloth.sh 없음", "WARNING")
            self.warnings.append("unsloth 설치 스크립트 미존재")
        
        return True  # unsloth는 선택사항이므로 True 반환
    
    def check_execution_scripts(self) -> bool:
        """실행 스크립트 검증"""
        print_header("5. 실행 스크립트 검증")
        
        # 다중 모델 실험 스크립트
        multi_script = project_root / "run_multi_model_experiments.sh"
        if not multi_script.exists():
            print_status("run_multi_model_experiments.sh 없음", "ERROR")
            self.errors.append("다중 모델 실험 스크립트 미존재")
            return False
        
        print_status("run_multi_model_experiments.sh 존재", "SUCCESS")
        
        # 실행 권한 확인
        if os.access(multi_script, os.X_OK):
            print_status("  → 실행 권한 있음", "SUCCESS")
        else:
            print_status("  → 실행 권한 없음 (chmod +x 필요)", "WARNING")
            self.warnings.append("실행 권한 설정 필요")
        
        # 스크립트 내용 검증
        try:
            with open(multi_script, 'r') as f:
                content = f.read()
            
            # 모델 목록 확인
            models = ['bart_base', 't5_base', 'mt5_base', 'flan_t5_base']
            included = [m for m in models if m in content]
            print_status(f"  → 포함된 모델: {', '.join(included)}", "INFO")
            
        except Exception as e:
            print_status(f"스크립트 내용 검증 실패: {e}", "WARNING")
        
        return True
    
    def print_summary(self):
        """검증 결과 요약"""
        print_header("검증 결과 요약")
        
        total_errors = len(self.errors)
        total_warnings = len(self.warnings)
        
        print(f"{Colors.BOLD}총 오류: {Colors.RED}{total_errors}{Colors.RESET}")
        print(f"{Colors.BOLD}총 경고: {Colors.YELLOW}{total_warnings}{Colors.RESET}")
        
        if total_errors == 0:
            print(f"\n{Colors.GREEN}{Colors.BOLD}✓ 송규헌님 요청사항이 올바르게 구현되었습니다!{Colors.RESET}")
            print(f"{Colors.GREEN}다양한 모델로 실험을 진행할 수 있습니다.{Colors.RESET}")
            
            print(f"\n{Colors.BLUE}실행 방법:{Colors.RESET}")
            print("  1. 개별 모델 실험:")
            print("     python code/trainer.py --config config/model_configs/bart_base.yaml")
            print("  2. 다중 모델 실험:")
            print("     ./run_multi_model_experiments.sh")
            print("  3. unsloth 설치 (선택):")
            print("     ./install_unsloth.sh")
            
            if total_warnings > 0:
                print(f"\n{Colors.YELLOW}참고사항:{Colors.RESET}")
                for warning in self.warnings:
                    print(f"  - {warning}")
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}✗ 구현에 문제가 있습니다!{Colors.RESET}")
            print(f"{Colors.RED}다음 오류를 수정해주세요:{Colors.RESET}")
            for error in self.errors:
                print(f"  - {error}")
            
            print(f"\n{Colors.YELLOW}해결 방법:{Colors.RESET}")
            print("  1. trainer.py 파일 확인")
            print("  2. config/model_configs/ 디렉토리 생성 및 설정 파일 추가")
            print("  3. run_multi_model_experiments.sh 스크립트 생성")


def main():
    """메인 함수"""
    validator = MultiModelValidator()
    success = validator.validate_all()
    
    # 결과를 파일로 저장
    report = {
        'success': success,
        'errors': validator.errors,
        'warnings': validator.warnings,
        'timestamp': str(Path(__file__).stat().st_mtime)
    }
    
    report_path = project_root / "validation_logs" / "multi_model_validation_report.json"
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n검증 보고서 저장: {report_path}")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
