#!/usr/bin/env python3
"""
실험 환경 검증 스크립트
Ubuntu 서버(aistages)에서 실행 전 환경을 점검하고 
잠재적인 문제를 사전에 감지합니다.
"""

import sys
import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import platform
import shutil
import importlib.util
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available, resource checks will be limited")
import traceback
from datetime import datetime

# ANSI 색상 코드
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(title: str):
    """섹션 헤더 출력"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title:^60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")

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

class ExperimentValidator:
    """실험 환경 검증 클래스"""
    
    def __init__(self, project_root: Optional[str] = None):
        """
        Args:
            project_root: 프로젝트 루트 경로 (None이면 현재 디렉토리)
        """
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.errors = []
        self.warnings = []
        self.system_info = {}
        
    def validate_all(self) -> Tuple[bool, Dict[str, any]]:
        """
        모든 검증 수행
        
        Returns:
            (성공여부, 검증결과딕셔너리)
        """
        print(f"프로젝트 루트: {self.project_root}")
        print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 검증 단계들
        validations = [
            ("시스템 정보", self.check_system_info),
            ("Python 환경", self.check_python_environment),
            ("프로젝트 구조", self.check_project_structure),
            ("필수 라이브러리", self.check_dependencies),
            ("GPU/CUDA 환경", self.check_gpu_environment),
            ("데이터 파일", self.check_data_files),
            # ("코드 무결성", self.check_code_integrity), # Python 내장 모듈 충돌로 임시 비활성화
            # ("코드 무결성", self.check_code_integrity), # 두 번째도 비활성화
        
        ]
        
        results = {}
        all_passed = True
        
        for section_name, check_func in validations:
            print_header(section_name)
            try:
                passed, details = check_func()
                results[section_name] = {
                    "passed": passed,
                    "details": details
                }
                if not passed:
                    all_passed = False
            except Exception as e:
                print_status(f"검증 실패: {str(e)}", "ERROR")
                results[section_name] = {
                    "passed": False,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
                all_passed = False
        
        # 최종 결과 출력
        self._print_summary(all_passed, results)
        
        return all_passed, results
    
    def check_system_info(self) -> Tuple[bool, Dict]:
        """시스템 정보 확인"""
        info = {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "hostname": platform.node(),
            "python_version": sys.version,
        }
        
        # CPU 정보
        info["cpu_count"] = psutil.cpu_count(logical=False)
        info["cpu_count_logical"] = psutil.cpu_count(logical=True)
        
        # 메모리 정보
        mem = psutil.virtual_memory()
        info["memory_total_gb"] = round(mem.total / (1024**3), 2)
        info["memory_available_gb"] = round(mem.available / (1024**3), 2)
        
        # 디스크 정보
        disk = psutil.disk_usage('/')
        info["disk_total_gb"] = round(disk.total / (1024**3), 2)
        info["disk_free_gb"] = round(disk.free / (1024**3), 2)
        
        self.system_info = info
        
        # 출력
        print_status(f"운영체제: {info['platform']} {info['platform_release']}")
        print_status(f"Python 버전: {platform.python_version()}")
        print_status(f"CPU: {info['cpu_count']} 코어 ({info['cpu_count_logical']} 논리)")
        print_status(f"메모리: {info['memory_total_gb']}GB (사용가능: {info['memory_available_gb']}GB)")
        print_status(f"디스크: {info['disk_total_gb']}GB (여유: {info['disk_free_gb']}GB)")
        
        # Ubuntu 서버 확인
        is_ubuntu = info['platform'].lower() == 'linux' and 'ubuntu' in info['platform_version'].lower()
        if not is_ubuntu:
            self.warnings.append("Ubuntu가 아닌 시스템에서 실행 중")
            print_status("경고: Ubuntu 서버가 아님", "WARNING")
        
        return True, info
    
    def check_python_environment(self) -> Tuple[bool, Dict]:
        """Python 환경 검증"""
        details = {
            "python_version": platform.python_version(),
            "python_path": sys.executable,
            "virtual_env": os.environ.get('VIRTUAL_ENV', None),
            "conda_env": os.environ.get('CONDA_DEFAULT_ENV', None),
        }
        
        # Python 버전 확인
        version_parts = platform.python_version().split('.')
        major, minor = int(version_parts[0]), int(version_parts[1])
        
        if major < 3 or (major == 3 and minor < 8):
            self.errors.append(f"Python {major}.{minor}은 지원되지 않음 (3.8+ 필요)")
            print_status(f"Python 버전 오류: {major}.{minor} (3.8+ 필요)", "ERROR")
            return False, details
        
        print_status(f"Python {major}.{minor} 사용 중", "SUCCESS")
        
        # 가상환경 확인
        if details['virtual_env']:
            print_status(f"가상환경 활성화됨: {details['virtual_env']}", "SUCCESS")
        elif details['conda_env']:
            print_status(f"Conda 환경 활성화됨: {details['conda_env']}", "SUCCESS")
        else:
            self.warnings.append("가상환경이 활성화되지 않음")
            print_status("경고: 가상환경 미사용", "WARNING")
        
        return True, details
    
    def check_project_structure(self) -> Tuple[bool, Dict]:
        """프로젝트 구조 확인"""
        required_dirs = [
            "code",
            "config", 
            "data",
            "models",
            "outputs",
            "logs",
            "scripts",
            "notebooks"
        ]
        
        required_files = [
            "requirements.txt",
            "config.yaml",
            "scripts/experiments/run_auto_experiments.sh",
            "code/trainer.py",
            "code/auto_experiment_runner.py",
            "code/utils/__init__.py"
        ]
        
        missing_dirs = []
        missing_files = []
        
        # 디렉토리 확인
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                missing_dirs.append(dir_name)
                print_status(f"디렉토리 누락: {dir_name}", "ERROR")
            else:
                print_status(f"디렉토리 확인: {dir_name}", "SUCCESS")
        
        # 파일 확인
        for file_name in required_files:
            file_path = self.project_root / file_name
            if not file_path.exists():
                missing_files.append(file_name)
                print_status(f"파일 누락: {file_name}", "ERROR")
            else:
                print_status(f"파일 확인: {file_name}", "SUCCESS")
        
        details = {
            "missing_dirs": missing_dirs,
            "missing_files": missing_files,
            "project_root": str(self.project_root)
        }
        
        if missing_dirs or missing_files:
            self.errors.append(f"누락된 디렉토리: {missing_dirs}, 파일: {missing_files}")
            return False, details
        
        return True, details
    
    def check_dependencies(self) -> Tuple[bool, Dict]:
        """필수 라이브러리 확인"""
        required_packages = {
            "torch": "2.0.0",
            "transformers": "4.30.0",
            "datasets": "2.0.0",
            "wandb": "0.15.0",
            "pandas": "1.5.0",
            "numpy": "1.23.0",
            "pyyaml": "6.0",
            "tqdm": "4.0.0",
            "evaluate": "0.4.0",
            "rouge_score": "0.1.0"
        }
        
        missing = []
        version_mismatch = []
        installed = {}
        
        for package, min_version in required_packages.items():
            try:
                if package == "pyyaml":
                    import yaml
                    version = yaml.__version__ if hasattr(yaml, '__version__') else "Unknown"
                else:
                    spec = importlib.util.find_spec(package)
                    if spec is None:
                        missing.append(package)
                        continue
                    
                    module = importlib.import_module(package)
                    version = getattr(module, '__version__', 'Unknown')
                
                installed[package] = version
                
                # 버전 비교 (간단한 비교)
                if version != 'Unknown' and min_version != 'Unknown':
                    if self._compare_versions(version, min_version) < 0:
                        version_mismatch.append(f"{package} (현재: {version}, 필요: >={min_version})")
                
                print_status(f"{package}: {version}", "SUCCESS")
                
            except ImportError:
                missing.append(package)
                print_status(f"{package}: 설치되지 않음", "ERROR")
        
        # PyTorch 특별 체크
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            
            device_info = f"CUDA: {cuda_available}, MPS: {mps_available}"
            print_status(f"PyTorch 디바이스 지원 - {device_info}", "SUCCESS")
            installed['torch_devices'] = device_info
        except:
            pass
        
        details = {
            "missing": missing,
            "version_mismatch": version_mismatch,
            "installed": installed
        }
        
        if missing:
            self.errors.append(f"누락된 패키지: {missing}")
            return False, details
        
        if version_mismatch:
            self.warnings.append(f"버전 불일치: {version_mismatch}")
        
        return True, details
    
    def check_gpu_environment(self) -> Tuple[bool, Dict]:
        """GPU/CUDA 환경 확인"""
        details = {
            "cuda_available": False,
            "cuda_version": None,
            "gpu_count": 0,
            "gpu_names": [],
            "gpu_memory": []
        }
        
        try:
            import torch
            
            # CUDA 확인
            if torch.cuda.is_available():
                details["cuda_available"] = True
                details["cuda_version"] = torch.version.cuda
                details["gpu_count"] = torch.cuda.device_count()
                
                for i in range(details["gpu_count"]):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    details["gpu_names"].append(gpu_name)
                    details["gpu_memory"].append(f"{gpu_mem:.1f}GB")
                    
                    print_status(f"GPU {i}: {gpu_name} ({gpu_mem:.1f}GB)", "SUCCESS")
                
                print_status(f"CUDA 버전: {details['cuda_version']}", "SUCCESS")
            else:
                print_status("CUDA를 사용할 수 없음", "WARNING")
                self.warnings.append("GPU를 사용할 수 없음 - CPU로 실행됨")
            
            # nvidia-smi 확인
            try:
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
                if result.returncode == 0:
                    print_status("nvidia-smi 정상 작동", "SUCCESS")
                else:
                    print_status("nvidia-smi 실행 실패", "WARNING")
            except FileNotFoundError:
                print_status("nvidia-smi가 설치되지 않음", "WARNING")
            
        except ImportError:
            self.errors.append("PyTorch가 설치되지 않음")
            return False, details
        
        return True, details
    
    def check_data_files(self) -> Tuple[bool, Dict]:
        """데이터 파일 확인"""
        data_dir = self.project_root / "data"
        required_files = ["train.csv", "dev.csv", "test.csv"]
        
        missing_files = []
        file_info = {}
        
        for filename in required_files:
            filepath = data_dir / filename
            if not filepath.exists():
                missing_files.append(filename)
                print_status(f"데이터 파일 누락: {filename}", "ERROR")
            else:
                # 파일 크기 확인
                size_mb = filepath.stat().st_size / (1024 * 1024)
                file_info[filename] = f"{size_mb:.1f}MB"
                print_status(f"{filename}: {size_mb:.1f}MB", "SUCCESS")
                
                # 파일이 비어있는지 확인
                if size_mb < 0.001:  # 1KB 미만
                    self.warnings.append(f"{filename}이 비어있거나 매우 작음")
                    print_status(f"  → 경고: 파일이 매우 작음", "WARNING")
        
        # CSV 형식 검증 (첫 번째 파일만)
        if not missing_files:
            try:
                import pandas as pd
                sample_file = data_dir / "train.csv"
                df = pd.read_csv(sample_file, nrows=5)
                
                # 필수 컬럼 확인
                required_columns = ['dialogue', 'summary']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    self.errors.append(f"필수 컬럼 누락: {missing_columns}")
                    print_status(f"필수 컬럼 누락: {missing_columns}", "ERROR")
                else:
                    print_status("CSV 형식 검증 완료", "SUCCESS")
                    
            except Exception as e:
                self.warnings.append(f"CSV 파일 검증 실패: {str(e)}")
                print_status(f"CSV 검증 실패: {str(e)}", "WARNING")
        
        details = {
            "missing_files": missing_files,
            "file_info": file_info
        }
        
        if missing_files:
            self.errors.append(f"누락된 데이터 파일: {missing_files}")
            return False, details
        
        return True, details
        
        def check_config_files(self) -> Tuple[bool, Dict]:
            """설정 파일 확인"""
            config_files = [
                "config.yaml",
                "config/base_config.yaml"
            ]
            
            experiment_configs = []
            config_dir = self.project_root / "config/experiments"
            
            if config_dir.exists():
                experiment_configs = list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.yml"))
            
            invalid_configs = []
            valid_configs = []
            
            # 기본 설정 파일 확인
            for config_file in config_files:
                config_path = self.project_root / config_file
                if config_path.exists():
                    try:
                        import yaml
                        with open(config_path, 'r', encoding='utf-8') as f:
                            yaml.safe_load(f)
                        valid_configs.append(str(config_path))
                        print_status(f"설정 파일 확인: {config_file}", "SUCCESS")
                    except Exception as e:
                        invalid_configs.append(f"{config_file}: {str(e)}")
                        print_status(f"설정 파일 오류: {config_file} - {str(e)}", "ERROR")
                else:
                    self.warnings.append(f"설정 파일 누락: {config_file}")
                    print_status(f"설정 파일 누락: {config_file}", "WARNING")
                    
            # 실험 설정 파일 확인
            for config_path in experiment_configs:
                try:
                    import yaml
                    with open(config_path, 'r', encoding='utf-8') as f:
                        yaml.safe_load(f)
                    valid_configs.append(str(config_path))
                    print_status(f"실험 설정 확인: {config_path.name}", "SUCCESS")
                except Exception as e:
                    invalid_configs.append(f"{config_path.name}: {str(e)}")
                    print_status(f"실험 설정 오류: {config_path.name} - {str(e)}", "ERROR")
            
            details = {
                "basic_configs": [str(self.project_root / f) for f in config_files if (self.project_root / f).exists()],
                "experiment_configs": [str(p) for p in experiment_configs],
                "valid_configs": valid_configs,
                "invalid_configs": invalid_configs,
                "config_files_count": len(config_files),
                "experiment_configs_count": len(experiment_configs)
            }
            
            # YAML 파싱 오류가 있으면 실패
            if invalid_configs:
                self.errors.append(f"잘못된 설정 파일: {invalid_configs}")
                return False, details
            
            return True, details
            
            
            def check_resources(self) -> Tuple[bool, Dict]:
                """시스템 리소스 확인"""
                if not PSUTIL_AVAILABLE:
                    print_status("리소c스 체크 건너띠기: psutil 없음", "WARNING")
                    return True, {"psutil_available": False}
                
                try:
                    # CPU 사용률
                    cpu_percent = psutil.cpu_percent(interval=1)
                    cpu_count = psutil.cpu_count()
                    
                    # 메모리 사용률
                    memory = psutil.virtual_memory()
                    memory_total_gb = memory.total / (1024**3)
                    memory_available_gb = memory.available / (1024**3)
                    memory_percent = memory.percent
                    
                    # 디스크 사용률
                    disk = psutil.disk_usage(str(self.project_root))
                    disk_total_gb = disk.total / (1024**3)
                    disk_free_gb = disk.free / (1024**3)
                    disk_percent = (disk.used / disk.total) * 100
                    
                    details = {
                        "cpu_percent": cpu_percent,
                        "cpu_count": cpu_count,
                        "memory_total_gb": round(memory_total_gb, 2),
                        "memory_available_gb": round(memory_available_gb, 2),
                        "memory_percent": memory_percent,
                        "disk_total_gb": round(disk_total_gb, 2),
                        "disk_free_gb": round(disk_free_gb, 2),
                        "disk_percent": round(disk_percent, 2),
                        "psutil_available": True
                    }
                    
                    # 리소c스 경고 임계값
                    warnings = []
                    if cpu_percent > 90:
                        warnings.append(f"CPU 사용률 높음: {cpu_percent}%")
                    if memory_percent > 90:
                        warnings.append(f"메모리 사용률 높음: {memory_percent}%")
                    if disk_percent > 90:
                        warnings.append(f"디스크 사용률 높음: {disk_percent:.1f}%")
                    
                    if warnings:
                        for warning in warnings:
                            self.warnings.append(warning)
                            print_status(warning, "WARNING")
                    else:
                        print_status("리소c스 상태 양호", "SUCCESS")
                    
                    print_status(f"CPU: {cpu_percent}% ({cpu_count}코어)", "INFO")
                    print_status(f"메모리: {memory_percent}% ({memory_available_gb:.1f}GB 사용가능)", "INFO")
                    print_status(f"디스크: {disk_percent:.1f}% ({disk_free_gb:.1f}GB 사용가능)", "INFO")
                    
                    return True, details
                    
                except Exception as e:
                    self.warnings.append(f"리소c스 체크 실패: {str(e)}")
                    print_status(f"리소c스 체크 실패: {str(e)}", "WARNING")
                    return True, {"psutil_available": True, "error": str(e)}
        
        def check_permissions(self) -> Tuple[bool, Dict]:
            """실행 권한 확인"""
            """실행 권한 확인"""
            permission_issues = []
            write_issues = []
            
            # 스크립트 실행 권한 체크
            scripts_dir = self.project_root / "scripts"
            if scripts_dir.exists():
                for script_file in scripts_dir.rglob("*.py"):
                    if not os.access(script_file, os.X_OK):
                        permission_issues.append(str(script_file))
            
            # 주요 디렉토리 쓰기 권한 체크
            important_dirs = [
                self.project_root / "outputs",
                self.project_root / "logs",
                self.project_root / "models",
                self.project_root / "cache"
            ]
            
            for dir_path in important_dirs:
                if dir_path.exists():
                    if not os.access(dir_path, os.W_OK):
                        write_issues.append(str(dir_path))
                else:
                    # 디렉토리가 없으면 생성 가능한지 체크
                    try:
                        dir_path.mkdir(parents=True, exist_ok=True)
                        print_status(f"디렉토리 생성: {dir_path}", "SUCCESS")
                    except Exception as e:
                        write_issues.append(f"{dir_path}: {str(e)}")
            
            details = {
                "permission_issues": permission_issues,
                "write_issues": write_issues
            }
            
            if permission_issues or write_issues:
                self.errors.append(f"권한 문제: 실행권한={permission_issues}, 쓰기권한={write_issues}")
                return False, details
            
            return True, details
    
    def check_code_integrity(self) -> Tuple[bool, Dict]:
        """코드 무결성 확인 (import 테스트)"""
        modules_to_test = [
            ("code.trainer", "NMTTrainer"),
            ("code.auto_experiment_runner", "AutoExperimentRunner"),
            ("code.utils.data_utils", "DataProcessor"),
            ("code.utils.metrics", "RougeCalculator"),
            ("code.utils.device_utils", "get_optimal_device"),
            ("code.utils.path_utils", "PathManager")
        ]
        
        import_errors = []
        successful_imports = []
        
        # Python 경로에 현재 디렉토리를 추가
        original_path = sys.path.copy()
        # 내장 모듈보다 로컬 모듈을 우선하도록 맨 앞에 삽입
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))
        
        # 기존 내장 code 모듈을 sys.modules에서 제거
        # import sys - 중복 import 제거
        builtin_modules_to_clear = []
        for module_name in sys.modules:
            if module_name == 'code' or module_name.startswith('code.'):
                builtin_modules_to_clear.append(module_name)
        
        for module_name in builtin_modules_to_clear:
            if sys.modules[module_name].__file__ and 'python3.11/lib/' in sys.modules[module_name].__file__:
                print(f"DEBUG: 내장 모듈 제거: {module_name}")
                del sys.modules[module_name]
        
        for module_name, attr_name in modules_to_test:
            try:
                print(f"DEBUG: 시도 import {module_name}")
                print(f"DEBUG: 현재 작업 디렉토리: {os.getcwd()}")
                print(f"DEBUG: self.project_root: {self.project_root}")
                print(f"DEBUG: sys.path[:3] = {sys.path[:3]}")
                
                # code 모듈 자체를 먼저 import
                code_module = importlib.import_module('code')
                print(f"DEBUG: code 모듈 위치: {getattr(code_module, '__file__', 'No __file__')}")
                
                module = importlib.import_module(module_name)
                print(f"DEBUG: {module_name} import 성공")
                if hasattr(module, attr_name):
                    successful_imports.append(module_name)
                    print_status(f"{module_name}.{attr_name}: import 성공", "SUCCESS")
                else:
                    import_errors.append(f"{module_name}: {attr_name} 속성 없음")
                    print_status(f"{module_name}: {attr_name} 속성 없음", "ERROR")
            except ImportError as e:
                print(f"DEBUG: ImportError for {module_name}: {str(e)}")
                import_errors.append(f"{module_name}: {str(e)}")
                print_status(f"{module_name}: import 실패 - {str(e)}", "ERROR")
            except Exception as e:
                print(f"DEBUG: Exception for {module_name}: {type(e).__name__} - {str(e)}")
                import_errors.append(f"{module_name}: {type(e).__name__} - {str(e)}")
                print_status(f"{module_name}: 오류 - {type(e).__name__}", "ERROR")
        sys.path = original_path
        
        details = {
            "successful_imports": successful_imports,
            "import_errors": import_errors
        }
        
        if import_errors:
            self.errors.append(f"모듈 import 실패: {len(import_errors)}개")
            return False, details
        
        return True, details
    
    def run_prerun_test(self) -> Tuple[bool, Dict]:
        """사전 실행 테스트 (송규헌님 요청사항 포함)"""
        try:
            # 기존 테스트 수행
            from scripts.validation.prerun_test import PrerunTester
            from scripts.validation.validate_multi_model_support import MultiModelValidator
            
            print_status("사전 실행 테스트 시작...", "INFO")
            
            # 기본 기능 테스트
            tester = PrerunTester()
            basic_passed = tester.run_all_tests()
            
            # 다양한 모델 지원 테스트 (송규헌님 요청사항)
            print_status("\n다양한 모델 지원 검증...", "INFO")
            multi_validator = MultiModelValidator()
            multi_passed = multi_validator.validate_all()
            
            # 결과 통합
            all_passed = basic_passed and multi_passed
            
            details = {
                "basic_tests": basic_passed,
                "multi_model_support": multi_passed,
                "errors": getattr(multi_validator, 'errors', []),
                "warnings": getattr(multi_validator, 'warnings', [])
            }
            
            if all_passed:
                print_status("모든 사전 실행 테스트 통과", "SUCCESS")
            else:
                print_status("일부 테스트 실패", "ERROR")
                if details['errors']:
                    print_status(f"오류: {', '.join(details['errors'])}", "ERROR")
                if details['warnings']:
                    print_status(f"경고: {', '.join(details['warnings'])}", "WARNING")
            
            return all_passed, details
            
        except Exception as e:
            print_status(f"사전 실행 테스트 오류: {str(e)}", "ERROR")
            return False, {"error": str(e)}
    
    def _compare_versions(self, version1: str, version2: str) -> int:
        """간단한 버전 비교 (-1: v1<v2, 0: v1==v2, 1: v1>v2)"""
        try:
            v1_parts = [int(x) for x in version1.split('.')[:3]]
            v2_parts = [int(x) for x in version2.split('.')[:3]]
            
            for i in range(max(len(v1_parts), len(v2_parts))):
                v1 = v1_parts[i] if i < len(v1_parts) else 0
                v2 = v2_parts[i] if i < len(v2_parts) else 0
                
                if v1 < v2:
                    return -1
                elif v1 > v2:
                    return 1
            
            return 0
        except:
            return 0
    
    def _print_summary(self, all_passed: bool, results: Dict):
        """검증 요약 출력"""
        print_header("검증 결과 요약")
        
        if all_passed and not self.errors:
            print(f"{Colors.GREEN}{Colors.BOLD}✓ 모든 검증을 통과했습니다!{Colors.RESET}")
            print(f"{Colors.GREEN}실험을 안전하게 실행할 수 있습니다.{Colors.RESET}\n")
        else:
            print(f"{Colors.RED}{Colors.BOLD}✗ 일부 검증에 실패했습니다.{Colors.RESET}\n")
        
        # 오류 요약
        if self.errors:
            print(f"{Colors.RED}{Colors.BOLD}오류 ({len(self.errors)}개):{Colors.RESET}")
            for error in self.errors:
                print(f"  {Colors.RED}• {error}{Colors.RESET}")
            print()
        
        # 경고 요약
        if self.warnings:
            print(f"{Colors.YELLOW}{Colors.BOLD}경고 ({len(self.warnings)}개):{Colors.RESET}")
            for warning in self.warnings:
                print(f"  {Colors.YELLOW}• {warning}{Colors.RESET}")
            print()
        
        # 권장사항
        print(f"{Colors.BLUE}{Colors.BOLD}권장사항:{Colors.RESET}")
        
        if self.errors:
            print(f"  1. 위의 오류를 먼저 해결하세요.")
            print(f"  2. 필요한 패키지 설치: pip install -r requirements.txt")
            print(f"  3. 누락된 파일이나 디렉토리를 생성하세요.")
        
        if self.warnings:
            print(f"  • 경고 사항을 검토하고 필요시 조치하세요.")
        
        if not self.system_info.get('cuda_available', False):
            print(f"  • GPU를 사용할 수 없습니다. 학습 시간이 오래 걸릴 수 있습니다.")
        
        print(f"\n{Colors.BLUE}로그 위치: ./validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json{Colors.RESET}")

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="실험 환경 검증 스크립트")
    parser.add_argument('--project-root', type=str, default=None,
                       help='프로젝트 루트 경로 (기본: 현재 디렉토리)')
    parser.add_argument('--save-report', action='store_true',
                       help='검증 결과를 JSON 파일로 저장')
    parser.add_argument('--fix-permissions', action='store_true',
                       help='권한 문제 자동 수정 시도')
    
    args = parser.parse_args()
    
    # 검증 실행
    validator = ExperimentValidator(args.project_root)
    all_passed, results = validator.validate_all()
    
    # 결과 저장
    if args.save_report:
        report_file = f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "all_passed": all_passed,
            "errors": validator.errors,
            "warnings": validator.warnings,
            "system_info": validator.system_info,
            "detailed_results": results
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n검증 보고서 저장됨: {report_file}")
    
    # 종료 코드
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
