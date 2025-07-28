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
import psutil
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
        print_header("실험 환경 검증 시작")
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
            ("설정 파일", self.check_config_files),
            ("메모리 및 디스크", self.check_resources),
            ("실행 권한", self.check_permissions),
            ("코드 무결성", self.check_code_integrity),
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
            "run_auto_experiments.sh",
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
                size_mb = filepath.stat().st_size / (1024**2)
                file_info[filename] = {
                    "size_mb": round(size_mb, 2),
                    "exists": True
                }
                
                # 간단한 유효성 검사
                try:
                    import pandas as pd
                    df = pd.read_csv(filepath, nrows=5)
                    
                    # test.csv는 summary가 없음
                    if filename == 'test.csv':
                        required_columns = ['fname', 'dialogue']
                    else:
                        required_columns = ['fname', 'dialogue', 'summary']
                    
                    missing_cols = [col for col in required_columns if col not in df.columns]
                    
                    if missing_cols:
                        self.errors.append(f"{filename}에 필수 컬럼 누락: {missing_cols}")
                        print_status(f"{filename}: 컬럼 누락 - {missing_cols}", "ERROR")
                    else:
                        print_status(f"{filename}: {size_mb:.1f}MB, 컬럼 정상", "SUCCESS")
                        file_info[filename]["rows"] = len(pd.read_csv(filepath))
                        
                except Exception as e:
                    self.errors.append(f"{filename} 읽기 실패: {str(e)}")
                    print_status(f"{filename}: 읽기 실패", "ERROR")
        
        details = {
            "missing_files": missing_files,
            "file_info": file_info,
            "data_dir": str(data_dir)
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
            filepath = self.project_root / config_file
            if filepath.exists():
                try:
                    import yaml
                    with open(filepath, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                    
                    # 필수 키 확인 (config.yaml의 구조에 맞게 수정)
                    if config_file == 'config.yaml':
                        # config.yaml은 general 키만 확인
                        required_keys = ['general']
                    else:
                        # base_config.yaml 등은 기존 키 확인
                        required_keys = ['model', 'training', 'tokenizer', 'generation']
                    
                    missing_keys = [key for key in required_keys if key not in config]
                    
                    if missing_keys:
                        invalid_configs.append(f"{config_file}: 필수 키 누락 - {missing_keys}")
                        print_status(f"{config_file}: 필수 키 누락", "ERROR")
                    else:
                        valid_configs.append(config_file)
                        print_status(f"{config_file}: 유효함", "SUCCESS")
                        
                except Exception as e:
                    invalid_configs.append(f"{config_file}: 파싱 실패 - {str(e)}")
                    print_status(f"{config_file}: 파싱 실패", "ERROR")
            else:
                print_status(f"{config_file}: 파일 없음", "WARNING")
        
        # 실험 설정 확인
        print_status(f"실험 설정 파일: {len(experiment_configs)}개 발견", "INFO")
        for exp_config in experiment_configs[:5]:  # 처음 5개만 확인
            try:
                import yaml
                with open(exp_config, 'r', encoding='utf-8') as f:
                    yaml.safe_load(f)
                print_status(f"  - {exp_config.name}: 유효함", "SUCCESS")
            except Exception as e:
                invalid_configs.append(f"{exp_config.name}: {str(e)}")
                print_status(f"  - {exp_config.name}: 파싱 실패", "ERROR")
        
        details = {
            "valid_configs": valid_configs,
            "invalid_configs": invalid_configs,
            "experiment_configs": len(experiment_configs)
        }
        
        if invalid_configs:
            self.errors.append(f"유효하지 않은 설정 파일: {invalid_configs}")
            return False, details
        
        return True, details
    
    def check_resources(self) -> Tuple[bool, Dict]:
        """시스템 리소스 확인"""
        # 메모리 확인
        mem = psutil.virtual_memory()
        mem_available_gb = mem.available / (1024**3)
        
        # 디스크 확인
        disk = psutil.disk_usage(str(self.project_root))
        disk_free_gb = disk.free / (1024**3)
        
        details = {
            "memory_available_gb": round(mem_available_gb, 2),
            "memory_percent_used": mem.percent,
            "disk_free_gb": round(disk_free_gb, 2),
            "disk_percent_used": disk.percent
        }
        
        # 최소 요구사항 확인
        min_memory_gb = 16  # 최소 16GB RAM
        min_disk_gb = 50    # 최소 50GB 여유 공간
        
        issues = []
        
        if mem_available_gb < min_memory_gb:
            issues.append(f"메모리 부족: {mem_available_gb:.1f}GB < {min_memory_gb}GB")
            print_status(f"메모리 부족: {mem_available_gb:.1f}GB 사용가능 (최소 {min_memory_gb}GB 권장)", "WARNING")
        else:
            print_status(f"메모리: {mem_available_gb:.1f}GB 사용가능", "SUCCESS")
        
        if disk_free_gb < min_disk_gb:
            issues.append(f"디스크 공간 부족: {disk_free_gb:.1f}GB < {min_disk_gb}GB")
            print_status(f"디스크 공간 부족: {disk_free_gb:.1f}GB 여유 (최소 {min_disk_gb}GB 권장)", "WARNING")
        else:
            print_status(f"디스크: {disk_free_gb:.1f}GB 여유", "SUCCESS")
        
        # CPU 사용률 확인
        cpu_percent = psutil.cpu_percent(interval=1)
        details["cpu_percent"] = cpu_percent
        
        if cpu_percent > 80:
            print_status(f"높은 CPU 사용률: {cpu_percent}%", "WARNING")
            self.warnings.append(f"CPU 사용률이 높음: {cpu_percent}%")
        else:
            print_status(f"CPU 사용률: {cpu_percent}%", "SUCCESS")
        
        if issues:
            self.warnings.extend(issues)
        
        return True, details
    
    def check_permissions(self) -> Tuple[bool, Dict]:
        """파일 권한 확인"""
        # 실행 권한이 필요한 스크립트
        scripts = [
            "run_auto_experiments.sh",
            "setup_env.sh",
            "check_env.sh"
        ]
        
        permission_issues = []
        
        for script in scripts:
            script_path = self.project_root / script
            if script_path.exists():
                if os.access(script_path, os.X_OK):
                    print_status(f"{script}: 실행 권한 있음", "SUCCESS")
                else:
                    permission_issues.append(script)
                    print_status(f"{script}: 실행 권한 없음", "ERROR")
                    # 권한 수정 시도
                    try:
                        script_path.chmod(0o755)
                        print_status(f"  → 실행 권한 부여됨", "SUCCESS")
                        permission_issues.remove(script)
                    except Exception as e:
                        print_status(f"  → 권한 수정 실패: {str(e)}", "ERROR")
        
        # 쓰기 권한 확인
        write_dirs = ["outputs", "logs", "models"]
        write_issues = []
        
        for dir_name in write_dirs:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                test_file = dir_path / ".write_test"
                try:
                    test_file.touch()
                    test_file.unlink()
                    print_status(f"{dir_name}: 쓰기 권한 있음", "SUCCESS")
                except Exception:
                    write_issues.append(dir_name)
                    print_status(f"{dir_name}: 쓰기 권한 없음", "ERROR")
        
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
            "code.trainer",
            "code.auto_experiment_runner",
            "code.utils.data_utils",
            "code.utils.metrics",
            "code.utils.device_utils",
            "code.utils.path_utils"
        ]
        
        import_errors = []
        successful_imports = []
        
        # 현재 디렉토리를 Python 경로에 추가
        original_path = sys.path.copy()
        sys.path.insert(0, str(self.project_root))
        
        for module in modules_to_test:
            try:
                importlib.import_module(module)
                successful_imports.append(module)
                print_status(f"{module}: import 성공", "SUCCESS")
            except ImportError as e:
                import_errors.append(f"{module}: {str(e)}")
                print_status(f"{module}: import 실패 - {str(e)}", "ERROR")
            except Exception as e:
                import_errors.append(f"{module}: {type(e).__name__} - {str(e)}")
                print_status(f"{module}: 오류 - {type(e).__name__}", "ERROR")
        
        # Python 경로 복원
        sys.path = original_path
        
        details = {
            "successful_imports": successful_imports,
            "import_errors": import_errors
        }
        
        if import_errors:
            self.errors.append(f"모듈 import 실패: {len(import_errors)}개")
            return False, details
        
        return True, details
    
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
