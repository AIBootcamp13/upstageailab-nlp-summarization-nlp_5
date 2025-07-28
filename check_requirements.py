#!/usr/bin/env python3
"""
의존성 패키지 확인 스크립트
"""

import sys
import subprocess
import importlib.util

def check_package(package_name):
    """패키지 설치 여부 확인"""
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        return False, None
    else:
        try:
            module = importlib.import_module(package_name)
            version = getattr(module, '__version__', 'unknown')
            return True, version
        except:
            return True, 'installed'

def check_requirements():
    """주요 패키지들의 설치 상태 확인"""
    
    print("🔍 Python 환경 정보:")
    print(f"Python 버전: {sys.version}")
    print(f"Python 경로: {sys.executable}")
    print("-" * 50)
    
    # 필수 패키지 목록
    required_packages = [
        'torch',
        'transformers',
        'datasets',
        'evaluate',
        'rouge',
        'rouge_score',
        'pandas',
        'numpy',
        'wandb',
        'requests',
        'charset_normalizer',  # requests 의존성
        'chardet',  # requests 의존성 대안
        'yaml',
        'tqdm',
        'sentencepiece',
        'accelerate',
        'bitsandbytes',
        'peft',
        'unsloth'
    ]
    
    print("\n📦 패키지 설치 상태:")
    print("-" * 50)
    
    missing_packages = []
    installed_packages = []
    
    for package in required_packages:
        installed, version = check_package(package)
        if installed:
            status = f"✅ {package:<20} : {version}"
            installed_packages.append(package)
        else:
            status = f"❌ {package:<20} : NOT INSTALLED"
            missing_packages.append(package)
        print(status)
    
    print("-" * 50)
    print(f"\n✅ 설치된 패키지: {len(installed_packages)}개")
    print(f"❌ 누락된 패키지: {len(missing_packages)}개")
    
    if missing_packages:
        print("\n⚠️  누락된 패키지 목록:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        
        print("\n💡 설치 명령어:")
        print(f"pip install {' '.join(missing_packages)}")
        
        # requirements.txt에서 정확한 버전 찾기
        print("\n📋 requirements.txt에서 권장 버전:")
        try:
            with open('requirements.txt', 'r') as f:
                lines = f.readlines()
                for pkg in missing_packages:
                    for line in lines:
                        if line.strip().startswith(pkg):
                            print(f"  - {line.strip()}")
                            break
        except:
            print("  requirements.txt 파일을 찾을 수 없습니다.")
    
    # CUDA 정보 확인
    print("\n🖥️  GPU/CUDA 정보:")
    print("-" * 50)
    try:
        import torch
        print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA 버전: {torch.version.cuda}")
            print(f"GPU 개수: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    except:
        print("PyTorch가 설치되지 않았습니다.")
    
    return len(missing_packages) == 0

if __name__ == "__main__":
    success = check_requirements()
    sys.exit(0 if success else 1)
