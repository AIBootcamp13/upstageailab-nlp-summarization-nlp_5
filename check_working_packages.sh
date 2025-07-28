#!/bin/bash

echo "=== 현재 작동하는 패키지 버전으로 requirements.txt 업데이트 ==="

# 1. 현재 설치된 주요 패키지 버전 확인
echo "1. 현재 설치된 주요 패키지 버전:"
python -c "
import sys
packages_to_check = [
    'torch', 'torchvision', 'torchaudio', 'transformers', 'datasets', 
    'bitsandbytes', 'unsloth', 'accelerate', 'gradio', 'wandb'
]

working_versions = {}
for pkg in packages_to_check:
    try:
        module = __import__(pkg)
        version = getattr(module, '__version__', 'N/A')
        working_versions[pkg] = version
        print(f'{pkg}=={version}')
    except ImportError:
        print(f'{pkg}: Not installed')

print('\n=== 작동하는 버전들 ===')
for pkg, ver in working_versions.items():
    if ver != 'N/A':
        print(f'{pkg}=={ver}')
"

# 2. requirements.txt 백업
echo -e "\n2. 기존 requirements.txt 백업..."
cp requirements.txt requirements.txt.working_backup

echo -e "\n3. 현재 디렉토리 위치 확인:"
pwd

echo -e "\n4. uv로 설치된 패키지 목록:"
uv pip list | head -20

echo -e "\n=== 완료 ==="
echo "requirements.txt 업데이트는 수동으로 진행하세요."
echo "백업 파일: requirements.txt.working_backup"
