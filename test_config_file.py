#!/usr/bin/env python3
"""
간단한 설정 파일 테스트
"""

import os
import sys
import yaml
from pathlib import Path

# 프로젝트 루트 경로 설정
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "code"))

print("🔍 설정 파일 테스트")
print("=" * 50)
print(f"현재 디렉토리: {os.getcwd()}")
print(f"프로젝트 루트: {project_root}")
print()

# 테스트할 설정 파일
test_config = "config/experiments/test_01_mt5_xlsum_1epoch.yaml"
config_path = project_root / test_config

print(f"📄 테스트 파일: {test_config}")
print(f"전체 경로: {config_path}")
print(f"파일 존재: {config_path.exists()}")
print()

if config_path.exists():
    print("✅ 파일을 찾았습니다. 내용 확인 중...")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print("\n📋 설정 파일 주요 내용:")
        print(f"  - experiment_name: {config.get('experiment_name', 'N/A')}")
        print(f"  - model_name: {config.get('general', {}).get('model_name', 'N/A')}")
        print(f"  - num_train_epochs: {config.get('training', {}).get('num_train_epochs', 'N/A')}")
        print(f"  - batch_size: {config.get('training', {}).get('per_device_train_batch_size', 'N/A')}")
        
        print("\n✨ 설정 파일이 정상입니다!")
        
    except Exception as e:
        print(f"❌ 설정 파일 읽기 오류: {e}")
else:
    print("❌ 파일을 찾을 수 없습니다!")
    print("\n🔍 config 디렉토리 확인:")
    config_dir = project_root / "config"
    if config_dir.exists():
        print(f"config 디렉토리 존재: ✅")
        experiments_dir = config_dir / "experiments"
        if experiments_dir.exists():
            print(f"config/experiments 디렉토리 존재: ✅")
            yaml_files = list(experiments_dir.glob("test_*.yaml"))
            print(f"\n테스트 YAML 파일 목록 ({len(yaml_files)}개):")
            for f in yaml_files[:5]:
                print(f"  - {f.name}")
        else:
            print(f"config/experiments 디렉토리 존재: ❌")
    else:
        print(f"config 디렉토리 존재: ❌")
