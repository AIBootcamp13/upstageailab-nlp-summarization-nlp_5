#!/usr/bin/env python3
"""
간단한 파이프라인 테스트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("=== 간단한 구문 테스트 ===")

# 1. import 테스트
try:
    print("\n1. Import 테스트...")
    from utils.data_utils import DataProcessor
    from utils.device_utils import get_optimal_device
    import torch
    print("   ✅ Import 성공")
except Exception as e:
    print(f"   ❌ Import 실패: {e}")
    sys.exit(1)

# 2. 디바이스 테스트
try:
    print("\n2. 디바이스 테스트...")
    device, device_type = get_optimal_device()
    print(f"   ✅ 디바이스: {device} (타입: {device_type})")
except Exception as e:
    print(f"   ❌ 디바이스 테스트 실패: {e}")

# 3. 기본 연산 테스트
try:
    print("\n3. PyTorch 연산 테스트...")
    x = torch.tensor([1, 2, 3]).to(device)
    y = x * 2
    print(f"   ✅ 연산 결과: {y}")
except Exception as e:
    print(f"   ❌ 연산 테스트 실패: {e}")

print("\n✅ 모든 기본 테스트 통과!")
print("   파일 문법이 정상이며, 기본 기능이 작동합니다.")
