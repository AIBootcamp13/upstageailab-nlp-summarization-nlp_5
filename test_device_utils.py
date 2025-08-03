#!/usr/bin/env python3
import sys
import os
sys.path.append('/data/ephemeral/home/nlp-5/nlp-sum-lyj')

from code.utils.device_utils import get_robust_optimal_device, print_device_summary, get_rtx3090_extreme_config

print("=== Device Detection Test ===")
device, device_info = get_robust_optimal_device()
print(f"Device: {device}")
print(f"Device Info: {device_info}")
print()

print("=== RTX 3090 Extreme Config Test ===")
if 'RTX 3090' in device_info.device_name:
    print("RTX 3090 detected! Testing extreme optimization configs:")
    for arch in ['mt5', 'bart', 't5', 'kobart']:
        config = get_rtx3090_extreme_config(arch, True)  # Unsloth enabled
        print(f"\n{arch.upper()} Extreme Config:")
        print(f"  Batch size: {config['per_device_train_batch_size']}")
        print(f"  Effective batch: {config['effective_batch_size']}")
        print(f"  Workers: {config['dataloader_num_workers']}")
        print(f"  Speed improvement: {config['expected_speed_improvement']}x")
else:
    print("RTX 3090 not detected - extreme optimization not available")

print("\n=== Device Summary ===")
print_device_summary()
