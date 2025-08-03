import torch
import os

print("CUDA Setup Check:")
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch CUDA version: {torch.version.cuda}")
print(f"Device count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    print(f"Current device: {torch.cuda.current_device()}")

print(f"\nEnvironment Variables:")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")
print(f"CUDA_ROOT: {os.environ.get('CUDA_ROOT', 'Not set')}")

# Test a simple CUDA operation
if torch.cuda.is_available():
    print(f"\nCUDA Test:")
    x = torch.randn(2, 3).cuda()
    print(f"Created tensor on GPU: {x.device}")
    print("✅ CUDA is working properly!")
else:
    print("❌ CUDA not available")
