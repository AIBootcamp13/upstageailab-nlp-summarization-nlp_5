try:
    import unsloth
    print('unsloth: AVAILABLE')
    from unsloth import FastLanguageModel
    print('FastLanguageModel: AVAILABLE')
except ImportError as e:
    print('unsloth: NOT INSTALLED')
    print(f'Error: {e}')

try:
    import peft
    print('peft: AVAILABLE')
except ImportError:
    print('peft: NOT INSTALLED')

try:
    import bitsandbytes
    print('bitsandbytes: AVAILABLE')
except ImportError:
    print('bitsandbytes: NOT INSTALLED')
    
print('\n=== Package Versions ===')
try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
except ImportError:
    print('PyTorch: NOT INSTALLED')
