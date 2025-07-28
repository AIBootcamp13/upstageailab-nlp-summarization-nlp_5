#!/usr/bin/env python3
import sys
import os

print("Current directory:", os.getcwd())
print("Python path:")
for p in sys.path:
    print(f"  - {p}")

try:
    from code.trainer import Trainer
    print("\nSuccess: code.trainer imported successfully!")
except ImportError as e:
    print(f"\nError importing code.trainer: {e}")

try:
    import code
    print("code module:", code)
    print("code.__file__:", getattr(code, '__file__', 'No __file__ attribute'))
except ImportError as e:
    print(f"Error importing code: {e}")
