import sys
sys.path.insert(0, '.')
try:
    from code.trainer import NMTTrainer
    print('✓ code.trainer.NMTTrainer import 성공!')
except ImportError as e:
    print(f'✗ code.trainer import 실패: {e}')

try:
    from code.auto_experiment_runner import AutoExperimentRunner
    print('✓ code.auto_experiment_runner import 성공!')
except ImportError as e:
    print(f'✗ code.auto_experiment_runner import 실패: {e}')
