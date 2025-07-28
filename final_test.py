#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

print("=== Import 테스트 ===")

# 1. trainer 모듈 테스트
try:
    from code.trainer import NMTTrainer
    print('✓ code.trainer.NMTTrainer import 성공!')
except ImportError as e:
    print(f'✗ code.trainer import 실패: {e}')

# 2. auto_experiment_runner 모듈 테스트  
try:
    from code.auto_experiment_runner import AutoExperimentRunner
    print('✓ code.auto_experiment_runner.AutoExperimentRunner import 성공!')
except ImportError as e:
    print(f'✗ code.auto_experiment_runner import 실패: {e}')

# 3. utils 모듈들 테스트
utils_modules = [
    ('code.utils.data_utils', 'DataProcessor'),
    ('code.utils.metrics', 'RougeCalculator'),
    ('code.utils.device_utils', 'get_optimal_device'),
    ('code.utils.path_utils', 'PathManager'),
]

for module_name, attr_name in utils_modules:
    try:
        module = __import__(module_name, fromlist=[attr_name])
        if hasattr(module, attr_name):
            print(f'✓ {module_name}.{attr_name} import 성공!')
        else:
            print(f'✗ {module_name}: {attr_name} 속성 없음')
    except ImportError as e:
        print(f'✗ {module_name} import 실패: {e}')

print("\n=== 데이터 파일 컬럼 확인 ===")
import pandas as pd
data_files = ['data/train.csv', 'data/dev.csv', 'data/test.csv']

for file in data_files:
    try:
        df = pd.read_csv(file, nrows=1)
        columns = df.columns.tolist()
        required = ['dialogue', 'summary']
        missing = [col for col in required if col not in columns]
        if missing:
            print(f'✗ {file}: 필수 컬럼 누락 - {missing}')
        else:
            print(f'✓ {file}: 필수 컬럼 확인됨')
    except Exception as e:
        print(f'✗ {file}: 읽기 실패 - {e}')

print("\n=== 완료 ===")
