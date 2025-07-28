import sys
sys.path.insert(0, '.')

try:
    from code.auto_experiment_runner import AutoExperimentRunner
    print("✓ AutoExperimentRunner import 성공!")
    
    # 인스턴스 생성 테스트
    runner = AutoExperimentRunner()
    print("✓ AutoExperimentRunner 인스턴스 생성 성공!")
    
    # 주요 메서드 확인
    methods = ['discover_experiment_configs', 'run_single_experiment', 'run_all_experiments']
    for method in methods:
        if hasattr(runner, method):
            print(f"✓ {method} 메서드 존재")
        else:
            print(f"✗ {method} 메서드 없음")
            
except Exception as e:
    print(f"✗ 오류 발생: {e}")
    import traceback
    traceback.print_exc()
