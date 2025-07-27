# 자동 실험 검증 스크립트
import sys
from pathlib import Path
import json

# code 디렉토리를 Python 경로에 추가
sys.path.insert(0, str(Path('code').resolve()))

def test_auto_experiment_system():
    """자동 실험 시스템 검증"""
    print("🧪 자동 실험 시스템 검증 시작")
    
    try:
        # 1. 자동 실험 러너 임포트 테스트
        from code.auto_experiment_runner import AutoExperimentRunner
        print("✅ AutoExperimentRunner 임포트 성공")
        
        # 2. 초기화 테스트 (상대 경로)
        runner = AutoExperimentRunner(
            base_config_path="config/base_config.yaml",
            output_dir="outputs/test_auto_experiments"
        )
        print("✅ AutoExperimentRunner 초기화 성공")
        print(f"   디바이스: {runner.device}")
        
        # 3. 샘플 설정 생성 테스트
        test_config_dir = "config/test_experiments"
        runner.create_sample_configs(test_config_dir)
        print("✅ 샘플 설정 생성 성공")
        
        # 4. 설정 발견 테스트
        configs = runner.discover_experiment_configs(test_config_dir)
        print(f"✅ 실험 설정 발견: {len(configs)}개")
        
        # 5. 설정 로딩 테스트
        if configs:
            config = runner.load_experiment_config(configs[0])
            print("✅ 설정 로딩 및 디바이스 최적화 성공")
            print(f"   최적화된 디바이스: {config['general']['device']}")
        
        # 6. 정리
        import shutil
        from utils.path_utils import PathManager
        
        test_config_path = PathManager.resolve_path(test_config_dir)
        if test_config_path.exists():
            shutil.rmtree(test_config_path)
            print("✅ 테스트 파일 정리 완료")
        
        return True
        
    except ImportError as e:
        print(f"❌ 모듈 임포트 실패: {e}")
        return False
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return False

def test_bash_script():
    """배시 스크립트 검증"""
    print("\n🧪 배시 스크립트 검증")
    
    script_path = Path("run_auto_experiments.sh")
    
    if not script_path.exists():
        print("❌ run_auto_experiments.sh 파일이 없습니다")
        return False
    
    # 실행 권한 확인
    import stat
    file_stat = script_path.stat()
    if file_stat.st_mode & stat.S_IEXEC:
        print("✅ 실행 권한 확인")
    else:
        print("⚠️ 실행 권한이 없습니다 (chmod +x run_auto_experiments.sh 실행)")
    
    # 스크립트 내용 기본 검증
    content = script_path.read_text()
    
    required_elements = [
        "auto_experiment_runner.py",
        "--run-all",
        "config/experiments",
        "outputs/auto_experiments"
    ]
    
    missing_elements = []
    for element in required_elements:
        if element not in content:
            missing_elements.append(element)
    
    if missing_elements:
        print(f"❌ 스크립트에 필수 요소 누락: {missing_elements}")
        return False
    else:
        print("✅ 스크립트 내용 검증 통과")
    
    return True

def main():
    """메인 검증 실행"""
    print("🚀 자동 실험 시스템 통합 검증 시작\n")
    
    tests = [
        ("자동 실험 시스템", test_auto_experiment_system),
        ("배시 스크립트", test_bash_script),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name} 테스트")
        if test_func():
            passed += 1
            print(f"✅ {test_name} 검증 통과")
        else:
            print(f"❌ {test_name} 검증 실패")
    
    print(f"\n📊 검증 결과: {passed}/{total} 통과")
    
    if passed == total:
        print("\n🎉 자동 실험 시스템 검증 완료!")
        print("\n🚀 사용 방법:")
        print("1. 샘플 설정 생성:")
        print("   python code/auto_experiment_runner.py --create-samples")
        print("\n2. 모든 실험 자동 실행:")
        print("   ./run_auto_experiments.sh")
        print("\n3. 특정 실험만 실행:")
        print("   python code/auto_experiment_runner.py --experiment 01_baseline.yaml")
        return 0
    else:
        print("\n❌ 일부 검증 실패 - 문제 해결 필요")
        return 1

if __name__ == "__main__":
    sys.exit(main())
