#!/usr/bin/env python3
"""
auto_experiment_runner.py에 추론 기능을 추가하는 패치 스크립트
"""

import re

def patch_auto_experiment_runner():
    # 파일 읽기
    with open('/data/ephemeral/home/upstageailab-nlp-summarization-nlp_5/code/auto_experiment_runner.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # 삽입할 추론 코드
    inference_code = '''                
                # 🆕 학습 완료 후 test.csv 추론 수행
                print(f"\\n📊 Test 추론 시작: {experiment_id}")
                
                try:
                    # 베스트 체크포인트 찾기
                    output_dir = Path(config.get('training', {}).get('output_dir', 'outputs'))
                    checkpoint_dirs = list(output_dir.glob('checkpoint-*'))
                    
                    if checkpoint_dirs:
                        # 가장 최근 체크포인트 선택
                        best_checkpoint = max(checkpoint_dirs, key=lambda p: p.stat().st_mtime)
                        print(f"🎯 베스트 체크포인트: {best_checkpoint}")
                        
                        # post_training_inference 활용
                        try:
                            from post_training_inference import generate_submission_after_training
                            
                            submission_path = generate_submission_after_training(
                                experiment_name=experiment_id,
                                model_path=str(best_checkpoint),
                                config_dict=config
                            )
                            
                            print(f"✅ 제출 파일 생성 완료: {submission_path}")
                            result = self._collect_results(config, Path(config_path).stem)
                            result['submission_path'] = submission_path
                            
                        except ImportError as ie:
                            print(f"⚠️ post_training_inference import 실패: {ie}")
                            # 대안: run_inference.py 직접 사용
                            try:
                                inference_cmd = [
                                    sys.executable,
                                    str(path_manager.resolve_path("code/run_inference.py")),
                                    "--model_path", str(best_checkpoint),
                                    "--input_file", "data/test.csv",
                                    "--output_file", f"outputs/auto_experiments/{experiment_id}_submission.csv",
                                    "--batch_size", "16"
                                ]
                                
                                print(f"🔄 대안 추론 실행: {' '.join(inference_cmd)}")
                                
                                inference_process = subprocess.run(
                                    inference_cmd,
                                    capture_output=True,
                                    text=True,
                                    env=env
                                )
                                
                                if inference_process.returncode == 0:
                                    submission_path = f"outputs/auto_experiments/{experiment_id}_submission.csv"
                                    print(f"✅ 대안 추론 성공: {submission_path}")
                                    result = self._collect_results(config, Path(config_path).stem)
                                    result['submission_path'] = submission_path
                                else:
                                    print(f"❌ 대안 추론 실패: {inference_process.stderr}")
                                    result = self._collect_results(config, Path(config_path).stem)
                                    result['inference_error'] = inference_process.stderr
                                    
                            except Exception as alt_e:
                                print(f"❌ 대안 추론 예외: {alt_e}")
                                result = self._collect_results(config, Path(config_path).stem)
                                result['inference_error'] = str(alt_e)
                                
                    else:
                        print("⚠️ 체크포인트를 찾을 수 없습니다.")
                        result = self._collect_results(config, Path(config_path).stem)
                        result['inference_error'] = "No checkpoint found"
                        
                except Exception as inf_e:
                    print(f"❌ 추론 실행 중 예외: {inf_e}")
                    result = self._collect_results(config, Path(config_path).stem)
                    result['inference_error'] = str(inf_e)'''

    # 타겟 라인 찾기 및 교체
    original_line = "                result = self._collect_results(config, Path(config_path).stem)"

    if original_line in content:
        # 원본 라인을 추론 코드로 교체 (추론 코드 내에 이미 result 할당 포함)
        new_content = content.replace(
            original_line,
            inference_code.rstrip()
        )
        
        # 파일 쓰기
        with open('/data/ephemeral/home/upstageailab-nlp-summarization-nlp_5/code/auto_experiment_runner.py', 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("✅ auto_experiment_runner.py 수정 완료!")
        print("추가된 기능: 학습 완료 후 자동 test.csv 추론")
        return True
    else:
        print("❌ 타겟 라인을 찾을 수 없습니다.")
        print("원본 라인:", repr(original_line))
        return False

if __name__ == "__main__":
    patch_auto_experiment_runner()
