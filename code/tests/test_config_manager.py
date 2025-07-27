"""
ConfigManager 클래스 단위 테스트

설정 파일 로딩, 병합, 검증 등의 기능이 올바르게 작동하는지 확인합니다.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import yaml
import os
import sys

# 상위 디렉토리를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

from utils.config_manager import ConfigManager, ConfigValidationError


class TestConfigManager(unittest.TestCase):
    """ConfigManager 테스트 클래스"""
    
    def setUp(self):
        """테스트 환경 설정"""
        # 임시 디렉토리 생성
        self.test_dir = Path(tempfile.mkdtemp())
        self.config_dir = self.test_dir / "config"
        self.config_dir.mkdir(parents=True)
        
        # 테스트용 메인 설정 파일 생성
        self.main_config = {
            'meta': {
                'experiment_name': 'test_experiment',
                'version': '1.0',
                'description': 'Test configuration'
            },
            'general': {
                'seed': 42,
                'model_name': 'digit82/kobart-summarization',
                'output_dir': './outputs'
            },
            'model': {
                'architecture': 'kobart',
                'checkpoint': 'digit82/kobart-summarization',
                'load_pretrained': True
            },
            'tokenizer': {
                'encoder_max_len': 512,
                'decoder_max_len': 128
            },
            'training': {
                'per_device_train_batch_size': 16,
                'per_device_eval_batch_size': 16,
                'learning_rate': 5e-5,
                'num_train_epochs': 3,
                'warmup_ratio': 0.1
            },
            'generation': {
                'num_beams': 4,
                'length_penalty': 1.0,
                'no_repeat_ngram_size': 2,
                'max_length': 100
            },
            'wandb': {
                'project': 'nlp-summarization',
                'entity': None,
                'mode': 'online'
            }
        }
        
        # 기존 형식 테스트용 설정
        self.legacy_config = {
            'general': {
                'seed': 42,
                'model_name': 'digit82/kobart-summarization',
                'output_dir': './outputs'
            },
            'tokenizer': {
                'encoder_max_len': 512,
                'decoder_max_len': 128
            },
            'training': {
                'per_device_train_batch_size': 16,
                'learning_rate': 5e-5,
                'num_train_epochs': 3,
                'generation_max_length': 100
            },
            'wandb': {
                'project': 'nlp-summarization'
            },
            'inference': {
                'num_beams': 4,
                'no_repeat_ngram_size': 2,
                'early_stopping': True
            }
        }
        
        # 설정 파일 저장
        self.main_config_path = self.config_dir / "base_config.yaml"
        with open(self.main_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.main_config, f)
        
        self.legacy_config_path = self.config_dir / "legacy_config.yaml"
        with open(self.legacy_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.legacy_config, f)
        
        # 구성 관리자 인스턴스 생성
        self.config_manager = ConfigManager(base_dir=self.test_dir)
    
    def tearDown(self):
        """테스트 환경 정리"""
        shutil.rmtree(self.test_dir)
    
    def test_load_new_config(self):
        """새로운 형식 설정 파일 로딩 테스트"""
        config = self.config_manager.load_config(self.main_config_path)
        
        # 기본 섹션 확인
        self.assertIn('meta', config)
        self.assertIn('general', config)
        self.assertIn('model', config)
        self.assertIn('tokenizer', config)
        self.assertIn('training', config)
        
        # 값 확인
        self.assertEqual(config['general']['seed'], 42)
        self.assertEqual(config['tokenizer']['encoder_max_len'], 512)
        self.assertEqual(config['training']['learning_rate'], 5e-5)
    
    def test_load_legacy_config(self):
        """기존 형식 설정 파일 로딩 및 마이그레이션 테스트"""
        config = self.config_manager.load_config(self.legacy_config_path)
        
        # 마이그레이션 확인
        self.assertTrue(self.config_manager.is_legacy)
        self.assertIn('meta', config)
        self.assertIn('model', config)
        self.assertIn('generation', config)
        
        # 값 확인
        self.assertEqual(config['general']['seed'], 42)
        self.assertEqual(config['generation']['max_length'], 100)
        self.assertEqual(config['generation']['num_beams'], 4)
    
    def test_merge_sweep_params(self):
        """WandB Sweep 파라미터 병합 테스트"""
        # 먼저 기본 설정 로드
        config = self.config_manager.load_config(self.main_config_path)
        
        # Sweep 파라미터 정의
        sweep_params = {
            'learning_rate': 1e-4,
            'per_device_train_batch_size': 32,
            'num_beams': 6,
            'warmup_ratio': 0.2
        }
        
        # 파라미터 병합
        merged_config = self.config_manager.merge_sweep_params(sweep_params)
        
        # 병합 확인
        self.assertEqual(merged_config['training']['learning_rate'], 1e-4)
        self.assertEqual(merged_config['training']['per_device_train_batch_size'], 32)
        self.assertEqual(merged_config['generation']['num_beams'], 6)
        self.assertEqual(merged_config['training']['warmup_ratio'], 0.2)
    
    def test_env_override(self):
        """환경변수 오버라이드 테스트"""
        # 환경변수 설정
        os.environ['LEARNING_RATE'] = '1e-3'
        os.environ['BATCH_SIZE'] = '8'
        os.environ['WANDB_PROJECT'] = 'test-project'
        
        try:
            # 설정 로딩
            config = self.config_manager.load_config(self.main_config_path)
            
            # 오버라이드 확인
            self.assertEqual(config['training']['learning_rate'], 1e-3)
            self.assertEqual(config['training']['per_device_train_batch_size'], 8)
            self.assertEqual(config['wandb']['project'], 'test-project')
        
        finally:
            # 환경변수 정리
            del os.environ['LEARNING_RATE']
            del os.environ['BATCH_SIZE']
            del os.environ['WANDB_PROJECT']
    
    def test_validation_error(self):
        """설정 검증 오류 테스트"""
        # 잘못된 설정 생성
        invalid_config = {
            'meta': {'experiment_name': 'invalid'},
            'training': {
                'learning_rate': 2.0,  # 범위 초과
                'per_device_train_batch_size': -1,  # 음수
                'num_train_epochs': 0  # 0
            }
        }
        
        invalid_config_path = self.config_dir / "invalid_config.yaml"
        with open(invalid_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(invalid_config, f)
        
        # 검증 오류 발생 확인
        with self.assertRaises(ConfigValidationError):
            self.config_manager.load_config(invalid_config_path)
    
    def test_save_config(self):
        """설정 저장 테스트"""
        # 설정 로딩
        config = self.config_manager.load_config(self.main_config_path)
        
        # 일부 수정
        config['training']['learning_rate'] = 1e-3
        config['general']['seed'] = 123
        
        # 저장
        save_path = self.config_dir / "saved_config.yaml"
        self.config_manager.save_config(config, save_path)
        
        # 저장된 파일 확인
        self.assertTrue(save_path.exists())
        
        # 다시 로딩하여 확인
        new_manager = ConfigManager(base_dir=self.test_dir)
        loaded_config = new_manager.load_config(save_path)
        
        self.assertEqual(loaded_config['training']['learning_rate'], 1e-3)
        self.assertEqual(loaded_config['general']['seed'], 123)
        self.assertIn('saved_at', loaded_config['meta'])
    
    def test_constraints_application(self):
        """메모리 최적화 제약 적용 테스트"""
        # 긴 시퀀스 설정
        config = self.config_manager.load_config(self.main_config_path)
        
        sweep_params = {
            'encoder_max_len': 1536,  # 매우 긴 시퀀스
            'per_device_train_batch_size': 32
        }
        
        merged_config = self.config_manager.merge_sweep_params(sweep_params)
        
        # 배치 크기가 자동으로 감소했는지 확인
        self.assertEqual(merged_config['training']['per_device_train_batch_size'], 8)
        self.assertGreaterEqual(merged_config['training'].get('gradient_accumulation_steps', 1), 2)
    
    def test_default_values(self):
        """기본값 적용 테스트"""
        # 최소 설정만 포함하는 파일 생성
        minimal_config = {
            'meta': {'experiment_name': 'minimal'},
            'general': {'model_name': 'test'},
            'model': {'architecture': 'kobart'},
            'tokenizer': {'encoder_max_len': 512},
            'training': {'learning_rate': 5e-5}
        }
        
        minimal_config_path = self.config_dir / "minimal_config.yaml"
        with open(minimal_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(minimal_config, f)
        
        # 로딩 (검증 비활성화)
        manager = ConfigManager(base_dir=self.test_dir, validate=False)
        config = manager.load_config(minimal_config_path)
        
        # 기본값 확인
        self.assertEqual(config['general']['seed'], 42)
        self.assertEqual(config['general']['device'], 'auto')
        self.assertEqual(config['training']['fp16'], True)
        self.assertEqual(config['generation']['early_stopping'], True)
        self.assertEqual(config['wandb']['save_code'], True)
    
    def test_get_available_configs(self):
        """사용 가능한 설정 목록 조회 테스트"""
        # 모델 설정 디렉토리 생성
        models_dir = self.config_dir / "models"
        models_dir.mkdir()
        
        # 테스트 모델 설정 파일 생성
        for model_name in ['kobart', 'kogpt2', 't5']:
            model_config = {'model': {'name': model_name}}
            with open(models_dir / f"{model_name}.yaml", 'w') as f:
                yaml.dump(model_config, f)
        
        # 목록 조회
        model_configs = self.config_manager.get_model_configs()
        
        self.assertEqual(len(model_configs), 3)
        self.assertIn('kobart', model_configs)
        self.assertIn('kogpt2', model_configs)
        self.assertIn('t5', model_configs)


if __name__ == '__main__':
    unittest.main()
