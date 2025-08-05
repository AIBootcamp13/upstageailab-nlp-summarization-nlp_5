#!/usr/bin/env python3
"""
Monkey Patch 테스트 스크립트
trainer.py의 monkey patch가 정상적으로 적용되는지 검증
"""
import sys
import os

# 프로젝트 루트를 Python path에 추가
sys.path.append(os.path.dirname(__file__))

def test_monkey_patch():
    """Monkey Patch 적용 및 기본 동작 테스트"""
    print("🔍 Monkey Patch 테스트 시작...")
    
    try:
        # 1. 기본 import 테스트
        print("📦 transformers import 테스트...")
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
        print("✅ transformers import 성공")
        
        # 2. 원본 save_pretrained 메서드 백업 확인
        print("🔧 원본 메서드 확인...")
        original_method = PreTrainedTokenizerBase.save_pretrained
        print(f"✅ 원본 메서드: {original_method}")
        
        # 3. trainer.py import (monkey patch 적용)
        print("🐒 Monkey Patch 적용 중...")
        from code.trainer import SafeSeq2SeqTrainer
        print("✅ trainer.py import 성공 (Monkey Patch 적용됨)")
        
        # 4. 패치된 메서드 확인
        patched_method = PreTrainedTokenizerBase.save_pretrained
        print(f"🔥 패치된 메서드: {patched_method}")
        
        # 5. 메서드가 실제로 변경되었는지 확인
        if patched_method != original_method:
            print("✅ Monkey Patch 성공적으로 적용됨!")
        else:
            print("❌ Monkey Patch 적용 실패")
            return False
            
        # 6. SafeSeq2SeqTrainer 클래스 확인
        print("🏗️ SafeSeq2SeqTrainer 클래스 확인...")
        trainer_class = SafeSeq2SeqTrainer
        print(f"✅ SafeSeq2SeqTrainer: {trainer_class}")
        
        return True
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_monkey_patch()
    if success:
        print("🎉 모든 테스트 통과!")
        sys.exit(0)
    else:
        print("💥 테스트 실패!")
        sys.exit(1)
