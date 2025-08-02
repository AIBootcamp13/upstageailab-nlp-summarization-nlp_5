#!/usr/bin/env python3
"""
auto_experiment_runner.py 구문 검사
"""

import sys
import ast

try:
    with open('/Users/jayden/Developer/Projects/upstageailab-nlp-summarization-nlp_5/nlp-sum-lyj/code/auto_experiment_runner.py', 'r') as f:
        code = f.read()
    
    # 구문 분석
    ast.parse(code)
    print("✅ auto_experiment_runner.py 구문 검사 통과!")
    
except SyntaxError as e:
    print(f"❌ 구문 오류 발생:")
    print(f"   파일: {e.filename}")
    print(f"   줄: {e.lineno}")
    print(f"   오류: {e.msg}")
    print(f"   위치: {e.text}")
    sys.exit(1)
    
except Exception as e:
    print(f"❌ 예외 발생: {e}")
    sys.exit(1)
