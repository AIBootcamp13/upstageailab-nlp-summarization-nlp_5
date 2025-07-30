#!/usr/bin/env python3
"""
실험 진행 상황 모니터링 스크립트
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
import subprocess

# 프로젝트 루트 경로 설정 (scripts/utils/에서 2단계 상위)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "code"))

def get_gpu_status():
    """GPU 상태 확인"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,temperature.gpu,utilization.gpu", 
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                parts = line.split(', ')
                if len(parts) >= 5:
                    name = parts[0]
                    mem_used = float(parts[1])
                    mem_total = float(parts[2])
                    temp = float(parts[3])
                    util = float(parts[4])
                    
                    mem_percent = (mem_used / mem_total) * 100
                    print(f"🖥️  GPU: {name}")
                    print(f"   메모리: {mem_used:.0f}/{mem_total:.0f} MB ({mem_percent:.1f}%)")
                    print(f"   온도: {temp}°C, 사용률: {util}%")
    except:
        print("❌ GPU 상태를 확인할 수 없습니다.")

def check_experiments_status():
    """실험 상태 확인"""
    experiments_dir = project_root / "outputs" / "auto_experiments" / "experiments"
    
    if not experiments_dir.exists():
        print("📁 실험 디렉토리가 없습니다.")
        return
    
    print("\n📊 실험 상태:")
    print("=" * 50)
    
    # 실험 디렉토리 목록
    exp_dirs = sorted([d for d in experiments_dir.iterdir() if d.is_dir()])
    
    for exp_dir in exp_dirs:
        info_file = exp_dir / "experiment_info.json"
        if info_file.exists():
            with open(info_file, 'r', encoding='utf-8') as f:
                info = json.load(f)
            
            status = info.get('status', 'unknown')
            name = info.get('name', 'unknown')
            start_time = info.get('start_time', '')
            
            status_icon = {
                '실행중': '🔄',
                'completed': '✅',
                '실패': '❌'
            }.get(status, '❓')
            
            print(f"\n{status_icon} {name}")
            print(f"   ID: {exp_dir.name}")
            print(f"   상태: {status}")
            print(f"   시작: {start_time}")
            
            # 최신 체크포인트 확인
            checkpoints = list(exp_dir.glob("checkpoint-*"))
            if checkpoints:
                latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
                print(f"   최신 체크포인트: {latest.name}")

def check_wandb_runs():
    """WandB 실행 확인"""
    wandb_dir = project_root / "wandb"
    if not wandb_dir.exists():
        return
    
    print("\n🌐 WandB 실행:")
    print("=" * 50)
    
    runs = list(wandb_dir.glob("run-*"))
    if runs:
        latest_runs = sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)[:5]
        for run in latest_runs:
            run_name = run.name
            modified = datetime.fromtimestamp(run.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            print(f"  - {run_name} (수정: {modified})")
    else:
        print("  실행 중인 WandB run이 없습니다.")

def monitor_loop():
    """모니터링 루프"""
    print("🔍 실험 모니터링 시작 (Ctrl+C로 종료)")
    print("=" * 60)
    
    try:
        while True:
            os.system('clear' if os.name == 'posix' else 'cls')
            
            print(f"🕐 현재 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 60)
            
            get_gpu_status()
            check_experiments_status()
            check_wandb_runs()
            
            print("\n⏱️  30초 후 업데이트... (Ctrl+C로 종료)")
            time.sleep(30)
    
    except KeyboardInterrupt:
        print("\n\n👋 모니터링 종료")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        # 한 번만 실행
        get_gpu_status()
        check_experiments_status()
        check_wandb_runs()
    else:
        # 계속 모니터링
        monitor_loop()
