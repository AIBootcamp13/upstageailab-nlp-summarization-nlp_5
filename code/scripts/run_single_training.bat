@echo off
REM 단일 학습 실행 스크립트 (Windows)

echo Starting single training run with base configuration...

cd ..
python trainer.py ^
    --config config\base_config.yaml ^
    --train-data ..\data\train.csv ^
    --val-data ..\data\dev.csv ^
    --test-data ..\data\test.csv

echo Training completed!
pause
