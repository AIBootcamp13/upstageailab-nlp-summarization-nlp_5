#!/bin/bash

# NLP 요약 프로젝트 개발 세션 시작 스크립트
PROJECT_DIR="/Users/jayden/Developer/Projects/upstageailab-nlp-summarization-nlp_5/nlp-sum-lyj"
SESSION_NAME="nlp-sum"

# 세션이 이미 존재하는지 확인
tmux has-session -t $SESSION_NAME 2>/dev/null

if [ $? != 0 ]; then
    # 새 세션 생성
    tmux new-session -s $SESSION_NAME -n editor -d -c $PROJECT_DIR
    
    # 윈도우 1: 편집기 (기본)
    tmux send-keys -t $SESSION_NAME:editor "cd $PROJECT_DIR" C-m
    tmux send-keys -t $SESSION_NAME:editor "source .venv/bin/activate" C-m
    tmux send-keys -t $SESSION_NAME:editor "clear" C-m
    
    # 윈도우 2: Jupyter
    tmux new-window -t $SESSION_NAME -n jupyter -c $PROJECT_DIR
    tmux send-keys -t $SESSION_NAME:jupyter "cd $PROJECT_DIR" C-m
    tmux send-keys -t $SESSION_NAME:jupyter "echo 'UV 환경에서 Jupyter 실행'" C-m
    tmux send-keys -t $SESSION_NAME:jupyter "jupyter notebook --no-browser --port=8888" C-m
    
    # 윈도우 3: 터미널
    tmux new-window -t $SESSION_NAME -n terminal -c $PROJECT_DIR
    tmux send-keys -t $SESSION_NAME:terminal "cd $PROJECT_DIR" C-m
    tmux send-keys -t $SESSION_NAME:terminal "source .venv/bin/activate" C-m
    tmux send-keys -t $SESSION_NAME:terminal "clear" C-m
    
    # 윈도우 4: Git
    tmux new-window -t $SESSION_NAME -n git -c $PROJECT_DIR
    tmux send-keys -t $SESSION_NAME:git "cd $PROJECT_DIR" C-m
    tmux send-keys -t $SESSION_NAME:git "git status" C-m
    
    # 첫 번째 윈도우 선택
    tmux select-window -t $SESSION_NAME:editor
fi

# 세션에 연결
tmux attach-session -t $SESSION_NAME
