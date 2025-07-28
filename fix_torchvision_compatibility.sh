#!/bin/bash

echo "=== PyTorch/torchvision 호환성 문제 해결 ==="

# 1. 환경 활성화
eval "$(/opt/conda/bin/conda shell.bash hook)"
conda activate venv

# 2. 현재 버전 확인 (가능한만큼)
echo "2. 현재 PyTorch 버전 확인..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || echo "PyTorch 버전 확인 불가"

# 3. 호환되지 않는 torchvision 제거
echo "3. 기존 torchvision 제거..."
uv pip uninstall torchvision -q || true

# 4. PyTorch 2.7.1과 호환되는 torchvision 설치
echo "4. 호환되는 torchvision 설치..."
# PyTorch 2.7.1+cu126과 호환되는 torchvision 0.22.1+cu126 설치
uv pip install torchvision==0.22.1+cu126 --index-url https://download.pytorch.org/whl/cu126

# 5. 설치 확인
echo "5. torchvision 설치 확인..."
python -c "
try:
    import torch
    import torchvision
    print(f'✅ PyTorch: {torch.__version__}')
    print(f'✅ torchvision: {torchvision.__version__}')
    print('✅ 호환성 문제 해결')
except Exception as e:
    print(f'❌ 여전히 문제: {e}')
    exit(1)
"

# 6. 호환 안 되면 PyTorch도 함께 다운그레이드
if [ $? -ne 0 ]; then
    echo "6. PyTorch와 torchvision 모두 안정 버전으로 재설치..."
    uv pip uninstall torch torchvision torchaudio -q || true
    
    # PyTorch 2.4.1 (더 안정적) + 해당 torchvision 설치
    uv pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121
fi

# 7. 최종 확인
echo "7. 최종 확인..."
python -c "
print('=== 호환성 테스트 ===')

try:
    import torch
    import torchvision
    print(f'✅ PyTorch: {torch.__version__}')
    print(f'✅ torchvision: {torchvision.__version__}')
    
    # GPU 테스트
    if torch.cuda.is_available():
        print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
        x = torch.randn(10, 10).cuda()
        print('✅ GPU 연산 성공')
    
    # torchvision 연산 테스트
    from torchvision.ops import nms
    import torch
    boxes = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=torch.float32)
    scores = torch.tensor([0.9, 0.8], dtype=torch.float32)
    if torch.cuda.is_available():
        boxes = boxes.cuda()
        scores = scores.cuda()
    result = nms(boxes, scores, 0.5)
    print('✅ torchvision NMS 연산 성공')
    
except Exception as e:
    print(f'❌ 오류: {e}')
    exit(1)
"

# 8. unsloth 재테스트
echo "8. unsloth 재테스트..."
python -c "
try:
    from unsloth import FastLanguageModel
    print('🎉 unsloth 성공!')
    print('✅ 모든 문제 해결됨')
except Exception as e:
    print(f'❌ unsloth 여전히 실패: {e}')
    print('하지만 PyTorch/torchvision은 해결됨')
"

echo ""
echo "=== 호환성 문제 해결 완료 ==="
echo ""
echo "최종 확인:"
echo "python -c \"from unsloth import FastLanguageModel; print('✅ 완료')\""
