# AIStages 서버 GPU 메모리 정리 가이드

## 🎯 개요

AIStages 서버에서 GPU 메모리가 과도하게 사용될 때 (15GB+ 사용) 실험 성능을 저하시킬 수 있습니다. 이 가이드는 서버에서 직접 GPU 메모리를 정리하는 방법을 제공합니다.

## 📊 상황별 대응 방법

### 1️⃣ **현재 GPU 상태 확인**

```bash
# 기본 GPU 정보
nvidia-smi

# 간단한 메모리 사용량 확인
nvidia-smi --query-gpu=memory.used,memory.total,temperature.gpu,utilization.gpu --format=csv,noheader,nounits
```

**해석:**
- `15098, 24576, 30, 0` → 15GB/24GB 사용, 30°C, 0% 활용률

### 2️⃣ **실행 중인 Python 프로세스 확인**

```bash
# GPU 사용 프로세스 확인
ps aux | grep python | head -10

# 특정 프로세스 상세 정보
ps -p [PID] -o pid,ppid,cmd,etime,pcpu,pmem
```

**일반적인 원인:**
- 이전 실험이 완료되지 않고 계속 실행 중
- Jupyter 노트북이나 Python 스크립트가 백그라운드에서 실행
- GPU 캐시가 정리되지 않음

### 3️⃣ **단계별 정리 방법**

#### **Step 1: PyTorch 캐시 정리 (가장 안전)**

```bash
python3 -c "
import torch
import gc

if torch.cuda.is_available():
    # CUDA 캐시 정리
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print('✅ CUDA 캐시 정리 완료')
    
    # 메모리 현황 출력
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        cached = torch.cuda.memory_reserved(i) / 1024**3
        print(f'GPU {i}: 할당 {allocated:.1f}GB, 캐시 {cached:.1f}GB')
else:
    print('CUDA 사용 불가')

# 가비지 컬렉션
gc.collect()
print('✅ 가비지 컬렉션 완료')
"
```

#### **Step 2: 특정 프로세스 종료 (주의 필요)**

```bash
# 특정 PID 프로세스 정상 종료
kill -TERM [PID]

# 5초 대기 후 강제 종료 (필요시)
sleep 5
kill -KILL [PID]
```

**⚠️ 주의사항:**
- 실험 중인 프로세스를 종료하면 **진행 중인 실험 결과가 손실**됩니다
- 중요한 실험이 진행 중이라면 완료될 때까지 대기하는 것을 권장

#### **Step 3: 시스템 캐시 정리 (선택사항)**

```bash
# 시스템 메모리 캐시 정리 (root 권한 필요)
sync
echo 3 > /proc/sys/vm/drop_caches  # 권한이 있을 경우만

# 임시 파일 정리
rm -rf /tmp/torch_* /tmp/transformers_* /tmp/huggingface_* 2>/dev/null || true
```

### 4️⃣ **정리 효과 확인**

```bash
# 정리 후 상태 재확인
nvidia-smi --query-gpu=memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits

# 실행 중인 Python 프로세스 재확인
ps aux | grep python | grep -v grep || echo "Python 프로세스 없음"
```

## 🎯 상황별 예시

### **예시 1: 캐시만 정리 (실험 보존)**

```bash
# 현재 상태 확인
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
# 출력: 15098, 24576

# PyTorch 캐시 정리
python3 -c "import torch; torch.cuda.empty_cache(); print('정리 완료')"

# 결과 확인
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
# 기대 출력: 12000, 24576 (약간 감소)
```

### **예시 2: 프로세스 종료 후 전체 정리**

```bash
# 실행 중인 Python 프로세스 찾기
ps aux | grep python
# 출력: root 22272 ... /opt/conda/bin/python3 trainer.py

# 프로세스 종료 (실험 결과 손실 주의!)
kill -TERM 22272
sleep 5
kill -KILL 22272  # 필요시

# GPU 캐시 정리
python3 -c "import torch; torch.cuda.empty_cache(); print('정리 완료')"

# 결과 확인
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
# 기대 출력: 3, 24576 (극적 개선)
```

## 🚀 성능 최적화 팁

### **1. 정리 전후 비교**

| 상황 | 메모리 사용량 | 사용률 | 가용 메모리 | 배치 크기 |
|------|---------------|--------|-------------|-----------|
| **정리 전** | 15GB | 62% | 9.5GB | 제한적 |
| **정리 후** | 3MB | 0.01% | 24.5GB | 최대 |

### **2. 실험 최적화 권장사항**

**Before 실험 시작:**
```bash
# 1. GPU 상태 확인
nvidia-smi

# 2. 캐시 정리 (안전)
python3 -c "import torch; torch.cuda.empty_cache()"

# 3. 5GB 미만 확인
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
```

**During 실험 중:**
- GPU 메모리 모니터링: `watch -n 30 nvidia-smi`
- 과도한 사용 시 (20GB+) 배치 크기 조정 고려

**After 실험 완료:**
```bash
# 자동 정리 (다음 실험 준비)
python3 -c "import torch; torch.cuda.empty_cache(); print('다음 실험 준비 완료')"
```

## ⚠️ 주의사항

### **DO NOT 절대 하지 말 것:**
1. **실험 진행 중 프로세스 강제 종료** (결과 손실)
2. **nvidia-smi reset** (시스템 불안정 위험)
3. **GPU 드라이버 재시작** (서버 재부팅 필요할 수 있음)

### **DO 권장 방법:**
1. **먼저 캐시 정리 시도**
2. **프로세스 종료는 신중하게**
3. **실험 완료 대기가 가장 안전**

## 🎯 RTX 3090 최적 활용 기준

| GPU 메모리 사용량 | 상태 | 권장 조치 |
|-------------------|------|-----------|
| **< 5GB** | ✅ 이상적 | 최대 배치 크기 사용 가능 |
| **5-17GB** | 🟡 보통 | 정상 실험 진행 |
| **17-20GB** | 🟠 주의 | 배치 크기 조정 고려 |
| **20-22.5GB** | ⚠️ 경고 | 캐시 정리 권장 |
| **> 22.5GB** | 🚨 위험 | 즉시 정리 필요 |

---

**📝 작성일**: 2025-08-03  
**📅 최종 업데이트**: 2025-08-03  
**🔄 검증 상태**: AIStages 서버에서 15GB→3MB 정리 성공 확인  
