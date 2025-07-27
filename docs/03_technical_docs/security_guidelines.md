# 🔒 보안 가이드라인

시스템 보안 강화 및 데이터 보호를 위한 종합적인 보안 지침입니다.

## 📋 목차

- [보안 정책](#보안-정책)
- [데이터 보안](#데이터-보안)
- [API 보안](#api-보안)
- [모델 보안](#모델-보안)

## 🛡️ 보안 정책

### 기본 보안 원칙
- **최소 권한 원칙**: 필요한 최소한의 권한만 부여
- **심층 방어**: 다중 계층 보안 체계 구축
- **정기 감사**: 보안 상태 정기 점검 및 업데이트
- **사고 대응**: 보안 사고 시 신속한 대응 체계

### 접근 제어
```python
# Role-based Access Control
class UserRole(Enum):
    ADMIN = "admin"
    RESEARCHER = "researcher"
    VIEWER = "viewer"

class AccessControl:
    def __init__(self):
        self.permissions = {
            UserRole.ADMIN: ["read", "write", "delete", "admin"],
            UserRole.RESEARCHER: ["read", "write"],
            UserRole.VIEWER: ["read"]
        }
    
    def check_permission(self, user_role, action):
        return action in self.permissions.get(user_role, [])
```

## 🔐 데이터 보안

### 개인정보 보호 (PII)
```python
import re
from typing import List

class PIIDetector:
    def __init__(self):
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}-\d{3,4}-\d{4}\b',
            'ssn': r'\b\d{6}-\d{7}\b',
            'card': r'\b\d{4}-\d{4}-\d{4}-\d{4}\b'
        }
    
    def detect_and_mask(self, text: str) -> str:
        for pii_type, pattern in self.patterns.items():
            text = re.sub(pattern, f'[{pii_type.upper()}_MASKED]', text)
        return text
```

### 데이터 암호화
```python
from cryptography.fernet import Fernet
import base64

class DataEncryption:
    def __init__(self, key: bytes = None):
        self.key = key or Fernet.generate_key()
        self.cipher = Fernet(self.key)
    
    def encrypt_text(self, text: str) -> str:
        encrypted = self.cipher.encrypt(text.encode())
        return base64.b64encode(encrypted).decode()
    
    def decrypt_text(self, encrypted_text: str) -> str:
        encrypted = base64.b64decode(encrypted_text.encode())
        decrypted = self.cipher.decrypt(encrypted)
        return decrypted.decode()
```

### 안전한 데이터 저장
```python
import hashlib
import os

def secure_file_storage(data: str, filename: str):
    # 파일 무결성 검증을 위한 해시
    file_hash = hashlib.sha256(data.encode()).hexdigest()
    
    # 안전한 디렉토리에 저장
    secure_dir = "/secure/data/"
    os.makedirs(secure_dir, mode=0o700, exist_ok=True)
    
    filepath = os.path.join(secure_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(data)
    
    # 해시 파일 별도 저장
    with open(f"{filepath}.hash", 'w') as f:
        f.write(file_hash)
```

## 🌐 API 보안

### 인증 및 인가
```python
import jwt
from datetime import datetime, timedelta

class JWTManager:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
    
    def generate_token(self, user_id: str, role: str) -> str:
        payload = {
            'user_id': user_id,
            'role': role,
            'exp': datetime.utcnow() + timedelta(hours=1),
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def verify_token(self, token: str) -> dict:
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            raise Exception("Token has expired")
        except jwt.InvalidTokenError:
            raise Exception("Invalid token")
```

### Rate Limiting
```python
from collections import defaultdict
import time

class RateLimiter:
    def __init__(self, max_requests: int = 100, time_window: int = 3600):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        client_requests = self.requests[client_id]
        
        # 시간 윈도우 밖의 요청 제거
        client_requests[:] = [req_time for req_time in client_requests 
                             if now - req_time < self.time_window]
        
        if len(client_requests) >= self.max_requests:
            return False
        
        client_requests.append(now)
        return True
```

### 입력 검증
```python
import re
from typing import Any

class InputValidator:
    @staticmethod
    def sanitize_input(text: str) -> str:
        # SQL 인젝션 방지
        dangerous_patterns = [
            r"[;\-\-]", r"union\s+select", r"drop\s+table",
            r"<script", r"javascript:", r"on\w+\s*="
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                raise ValueError(f"Potentially dangerous input detected")
        
        return text.strip()
    
    @staticmethod
    def validate_file_upload(filename: str, content: bytes) -> bool:
        allowed_extensions = {'.txt', '.csv', '.json', '.md'}
        max_size = 10 * 1024 * 1024  # 10MB
        
        # 확장자 검증
        ext = os.path.splitext(filename)[1].lower()
        if ext not in allowed_extensions:
            return False
        
        # 파일 크기 검증
        if len(content) > max_size:
            return False
        
        return True
```

## 🤖 모델 보안

### 모델 무결성 검증
```python
import hashlib
import pickle

class ModelSecurity:
    @staticmethod
    def generate_model_hash(model_path: str) -> str:
        """모델 파일의 무결성 해시 생성"""
        hash_md5 = hashlib.md5()
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    @staticmethod
    def verify_model_integrity(model_path: str, expected_hash: str) -> bool:
        """모델 파일 무결성 검증"""
        current_hash = ModelSecurity.generate_model_hash(model_path)
        return current_hash == expected_hash
```

### 적대적 공격 방어
```python
def detect_adversarial_input(text: str) -> bool:
    """적대적 입력 탐지"""
    suspicious_patterns = [
        r"(.)\1{20,}",  # 과도한 문자 반복
        r"[^\w\s가-힣]{10,}",  # 과도한 특수문자
        r"\b(\w+\s+){100,}",  # 과도하게 긴 텍스트
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, text):
            return True
    
    return False
```

## 🔍 보안 모니터링

### 로그 보안
```python
import logging
from datetime import datetime

class SecurityLogger:
    def __init__(self):
        self.logger = logging.getLogger('security')
        handler = logging.FileHandler('/secure/logs/security.log')
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_access(self, user_id: str, resource: str, action: str):
        self.logger.info(f"Access: {user_id} {action} {resource}")
    
    def log_security_event(self, event_type: str, details: str):
        self.logger.warning(f"Security Event: {event_type} - {details}")
```

### 이상 탐지
```python
class AnomalyDetector:
    def __init__(self):
        self.baseline_metrics = {}
    
    def detect_unusual_activity(self, metrics: dict) -> List[str]:
        anomalies = []
        
        for metric, value in metrics.items():
            baseline = self.baseline_metrics.get(metric, value)
            
            # 기준값의 3배 이상 차이나는 경우 이상으로 판단
            if abs(value - baseline) > baseline * 3:
                anomalies.append(f"{metric}: {value} (baseline: {baseline})")
        
        return anomalies
```

## 🔗 관련 문서

- **연계**: [에러 처리](./error_handling.md)
- **연계**: [시스템 아키텍처](./system_architecture.md)
- **심화**: [디버깅 가이드](../06_troubleshooting/debugging_guide.md)

---
📍 **위치**: `docs/03_technical_docs/security_guidelines.md`
