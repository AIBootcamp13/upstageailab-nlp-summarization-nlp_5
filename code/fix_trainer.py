with open('trainer.py', 'r') as f:
    content = f.read()

# 메서드 추가할 위치 찾기
insert_pos = content.find('    def _load_model(self) -> None:')

if insert_pos != -1:
    # 새 메서드 정의
    new_method = '''    def _get_architecture_from_model_name(self) -> str:
        """모델 이름으로부터 아키텍처 추론"""
        model_name = self.config.get('general', {}).get('model_name', '').lower()
        
        if 'bart' in model_name or 'kobart' in model_name:
            return 'bart'
        elif 'mt5' in model_name:
            return 'mt5'
        elif 't5' in model_name:
            return 't5'
        elif 'gpt' in model_name:
            return 'gpt2'
        else:
            return 'bart'  # 기본값
    
'''
    # 메서드 삽입
    content = content[:insert_pos] + new_method + content[insert_pos:]
    
    # 파일 저장
    with open('trainer.py', 'w') as f:
        f.write(content)
    print('Successfully added _get_architecture_from_model_name method')
else:
    print('Could not find insertion point')
