import pandas as pd
import re

# 지시표현 보완 함수: 직전 발화자 정보로 지시어 대체
def resolve_deictic_with_speaker(dialogue: str) -> str:
    deictic_phrases = ['그 사람', '이 사람', '그거', '이거', '그건', '이건', '거기', '저기', '여기']
    lines = str(dialogue).split('\n')
    resolved = []
    last_speaker = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = re.match(r'^(#Person\d+#):\s*(.*)', line)
        if match:
            speaker = match.group(1)
            utterance = match.group(2)

            for deictic in deictic_phrases:
                if deictic in utterance and last_speaker:
                    utterance = utterance.replace(deictic, f'{last_speaker}가 말한')

            last_speaker = speaker
            resolved.append(f"{speaker}: {utterance}")
        else:
            resolved.append(line)

    return '\n'.join(resolved)

# 텍스트 클린 함수
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    
    ### Sepcial Token 처리 추가
    
    
    # 줄바꿈 표현 통일
    text = text.replace("\\n", "\n").replace("<br>", "\n").replace("</s>", "\n")

    ### 특이 케이스 : train.csv에는 'ㅎㅎ'가 오직 1개 존재한다. 그런데 이것이 #Person2#: ㅎㅎ 라서 빈문자열로 대체하면 말이 없어진다.
    # 문맥과 summary에 맞춰 '나도 행복해.'로 바꾼다.
    text = text.replace("ㅎㅎ", "나도 행복해.")

    # 자소만 있는 단어 제거 (예: ㅋㅋ, ㅇㅋ, ㅜㅜ) > 이모티콘
    text = re.sub(r"\b[ㄱ-ㅎㅏ-ㅣ]{2,}\b", "", text)

    # 중복 줄바꿈 제거
    text = re.sub(r"\n+", r"\n", text)

    # 중복 공백 제거
    text = re.sub(r"[ \t]+", ' ', text)

    return text.strip()

def add_instructions(row:pd.Series) -> pd.Series:
    """지시어 프롬프트 추가.

    :param str dialogue: _description_
    :return str: _description_
    """
    try:
        topic = str(row['topic']).strip()
        dialogue = row['dialogue']
        dialogue = f"#Topic#{topic}#SEP##Dialogue#{dialogue}"
        row['dialogue'] = dialogue
    ##Topic#','#Dialogue#','#Summary#','#SEP#
    except:
        return row
    return row

# 데이터 전처리를 위한 클래스로, 데이터셋을 데이터프레임으로 변환하고 인코더와 디코더의 입력을 생성합니다.
class Preprocess:
    # 클래스 초기화 메서드
    def __init__(self,
            bos_token: str, # 문장의 시작을 알리는 토큰
            eos_token: str, # 문장의 끝을 알리는 토큰
        ) -> None:

        self.bos_token = bos_token # 시작 토큰을 인스턴스 변수에 저장
        self.eos_token = eos_token # 종료 토큰을 인스턴스 변수에 저장

    @staticmethod
    # 실험에 필요한 컬럼을 가져옵니다.
    # 정적 메서드로, 클래스 인스턴스 없이 호출 가능
    def make_set_as_df(file_path, is_train = True, config=None):
        df = pd.read_csv(file_path) # CSV 파일을 읽어 데이터프레임 생성
        # 🔁 발화자 기반 지시표현 보완 전처리 적용
        df['dialogue'] = df['dialogue'].apply(resolve_deictic_with_speaker)
        # 🔁 텍스트 클린 함수
        df['dialogue'] = df['dialogue'].apply(clean_text)

        ### special token에 #Topic# 이 있으면, 지시어 프롬프트에 추가.
        if config is not None and '#Topic#' in config['tokenizer']['special_tokens']:
            df['dialogue'] = df['dialogue'].apply(add_instructions)

        # is_train 플래그가 True이면 학습용 데이터로 처리
        if is_train:
            train_df = df[['fname','dialogue','summary']] # 'fname', 'dialogue', 'summary' 컬럼 선택
            return train_df # 생성된 학습 데이터프레임 반환
        # is_train 플래그가 False이면 테스트용 데이터로 처리
        else:
            test_df = df[['fname','dialogue']] # 'fname', 'dialogue' 컬럼 선택
            return test_df # 생성된 테스트 데이터프레임 반환

    # BART 모델의 입력, 출력 형태를 맞추기 위해 전처리를 진행합니다.
    def make_input(self, dataset,is_test = False):
        # is_test 플래그가 True이면 테스트 데이터셋용 입력 생성
        if is_test:
            encoder_input = dataset['dialogue'] # 인코더 입력으로 'dialogue' 컬럼 사용
            decoder_input = [self.bos_token] * len(dataset['dialogue']) # 디코더 입력은 시작 토큰(bos_token)으로만 구성 -> dialogue 개수만큼 bos_token 생성.
            return encoder_input.tolist(), list(decoder_input) # 인코더 입력과 디코더 입력을 리스트 형태로 반환
        # is_test 플래그가 False이면 학습/검증 데이터셋용 입력 생성
        else:
            encoder_input = dataset['dialogue'] # 인코더 입력으로 'dialogue' 컬럼 사용
            decoder_input = dataset['summary'].apply(lambda x : self.bos_token + str(x)) # 디코더 입력은 'summary' 앞에 시작 토큰(bos_token)을 추가하여 생성
            decoder_output = dataset['summary'].apply(lambda x : str(x) + self.eos_token) # 디코더 출력(레이블)은 'summary' 뒤에 종료 토큰(eos_token)을 추가하여 생성
            return encoder_input.tolist(), decoder_input.tolist(), decoder_output.tolist() # 인코더 입력, 디코더 입력, 디코더 출력을 리스트 형태로 반환
