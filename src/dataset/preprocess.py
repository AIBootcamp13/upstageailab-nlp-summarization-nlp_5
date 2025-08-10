import pandas as pd
import re

# 1. 지시어(대명사 등) 발화자 기반 보완 함수
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


# 2. 텍스트 클린 함수 (이모티콘, 중복 공백 등 정리)
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    
    # 줄바꿈 통일
    text = text.replace("\\n", "\n").replace("<br>", "\n").replace("</s>", "\n")

    # 'ㅎㅎ'는 '나도 행복해.'로 변환 (특이 케이스)
    text = text.replace("ㅎㅎ", "나도 행복해.")

    # 자음 자모만 있는 이모티콘 제거 (예: ㅋㅋ, ㅜㅜ)
    text = re.sub(r"\b[ㄱ-ㅎㅏ-ㅣ]{2,}\b", "", text)

    # 중복 줄바꿈 제거
    text = re.sub(r"\n+", r"\n", text)

    # 중복 공백 제거
    text = re.sub(r"[ \t]+", ' ', text)

    return text.strip()


# 3. 요약문 내 주요 동사 중심 문장 필터링 함수 (전처리용)
def filter_summary_main_verbs(summary: str) -> str:
    main_verbs = ['요청', '제안', '설명', '논의', '추천', '결정', '조언', '안내', '거절', '공유']
    sents = re.split(r'(?<=[.?!])\s+', summary.strip())
    filtered = [s for s in sents if any(verb in s for verb in main_verbs)]
    return ' '.join(filtered) if filtered else summary


# 4. 지시어 프롬프트 추가 (옵션)
def add_instructions(row: pd.Series) -> pd.Series:
    try:
        topic = str(row['topic']).strip()
        dialogue = row['dialogue']
        dialogue = f"#Topic#{topic}#SEP##Dialogue#{dialogue}"
        row['dialogue'] = dialogue
    except:
        pass
    return row


# 5. 전처리 클래스
class Preprocess:
    def __init__(self, bos_token: str, eos_token: str) -> None:
        self.bos_token = bos_token
        self.eos_token = eos_token

    @staticmethod
    def make_set_as_df(file_path, is_train=True, config=None):
        df = pd.read_csv(file_path)

        # 대화문 지시어 보완 및 텍스트 클린
        df['dialogue'] = df['dialogue'].apply(resolve_deictic_with_speaker)
        df['dialogue'] = df['dialogue'].apply(clean_text)

        # 요약문 주요 동사 중심 문장 필터링 (학습용 데이터만)
        if is_train:
            df['summary'] = df['summary'].apply(filter_summary_main_verbs)

        # special token에 #Topic# 포함 시 지시어 프롬프트 추가
        if config is not None and '#Topic#' in config['tokenizer']['special_tokens']:
            df = df.apply(add_instructions, axis=1)

        if is_train:
            return df[['fname', 'dialogue', 'summary']]
        else:
            return df[['fname', 'dialogue']]

    def make_input(self, dataset, is_test=False):
        if is_test:
            encoder_input = dataset['dialogue']
            decoder_input = [self.bos_token] * len(dataset['dialogue'])
            return encoder_input.tolist(), list(decoder_input)
        else:
            encoder_input = dataset['dialogue']
            decoder_input = dataset['summary'].apply(lambda x: self.bos_token + str(x))
            decoder_output = dataset['summary'].apply(lambda x: str(x) + self.eos_token)
            return encoder_input.tolist(), decoder_input.tolist(), decoder_output.tolist()