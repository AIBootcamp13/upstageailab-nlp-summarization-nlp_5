import pandas as pd
import re

# 지시표현 보완 함수: 등장인물/실명 기반 우선 해소, 불가능할 때만 자연어 그대로 두기
def resolve_deictic_with_real_entity(dialogue: str) -> str:
    """
    - "그 사람", "이 사람" 등 대명사는 직전(혹은 직후) 언급된 인물/사물 엔티티가 명확할 때만 치환
    - 명확히 찾을 수 없으면 자연어 그대로 둠 (추상적 의미 토큰/상대방 본인 치환 금지)
    """
    masking_pattern = r"(#\w+#|[A-Z][a-z]+(?:\s[A-Z][a-z]+)*)"
    first_person = ["나", "저", "내", "저의", "본인", "난", "저희"]
    second_person = ["너", "네", "당신", "너네", "너희", "자기", "여보", "자기야", "넌"]
    deictic_pronouns = ["그 사람", "이 사람", "저 사람"]
    lines = str(dialogue).split('\n')
    resolved = []

    # 최근 등장한 엔티티(사람/사물/마스킹/실명) 추적용
    entity_memory = []

    # 대화 참여자 추출 (#PersonN# 기준)
    participants = []
    for line in lines:
        match = re.match(r'^(#Person\d+#):', line)
        if match:
            speaker = match.group(1)
            if speaker not in participants:
                participants.append(speaker)

    # 이/그/저 + 명사 → 명사로만 치환하는 패턴
    deictic_noun_pattern = re.compile(r'\b([이그저])\s?([가-힣]+)\b')

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        match = re.match(r'^(#Person\d+#):\s*(.*)', line)
        if match:
            speaker = match.group(1)
            utterance = match.group(2)

            # 최근 등장 실명/마스킹토큰 추적 (인물/사물/명사 등)
            entities = re.findall(masking_pattern, utterance)
            if entities:
                entity_memory.extend(entities)

            # 대명사/지시어 치환 (1, 2인칭)
            for fp in first_person:
                utterance = re.sub(rf"\b{fp}\b", speaker, utterance)
            for sp in second_person:
                # participants에서 speaker가 아닌 사람 중 첫 번째를 you로 가정
                if len(participants) >= 2:
                    you = [p for p in participants if p != speaker][0]
                    utterance = re.sub(rf"\b{sp}\b", you, utterance)

            # 대명사(그 사람/이 사람/저 사람) 치환: 직전 등장 엔티티가 있을 때만
            def deictic_repl(m):
                # 직전(혹은 직후) 발화에 엔티티가 있으면 그것으로 치환
                if entity_memory:
                    return entity_memory[-1]
                # 없으면 자연어 그대로 둠 (추상 토큰 변환 X)
                return m.group(0)
            for pronoun in deictic_pronouns:
                utterance = re.sub(re.escape(pronoun), deictic_repl, utterance)

            # “이/그/저 + 명사” 패턴 → “명사”만 남김
            utterance = deictic_noun_pattern.sub(r'\2', utterance)

            resolved.append(f"{speaker}: {utterance}")
        else:
            resolved.append(line)

    return '\n'.join(resolved)

# 텍스트 클린 함수 (원래 주석, 기능 그대로)
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # 줄바꿈 표현 통일
    text = text.replace("\\n", "\n").replace("<br>", "\n").replace("</s>", "\n")
    # 자소만 있는 단어 제거 (예: ㅋㅋ, ㅇㅋ, ㅜㅜ)
    text = re.sub(r"\b[ㄱ-ㅎㅏ-ㅣ]{2,}\b", "", text)
    # 자소 반복 제거 (예: ㅋㅋㅋㅋ)
    text = re.sub(r"[ㄱ-ㅎㅏ-ㅣ]{3,}", "", text)
    # 중복 줄바꿈 제거
    text = re.sub(r"\n+", "\n", text)
    return text.strip()

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
    def make_set_as_df(file_path, is_train=True):
        df = pd.read_csv(file_path)
        # 등장인물/실명 기반 지시표현 해소 전처리 적용 (불가시 의미 토큰 유지)
        df['dialogue'] = df['dialogue'].apply(resolve_deictic_with_real_entity)
        # 추가로 클린 전처리
        df['dialogue'] = df['dialogue'].apply(clean_text)
        # is_train 플래그가 True이면 학습용 데이터로 처리
        if is_train:
            df = df[['fname','dialogue','summary']]
        else:
            df = df[['fname','dialogue']]
        return df

    # BART 모델의 입력, 출력 형태를 맞추기 위해 전처리를 진행합니다.
    def make_input(self, dataset, is_test=False):
        # is_test 플래그가 True이면 테스트 데이터셋용 입력 생성
        if is_test:
            encoder_input = dataset['dialogue'].apply(clean_text)
            decoder_input = [self.bos_token] * len(dataset)
            return encoder_input.tolist(), decoder_input
        else:
            encoder_input = dataset['dialogue'].apply(clean_text)
            summary_cleaned = dataset['summary'].apply(clean_text)
            # 전처리된 summary에 bos/eos 붙이기
            decoder_input = summary_cleaned.apply(lambda x: self.bos_token + str(x))
            decoder_output = summary_cleaned.apply(lambda x: str(x) + self.eos_token)
            return encoder_input.tolist(), decoder_input.tolist(), decoder_output.tolist()