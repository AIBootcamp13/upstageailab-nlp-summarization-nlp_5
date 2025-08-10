import pandas as pd
import re

# 단독형 지시어/대명사별로 치환할 목록 (사람/사물/장소/시간 등 확장)
DEICTIC_PERSON = [
    "그 사람", "이 사람", "저 사람", "그녀", "그분", "그 친구", "그 애", "그 남자", "그 여자", "그분들",
    "이 사람들", "그 사람들"
]
DEICTIC_THING = [
    "그것", "이것", "저것", "그거", "이거", "저거", "그건", "이건", "저건", "그걸", "이걸", "저걸", "그것들"
]
DEICTIC_TIME = [
    "그때", "이때", "저때", "그날", "이날", "저날"
]
DEICTIC_LOCATION = [
    "저기", "여기", "거기", "그곳", "이곳", "저곳", "그쪽", "이쪽", "저쪽"
]
JOSA_LIST = [
    "에게", "으로", "에서", "까지", "부터", "만", "보다", "처럼", "같이", "도", "는", "은", "이", "가", "를", "을", "에",
    "과", "와", "랑", "한테", "께", "하고"
]

# 조사 분리 함수
def split_josa(word):
    for josa in sorted(JOSA_LIST, key=len, reverse=True):
        if word.endswith(josa):
            return word[:-len(josa)], josa
    return word, ""

# 지시표현 보완 함수: 등장인물/실명 기반 우선 해소, 불가능할 때만 의미별 토큰 치환(사람/사물/장소/시간)
def resolve_deictic_with_real_entity(dialogue: str) -> str:
    # 패턴: "#PersonN#" 등 마스킹 토큰, 고유명사 추출
    masking_pattern = r"(#Person\d+#|#Name#|#PersonName#|[A-Z][a-z]+(?:\s[A-Z][a-z]+)*)"
    first_person = ["나", "저", "내", "저의", "본인"]
    second_person = ["너", "네", "당신", "너네", "너희"]
    lines = str(dialogue).split('\n')
    resolved = []

    # 대화 참여자 추출 (#PersonN# 기준)
    participants = []
    for line in lines:
        match = re.match(r'^(#Person\d+#):', line)
        if match:
            speaker = match.group(1)
            if speaker not in participants:
                participants.append(speaker)

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        match = re.match(r'^(#Person\d+#):\s*(.*)', line)
        if match:
            speaker = match.group(1)
            utterance = match.group(2)

            # 대화 참여자 구분(2인 대화)
            if len(participants) == 2:
                me, you = speaker, [p for p in participants if p != speaker][0]
            else:
                me, you = speaker, None

            # 대명사/지시어 치환 (1인칭, 2인칭)
            for fp in first_person:
                utterance = re.sub(rf"\b{fp}\b", me, utterance)
            for sp in second_person:
                if you:
                    utterance = re.sub(rf"\b{sp}\b", you, utterance)

            # ----- 단독형 지시어/대명사에 한해 치환 -----
            def replace_deictic_group(utter, word_list, token):
                stems = sorted(word_list, key=len, reverse=True)
                for stem in stems:
                    patt = re.compile(f"{re.escape(stem)}([가-힣]*)")
                    def repl(m):
                        s, j = split_josa(m.group(0))
                        return token + j
                    utter = patt.sub(repl, utter)
                return utter

            utterance = replace_deictic_group(utterance, DEICTIC_PERSON, "#지시_사람#")
            utterance = replace_deictic_group(utterance, DEICTIC_THING, "#지시_사물#")
            utterance = replace_deictic_group(utterance, DEICTIC_TIME, "#지시_시간#")
            utterance = replace_deictic_group(utterance, DEICTIC_LOCATION, "#지시_장소#")
            # -----------------------------------------

            resolved.append(f"{speaker}: {utterance}")
        else:
            resolved.append(line)

    return '\n'.join(resolved)

# 텍스트 클린 함수
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

        # 🔁 등장인물/실명 기반 지시표현 해소 전처리 적용 (불가시 의미 토큰 유지)
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