import pandas as pd
import re

# 데이터 기반 빈번 지시명사(어간) 리스트 (5회 이상 등장한 복합 지시명사 포함)
DEICTIC_NOUNS = [
    # 아래 리스트는 네가 이미지로 준 "5회 이상" 등장 빈도 기준으로 추출함
    # 실제 프로젝트에선 통계 기반으로 더 확장/조정 가능
    "그게","그런","그거","그건","그녀","저기","그 사람","이거","그때","이건","그걸","그분","이게","이 집","그곳",
    "그것","그 친구","그쪽","이걸","그의","그 얘기","그 일","이 다음","저","이곳","저 여자","이 회사","이 차",
    "이것","그녀의","저쪽","그","그날","그 말","그 사람들","이 도시","그 돈","이 책","그 애","이 시간","그 문제",
    "이 문제","그다음에","이 지역","그 사람의","그 남자","저번","이 일","이 버스","그 수업","이 약","그 책",
    "그 자리","저 사람","이 셔츠","이쪽","이 선물","이 자리","그 후","그 정보","그 영화","그분들","이 날","저거",
    "이 새","이 요즘","그 회사","저건","이 신발","이 책들","이 사람들","저의","그 차","이 드레스","이 길","저요",
    "이 수업","그 정도면","그 전","그 이후에","이 보고서","그 시간","이 선생님","그 부분","그 여자","그 다음엔",
    "이 방","이 가게","이 초콜릿","이 근처","이 도움","그 가격","이 소포","그 기간","이 서류","이 영화","그것들",
    "이 색","이 계약","이 주소","저게","그 길","그 이후","그 순간","그 표지판","그 근처","이 지역의","이 음악",
    "이 가방","이 동네","이 모델","이 스타일","이 돈","그 생각","이곳의","이 가격","저 건물","이디어","이 바지",
    "그분의","이대","그림","그 집","이 서비스","그 점","그다음","그 외에","이 코트","그 마음","이후에","이 정도면",
    "그러시군요","이 노래","이 처방전","저 길","이 옷","이 문","이 주말","이 반지","이 사진","이 여기","그 후에",
    "그 다음에","저 ","그 외","그 다음","이 편지","그 정","이 선반","이 공","그 중","이 지금","이 제품","이 사람",
    "이 직업","이 시간대에","그 둘","이 그","그 분야","이 꼭","이블","이웃","그땐","이 학교","이 양식","이 종이",
    "이 분야","이 줄","이 생각해","그곳의","이 사무실","그중","이 회사의","이 기계","그 잡지","그 나라의","그 전에",
    "그 버스","이 신문","이 모두","이 스웨터","이 운동","이 우리","이 나라","이 제안","그 색상","이 기간","이 아파트",
    "그 후엔","이 도시의","그 드레스","이 신청서","이 음식","그러네요","이 건물","이 준비되어","이 때문","그 가격대",
    "이 직책","이 프로젝트","이 도시에서","그 학생","이 그림","그 나라","이룰","그 식당","이웃들","이 하루","그 이후로",
    "이 여기서","이 얘기","그 선생님","이 인기","그 소식","이 실제","이 직무","이 첫","그 노래","이 정","그 비행기",
    "이 비용","이 완벽","그 안","이 말","이 분야의","그 목표","이 거리","그 분","이 세계","이 광고","그 뒤","그 정도",
    "이 너희","이 컴퓨터","이 기회","그때면","이 환전","이 대부분","이 치마","그때까지","그 얘긴","이 잡지","그 학교",
    "이름이","그 선물","이 경기","이 프로그램","이 서류들","그 이야기","이 나무"
]

# 조사 리스트 (붙임 조사 분리용)
JOSA_LIST = [
    "에게","으로","에서","까지","부터","만","보다","처럼","같이","도","는","은","이","가","를","을","에",
    "과","와","랑","한테","께","하고"
]
def split_josa(word):
    for josa in sorted(JOSA_LIST, key=len, reverse=True):
        if word.endswith(josa):
            return word[:-len(josa)], josa
    return word, ""

# 복합형 지시명사 패턴 처리 (A): "그 집", "이 사진" 등 → 직전 등장 엔티티 매핑
def replace_deictic_noun_patterns(utterance, entity_memory):
    for dword in sorted(DEICTIC_NOUNS, key=len, reverse=True):
        patt = re.compile(re.escape(dword) + r'([가-힣]*)')
        def repl(m):
            noun = dword.split()[-1]
            # 엔티티 memory에서 해당 명사 포함된 엔티티(마스킹, 실명 등) 찾기
            for prev in reversed(entity_memory):
                if noun in prev:
                    return prev + m.group(1)
            return "#지시_사물#" + m.group(1)  # fallback
        utterance = patt.sub(repl, utterance)
    return utterance

# 단독형(지시어) 토큰화용 리스트 (B)
DEICTIC_PERSON = [
    "그 사람","이 사람","저 사람","그녀","그분","그 친구","그 애","그 남자","그 여자","그분들","이 사람들","그 사람들"
]
DEICTIC_THING = [
    "그것","이것","저것","그거","이거","저거","그건","이건","저건","그걸","이걸","저걸","그것들"
]
DEICTIC_TIME = [
    "그때","이때","저때","그날","이날","저날"
]
DEICTIC_LOCATION = [
    "저기","여기","거기","그곳","이곳","저곳","그쪽","이쪽","저쪽"
]
def replace_deictic_group(utter, word_list, token):
    # 조사 분리 치환
    stems = sorted(word_list, key=len, reverse=True)
    for stem in stems:
        patt = re.compile(f"{re.escape(stem)}([가-힣]*)")
        def repl(m):
            s, j = split_josa(m.group(0))
            return token + j
        utter = patt.sub(repl, utter)
    return utter

# 실제 전처리 메인 함수
def resolve_deictic_with_real_entity(dialogue: str) -> str:
    """
    - 1, 2인칭 대명사 → #PersonN#으로 치환
    - 복합형 지시명사("그 집" 등) → 앞선 등장 엔티티로 연결, 없으면 의미 토큰
    - 단독형 지시어(사람/사물/장소/시간) → 의미 토큰으로 변환
    """
    masking_pattern = r"(#\w+#|[A-Z][a-z]+(?:\s[A-Z][a-z]+)*)"
    first_person = ["나", "저", "내", "저의", "본인", "난", "저희"]
    second_person = ["너", "네", "당신", "너네", "너희", "자기", "여보", "자기야", "넌"]
    lines = str(dialogue).split('\n')
    resolved = []
    entity_memory = []  # 대화 내 등장한 엔티티 기록(최근 등장 우선)

    # 대화 참여자 추출 (#PersonN# 기반)
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

            # 대화 참여자 구분(2인 대화 기준)
            if len(participants) == 2:
                me, you = speaker, [p for p in participants if p != speaker][0]
            else:
                me, you = speaker, None

            # [A] 복합형 지시명사 처리 (코어퍼런스)
            utterance = replace_deictic_noun_patterns(utterance, entity_memory)
            # 현재 라인 entity 추출해서 memory에 저장
            found_entities = re.findall(masking_pattern, utterance)
            entity_memory.extend(found_entities)

            # 1,2인칭 대명사 치환
            for fp in first_person:
                utterance = re.sub(rf"\b{fp}\b", me, utterance)
            for sp in second_person:
                if you:
                    utterance = re.sub(rf"\b{sp}\b", you, utterance)

            # [C] 단독형 지시어/대명사 토큰화
            utterance = replace_deictic_group(utterance, DEICTIC_PERSON, "#지시_사람#")
            utterance = replace_deictic_group(utterance, DEICTIC_THING, "#지시_사물#")
            utterance = replace_deictic_group(utterance, DEICTIC_TIME, "#지시_시간#")
            utterance = replace_deictic_group(utterance, DEICTIC_LOCATION, "#지시_장소#")

            resolved.append(f"{speaker}: {utterance}")
        else:
            resolved.append(line)
    return '\n'.join(resolved)

# 텍스트 클린 함수: 불필요한 이모티콘/자소/줄바꿈 등 정제
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # 줄바꿈 표현 통일
    text = text.replace("\\n", "\n").replace("<br>", "\n").replace("</s>", "\n")
    # 자소/이모티콘 등 제거
    text = re.sub(r"\b[ㄱ-ㅎㅏ-ㅣ]{2,}\b", "", text)
    text = re.sub(r"[ㄱ-ㅎㅏ-ㅣ]{3,}", "", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()

# 데이터 전처리 클래스: CSV → DataFrame, 인코더/디코더 입력 생성
class Preprocess:
    # 클래스 초기화 메서드
    def __init__(self, bos_token: str, eos_token: str) -> None:
        self.bos_token = bos_token  # 시작 토큰
        self.eos_token = eos_token  # 종료 토큰

    @staticmethod
    # 실험에 필요한 컬럼만 추출, 전처리까지 적용
    def make_set_as_df(file_path, is_train=True):
        df = pd.read_csv(file_path)
        # 지시표현/코어퍼런스 해소 전처리 적용
        df['dialogue'] = df['dialogue'].apply(resolve_deictic_with_real_entity)
        # 텍스트 클린
        df['dialogue'] = df['dialogue'].apply(clean_text)
        # 반환 컬럼 제한
        if is_train:
            df = df[['fname','dialogue','summary']]
        else:
            df = df[['fname','dialogue']]
        return df

    # BART 입력 형태 맞추는 전처리
    def make_input(self, dataset, is_test=False):
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