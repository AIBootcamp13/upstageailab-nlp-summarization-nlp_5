import pandas as pd
import re

# 인물명 추출 및 #PersonN# 매핑 (한글+영문 대응)
def extract_speakers(dialogue):
    speakers = re.findall(r'(?:Ms\.?\s*)?([A-Z][a-zA-Z]+):|([가-힣]+):', dialogue)
    names = []
    for en, ko in speakers:
        name = en if en else ko
        if name and name not in names:
            names.append(name)
    return names

def make_name2token_map(speakers):
    return {name: f"#Person{idx+1}#" for idx, name in enumerate(speakers)}

def replace_names_with_tokens(text, name2token):
    for name, token in name2token.items():
        text = re.sub(rf'(Ms\.?\s*)?{re.escape(name)}', token, text)
    return text

def clean_redundant(text):
    text = re.sub(r'(#Person\d+#)(\s*,?\s*\1)+', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def keep_main_verbs(text):
    # GOLD 기준 확장된 주요 동사 리스트
    main_verbs = [
        '요청', '제안', '설명', '논의', '추천', '결정', '조언', '안내',
        '거절', '공유', '수락', '전달', '축하', '권유', '토론'
    ]
    sents = re.split(r'(?<=[.?!])\s+', text.strip())
    filtered = [s for s in sents if any(verb in s for verb in main_verbs)]
    if filtered:
        # 동사 포함 문장 + 첫 문장도 포함 (중복 제거)
        if sents[0] not in filtered:
            filtered.insert(0, sents[0])
        return ' '.join(filtered)
    else:
        return sents[0] if sents else text

def gold_style_postprocess(summary, dialogue, max_sent=2, max_char=140):
    speakers = extract_speakers(dialogue)
    name2token = make_name2token_map(speakers)
    summary = replace_names_with_tokens(summary, name2token)
    summary = clean_redundant(summary)
    summary = keep_main_verbs(summary)
    sents = re.split(r'(?<=[.?!])\s+', summary.strip())
    out = ' '.join(sents[:max_sent])
    out = out[:max_char]
    return out

# 파일 경로는 본인 환경에 맞게 변경하세요
PRED_PATH = '/data/ephemeral/home/nlp-5/eunbyul/ah/prediction/result1.csv'
GOLD_PATH = '/data/ephemeral/home/nlp-5/eunbyul/ah/data/dev.csv'
OUTPUT_PATH = '/data/ephemeral/home/nlp-5/eunbyul/ah/prediction/post_result1.csv'

pred = pd.read_csv(PRED_PATH)
gold = pd.read_csv(GOLD_PATH)

# dev와 pred 행수 일치 가정
post_summaries = []
for i, row in pred.iterrows():
    dialogue = gold.loc[i, 'dialogue']
    summary = row['summary']
    post_summary = gold_style_postprocess(summary, dialogue)
    post_summaries.append(post_summary)

pred['summary'] = post_summaries
pred.to_csv(OUTPUT_PATH, index=False)
print(f"Postprocessed summaries saved to {OUTPUT_PATH}")