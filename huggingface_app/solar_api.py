import pandas as pd
import os
# from dotenv import load_dotenv
import openai
from tqdm import tqdm
import concurrent.futures
import time
import logging
import re

# Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
PROJECT_DIR = "/mnt/c/SKH/ai_lab_13/projects/nlp-text-summarization/song"
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
DEV_CSV = os.path.join(DATA_DIR, "dev.csv")

# load_dotenv(os.path.join(PROJECT_DIR, ".env"))
# UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")

def get_client(UPSTAGE_API_KEY):
    client = openai.OpenAI(
        api_key=UPSTAGE_API_KEY,
        base_url="https://api.upstage.ai/v1/solar"
    )
    return client

# Prompt를 생성하는 함수를 수정합니다.
def build_prompt(dialogue, type='summarization'):
    if type=='summarization':
        system_prompt = "You are a expert in the field of dialogue summarization, summarize the given dialogue in a concise manner. Follow the user\'s instruction carefully and provide a summary that is relevant to the dialogue."

        user_prompt = (
            "Following the instructions below, summarize the given document.\n"
            "Instructions:\n"
            "1. Read the dialogue carefully.\n"
            "2. Preserve named entities in the summary.\n"
            "3. Among special characters and symbols, only Arabic numerals, commas, and periods may be used.\n"
            "4. Reflect discourse relations, speech acts, and conversational intentions in the summary.\n"
            "5. Keep the summary concise and brief.\n"
            "6. Response in KOREAN.\n\n"
            "Dialogue:\n"
            f"{dialogue}\n\n"
            "Summary:\n"
        )
    elif type=='ko2en':
        system_prompt = "You are a expert in the field of translation. Translate the given Korean dialogue into English. Follow the user\'s instruction carefully and provide a translation that is relevant to the original korean dialogue."

        user_prompt = (
            "Following the instructions below, translate the given dialogue.\n"
            "Instructions:\n"
            "1. Read the dialogue carefully.\n"
            "2. Preserve named entities or english name in the dialogue.\n"
            "3. Each turn is distinguished by line feed, preserve the number of turns and representation of speaker such as #Person1#.\n"
            "4. Translate Korean to English.\n\n"
            "Korean Dialogue:\n"
            f"{dialogue}\n\n"
            "Translation:\n"
        )
    elif type=='en2ko':
        system_prompt = "You are a expert in the field of translation. Translate the given English dialogue into Korean. Follow the user\'s instruction carefully and provide a translation that is relevant to the original english dialogue."

        user_prompt = (
            "Following the instructions below, translate the given dialogue.\n"
            "Instructions:\n"
            "1. Read the dialogue carefully.\n"
            "2. Preserve named entities or english name in the dialogue.\n"
            "3. Each turn is distinguished by line feed, preserve the number of turns and representation of speaker such as #Person1#.\n"
            "4. Preserve Personal Identity Information masking such as #Person1#, #Email#, #Address#, etc."
            "5. Translate English to Korean.\n\n"
            "English Dialogue:\n"
            f"{dialogue}\n\n"
            "Translation:\n"
        )
    elif type=='topic':
        system_prompt = "You are a expert in the field of topic classification. Extract discourse relations, speech acts, and conversational intentions in the summary and represents it as topic. Follow the user\'s instruction carefully and provide a topic that is relevant to the dialogue."

        user_prompt = (
            "Following the instructions below, extract topic in the given dialogue.\n"
            "Instructions:\n"
            "1. Read the dialogue carefully.\n"
            "2. Focus on named entities in the dialogue.\n"
            "3. Topic must be at most 3 words.\n"
            "4. Response in KOREAN with no prefix or suffix, only the topic.\n\n"
            "Dialogue:\n"
            f"{dialogue}\n\n"
            "Topic:\n"
        )
    elif type == 'ner':
        system_prompt = "You are an expert in Named Entity Recognition. Extract named entities from the given dialogue."
        user_prompt = (
            "Following the instructions below, extract named entities from the given dialogue.\n"
            "Instructions:\n"
            "1. Read the dialogue carefully.\n"
            "2. Extract all named entities, including names of people, places, organizations, etc.\n"
            "3. Return the extracted entities as a comma-separated list.\n"
            "4. If no entities are found, return an empty string.\n\n"
            "Dialogue:\n"
            f"{dialogue}\n\n"
            "Named Entities:\n"
        )
    
    return [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

def chat_solar(dialogue, type, UPSTAGE_API_KEY, model="solar-pro2"):
    client = get_client(UPSTAGE_API_KEY)
    max_tokens = 170
    if type in ['en2ko', 'ko2en']:
        max_tokens = None # 따로 설정하지 않는다.
    elif type == 'topic':
        max_tokens = 15
    elif type == 'ner':
        max_tokens = 50
        
    prompt = build_prompt(dialogue, type)
    
    retries = 3
    delay = 1
    for i in range(retries):
        try:
            if max_tokens is not None:
                output = client.chat.completions.create(
                    model=model,
                    messages=prompt,
                    temperature=0.2,
                    top_p=0.3,
                    max_tokens=max_tokens,
                )
            else:
                output = client.chat.completions.create(
                    model=model,
                    messages=prompt,
                    temperature=0.2,
                    top_p=0.3,
                )
            return output.choices[0].message.content
        except openai.RateLimitError as e:
            logging.warning(f"Rate limit exceeded. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            return None
    logging.error("Failed to get response after several retries.")
    return None

def process_row(row, UPSTAGE_API_KEY, model="solar-pro2"):
    idx, data = row
    dialogue = data['dialogue']
    fname = data['fname']
    # print("="*15,fname,"="*15)
    try:
        summary = chat_solar(dialogue, type='summarization', UPSTAGE_API_KEY=UPSTAGE_API_KEY, model=model)
    except Exception as e:
        print(f"[{idx}] Error in summarization: {e}")
        summary = None

    try:
        ko2en = chat_solar(dialogue, type='ko2en', UPSTAGE_API_KEY=UPSTAGE_API_KEY, model=model)
    except Exception as e:
        print(f"[{idx}] Error in ko2en: {e}")
        ko2en = None

    try:
        en2ko = chat_solar(ko2en, type='en2ko', UPSTAGE_API_KEY=UPSTAGE_API_KEY, model=model) if ko2en else None
    except Exception as e:
        print(f"[{idx}] Error in en2ko: {e}")
        en2ko = None

    try:
        re_summary = chat_solar(en2ko, type='summarization', UPSTAGE_API_KEY=UPSTAGE_API_KEY, model=model) if en2ko else None
    except Exception as e:
        print(f"[{idx}] Error in re_summary: {e}")
        re_summary = None

    try:
        topic = chat_solar(en2ko, type='topic', UPSTAGE_API_KEY=UPSTAGE_API_KEY, model=model) if en2ko else None
    except Exception as e:
        print(f"[{idx}] Error in topic: {e}")
        topic = None

    try:
        ner = chat_solar(dialogue, type='ner', UPSTAGE_API_KEY=UPSTAGE_API_KEY, model=model)
    except Exception as e:
        print(f"[{idx}] Error in ner: {e}")
        ner = None

    return fname, summary, topic, ko2en, en2ko, re_summary, ner

# def retranslate_all_multi_thread(df):
#     results = []
#     with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
#         futures = [executor.submit(process_row, row) for row in df.iterrows()]
#         for future in tqdm(concurrent.futures.as_completed(futures), total=len(df)):
#             try:
#                 results.append(future.result())
#             except Exception as e:
#                 print(f"Error in processing row: {e}")

#     results_df = pd.DataFrame(
#         results, columns=['fname', 'summary_solar', 'topic_solar', 'dialogue_ko2en', 'dialogue_en2ko', 're_summary_solar', 'ner_solar']
#     )
#     return results_df

def filter_solar(data):
    """
    주어진 텍스트 데이터에 대해 다음을 수행합니다:
    1. \n\n 이후의 텍스트를 모두 제거합니다.
    2. 괄호 표현 ((), [], {}, <>, #)을 제거합니다.

    Args:
        data (str): 필터링할 텍스트 데이터.

    Returns:
        str: 필터링된 텍스트 데이터.
    """
    # 1. \n\n 이후의 텍스트 제거
    if not isinstance(data, str):
        return ""
    filtered_data = re.split(r'\n\n', data, 1)[0]

    # 2. 괄호 표현 제거 ((), [], {}, <>, ** **)
    # 괄호와 그 안의 내용을 제거하는 정규 표현식
    # \((.*?)\): () 안의 내용 제거
    # \[.*?\]: [] 안의 내용 제거
    # \{.*?\}: {} 안의 내용 제거
    # \<.*?\>: <> 안의 내용 제거
    # \*\*.*?\*\*: ** 안의 내용 제거
    # \*.*?\*: * 안의 내용 제거
    filtered_data = re.sub(r'\([^)]*\)|\[[^\]]*\]|\{[^}]*\}|\<[^>]*\>|\*\*.*?\*\*|\*.*?\*', '', filtered_data)
    return filtered_data.strip() # 공백 제거

def retranslate_single(series, UPSTAGE_API_KEY, model="solar-pro2"):
    fname, summary, topic, _, en2ko, re_summary, ner = process_row(series, UPSTAGE_API_KEY, model=model)
    results = []
    
    for data in [summary, topic, en2ko, re_summary, ner]:
        results.append(filter_solar(data))

    return fname, results

if __name__ == '__main__':
    pass
    # train_df = pd.read_csv(TRAIN_CSV)
    # val_df = pd.read_csv(DEV_CSV)

    # print("Processing train_df...")
    # train_results = retranslate_all_multi_thread(train_df)
    # train_results.to_csv(os.path.join(DATA_DIR, "train_solar_results.csv"), index=False)
    # print("Train results saved to data/train_solar_results.csv")

    # print("Processing val_df...")
    # val_results = retranslate_all_multi_thread(val_df)
    # val_results.to_csv(os.path.join(DATA_DIR, "val_solar_results.csv"), index=False)
    # print("Validation results saved to data/val_solar_results.csv")