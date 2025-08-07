import pandas as pd
import os
from dotenv import load_dotenv
import openai
from rouge import Rouge
from tqdm import tqdm
import concurrent.futures
import time
import logging
tqdm.pandas()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
PROJECT_DIR = "/data/ephemeral/home/nlp-5/pyeon/upstageailab-nlp-summarization-nlp_5"
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
TRAIN_CSV = os.path.join(DATA_DIR, "train_clean_special_tokens.csv")
DEV_CSV = os.path.join(DATA_DIR, "dev.csv")

load_dotenv(os.path.join(PROJECT_DIR, ".env"))
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")

client = openai.OpenAI(
    api_key=UPSTAGE_API_KEY,
    base_url="https://api.upstage.ai/v1/solar"
)


# Prompt를 생성하는 함수를 수정합니다.
def build_prompt(dialogue, type='summarization'):
    if type=='summarization':
        system_prompt = "You are a expert in the field of dialogue summarization, summarize the given dialogue in a concise manner. Follow the user's instruction carefully and provide a summary that is relevant to the dialogue."

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
        system_prompt = "You are a expert in the field of translation. Translate the given Korean dialogue into English. Follow the user's instruction carefully and provide a translation that is relevant to the original korean dialogue."

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
    elif type=='en2ja':
        system_prompt = "You are a expert in the field of translation. Translate the given English dialogue into Japanese. Follow the user's instruction carefully and provide a translation that is relevant to the original english dialogue."

        user_prompt = (
            "Following the instructions below, translate the given dialogue.\n"
            "Instructions:\n"
            "1. Read the dialogue carefully.\n"
            "2. Preserve named entities or english name in the dialogue.\n"
            "3. Each turn is distinguished by line feed, preserve the number of turns and representation of speaker such as #Person1#.\n"
            "4. Preserve Personal Identity Information masking such as #Person1#, #Email#, #Address#, etc."
            "5. Translate English to Japanese.\n\n"
            "English Dialogue:\n"
            f"{dialogue}\n\n"
            "Translation:\n"
        )
    elif type=='ja2ko':
        system_prompt = "You are a expert in the field of translation. Translate the given Japanese dialogue into Korean. Follow the user's instruction carefully and provide a translation that is relevant to the original japanese dialogue."

        user_prompt = (
            "Following the instructions below, translate the given dialogue.\n"
            "Instructions:\n"
            "1. Read the dialogue carefully.\n"
            "2. Preserve named entities or english name in the dialogue.\n"
            "3. Each turn is distinguished by line feed, preserve the number of turns and representation of speaker such as #Person1#.\n"
            "4. Translate Japanese to Korean.\n\n"
            "Japanese Dialogue:\n"
            f"{dialogue}\n\n"
            "Translation:\n"
        )
    elif type=='topic':
        system_prompt = "You are a expert in the field of topic classification. Extract discourse relations, speech acts, and conversational intentions in the summary and represents it as topic. Follow the user's instruction carefully and provide a topic that is relevant to the dialogue."

        user_prompt = (
            "Following the instructions below, extract topic in the given dialogue.\n"
            "Instructions:\n"
            "1. Read the dialogue carefully.\n"
            "2. Focus on named entities in the dialogue.\n"
            "3. Topic must be at most 3 words.\n"
            "4. Response in KOREAN with no prefix or suffix, only the topic.\n"
            "5. But if topic is English in dialogue, remain English.\n\n"
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

def chat_solar(dialogue, type='summarization'):
    max_tokens = 170
    if type in ['ko2en', 'en2ja', 'ja2ko']:
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
                    model="solar-pro2",
                    messages=prompt,
                    temperature=0.2,
                    top_p=0.3,
                    max_tokens=max_tokens,
                )
            else:
                output = client.chat.completions.create(
                    model="solar-pro2",
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

def process_row(row):
    idx, data = row
    dialogue = data['dialogue']
    fname = data['fname']
    # print("="*15,fname,"="*15)
    # try:
    #     summary = chat_solar(dialogue, type='summarization')
    # except Exception as e:
    #     print(f"[{idx}] Error in summarization: {e}")
    #     summary = None

    try:
        ko2en = chat_solar(dialogue, type='ko2en')
    except Exception as e:
        print(f"[{idx}] Error in ko2en: {e}")
        ko2en = None

    try:
        en2ja = chat_solar(ko2en, type='en2ja') if ko2en else None
    except Exception as e:
        print(f"[{idx}] Error in en2ja: {e}")
        en2ja = None
    
    try:
        ja2ko = chat_solar(en2ja, type='ja2ko') if en2ja else None
    except Exception as e:
        print(f"[{idx}] Error in ja2ko: {e}")
        ja2ko = None

    # try:
    #     re_summary = chat_solar(ja2ko, type='summarization') if ja2ko else None
    # except Exception as e:
    #     print(f"[{idx}] Error in re_summary: {e}")
    #     re_summary = None

    # try:
    #     topic = chat_solar(ja2ko, type='topic') if ja2ko else None
    # except Exception as e:
    #     print(f"[{idx}] Error in topic: {e}")
    #     topic = None

    # try:
    #     ner = chat_solar(dialogue, type='ner')
    # except Exception as e:
    #     print(f"[{idx}] Error in ner: {e}")
    #     ner = None

    return fname, ko2en, en2ja, ja2ko

def process_topic(row):
    idx, data = row
    dialogue = data['dialogue']
    fname = data['fname']
    try:
        topic = chat_solar(dialogue, type='topic') if dialogue else None
    except Exception as e:
        print(f"[{idx}] Error in topic: {e}")
        topic = None
    return fname, dialogue, topic

def retranslate_all(df):
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(process_row, row) for row in df.iterrows()]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(df)):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error in processing row: {e}")

    results_df = pd.DataFrame(
        #results, columns=['fname', 'summary_solar', 'topic_solar', 'dialogue_ko2en', 'dialogue_en2ja', 'dialogue_ja2ko', 're_summary_solar', 'ner_solar']
        results, columns=['fname', 'dialogue_ko2en', 'dialogue_en2ja', 'dialogue_ja2ko']
    )
    return results_df

import re
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

# def filter_topic(data):
#     if not isinstance(data, str):
#         return ""
#     filtered_data = re.split(r'\n', data, 1)[0]
#     return filtered_data

# def extract_topic_all(df):
#     results = []
#     with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
#         futures = [executor.submit(process_topic, row) for row in df.iterrows()]
#         for future in tqdm(concurrent.futures.as_completed(futures), total=len(df)):
#             try:
#                 results.append(future.result())
#             except Exception as e:
#                 print(f"Error in processing row: {e}")

#     results_df = pd.DataFrame(
#         results, columns=['fname', 'dialogue', 'topic']
#     )
#     results_df['topic'] = results_df['topic'].apply(filter_topic)
#     return results_df

if __name__ == '__main__':
    train_df = pd.read_csv(TRAIN_CSV)
    train_df = train_df.iloc[6000:]

    print("Processing train_df...")
    train_results = retranslate_all(train_df)
    train_results.to_csv(os.path.join(DATA_DIR, "train_backtranslation_ko2en2ja2ko_results_dialogue_6000-.csv"), index=False)
    print("Train results saved to data/train_backtranslation_ko2en2ja2ko_results_dialogue_6000-.csv")

    # print("Processing val_df...")
    # val_results = retranslate_all(val_df)
    # val_results.to_csv(os.path.join(DATA_DIR, "val_solar_results.csv"), index=False)
    # print("Validation results saved to data/val_solar_results.csv")

    # test_results = extract_topic_all(test_df)
    # test_results.to_csv(os.path.join(DATA_DIR, "test_topic_solar.csv"), index=False)
    # print("Validation results saved to data/test_solar_results.csv")
