import gradio as gr
import pandas as pd
import os
import re
# from dotenv import load_dotenv
import sys
import gc
import traceback
from transformers import AutoTokenizer
from nltk.util import ngrams
from difflib import SequenceMatcher
import io

sys.path.append(os.path.dirname(__file__))
from solar_api import retranslate_single

# --- Configuration ---
# Assuming the script is run from the root of the project
# load_dotenv()
DATA_DIR = './data'
TRAIN1_CSV = os.path.join(DATA_DIR, "train_1.csv")
TRAIN2_CSV = os.path.join(DATA_DIR, "train_2.csv")
DEV_CSV = os.path.join(DATA_DIR, "dev.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
TRAIN_SOLAR_CSV = os.path.join(DATA_DIR, "train_solar_results_filtered")
DEV_SOLAR_CSV = os.path.join(DATA_DIR, "val_solar_results_filtered.csv")
SPECIAL_TOKENS = [
    '#Person1#','#Person2#','#Person3#','#Person4#',
    '#Person5#','#Person6#','#Person7#',
    '#PhoneNumber#','#Address#','#DateOfBirth#','#PassportNumber#','#SSN#','#CardNumber#','#CarNumber#','#Email#'
]

# UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")

# --- Load Data ---
# try:
t1 = pd.read_csv(TRAIN1_CSV)
t2 = pd.read_csv(TRAIN2_CSV)
train_df = pd.concat([t1,t2])
del t1, t2
gc.collect()
val_df = pd.read_csv(DEV_CSV)
test_df = pd.read_csv(TEST_CSV)
llm_train_df = None
llm_val_df = None

dfs = []
for i in range(8):
    df = pd.read_csv(f"{TRAIN_SOLAR_CSV}{i+1}.csv")
    dfs.append(df)
llm_train_df = pd.concat(dfs)
del dfs, df
gc.collect()
llm_val_df = pd.read_csv(DEV_SOLAR_CSV)

# --- Block 1: Explore by fname ---

def get_NER(dialogue):
    # not implemented
    return None

def get_sample_by_index(dataset_name, index, use_NER=True):
    df = train_df if dataset_name == "train" else val_df
    llm_df = llm_train_df if dataset_name == "train" else llm_val_df
    index = int(index)

    if 0 <= index < len(df):
        sample = df.iloc[index]

        # get NER by NER-model if exists.
        ner = get_NER(sample['dialogue']) if use_NER else None
        if ner is None:
            ner = "NER model is not ready."
# summary_solar_filter,topic_solar_filter,dialogue_ko2en_filter,dialogue_en2ko_filter,re_summary_solar_filter,ner_solar_filter
        try:
            llm_sample = llm_df.iloc[index]
        except:
            llm_sample = {'summary_solar_filter': 'N/A','ner_solar_filter': 'N/A','re_summary_solar_filter': 'N/A','topic_solar_filter': 'N/A', 'dialogue_en2ko_filter': 'N/A'}

        return (sample['fname'], sample['dialogue'], sample['summary'], ner, sample['topic'],
                llm_sample['summary_solar_filter'],llm_sample['ner_solar_filter'],llm_sample['re_summary_solar_filter'],llm_sample['topic_solar_filter'],llm_sample['dialogue_en2ko_filter'], index)
    return "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", index

def update_outputs(dataset_name, index):
    fname, dialogue, summary, ner, topic, llm_summary, llm_ner, re_summary, re_topic, re_dialogue, _ = get_sample_by_index(dataset_name, index)
    return fname, f"```\n{dialogue}\n```", f"```\n{summary}\n```", f"```\n{ner}\n```", f"```\n{topic}\n```", f"```\n{llm_summary}\n```", f"```\n{llm_ner}\n```", f"```\n{re_summary}\n```", f"```\n{re_topic}\n```", f"```\n{re_dialogue}\n```"

def next_sample(dataset_name, index):
    df = train_df if dataset_name == "train" else val_df
    new_index = min(int(index) + 1, len(df) - 1)
    fname, dialogue, summary, ner, topic, llm_summary, llm_ner, re_summary, re_topic, re_dialogue, _ = get_sample_by_index(dataset_name, new_index)
    return new_index, fname, f"```\n{dialogue}\n```", f"```\n{summary}\n```", f"```\n{ner}\n```", f"```\n{topic}\n```", f"```\n{llm_summary}\n```", f"```\n{llm_ner}\n```", f"```\n{re_summary}\n```", f"```\n{re_topic}\n```", f"```\n{re_dialogue}\n```"

def prev_sample(dataset_name, index):
    new_index = max(int(index) - 1, 0)
    fname, dialogue, summary, ner, topic, llm_summary, llm_ner, re_summary, re_topic, re_dialogue, _ = get_sample_by_index(dataset_name, new_index)
    return new_index, fname, f"```\n{dialogue}\n```", f"```\n{summary}\n```", f"```\n{ner}\n```", f"```\n{topic}\n```", f"```\n{llm_summary}\n```", f"```\n{llm_ner}\n```", f"```\n{re_summary}\n```", f"```\n{re_topic}\n```", f"```\n{re_dialogue}\n```"

def reset_index_on_split_change(dataset_name):
    # When the dataset changes, reset the index to 0 and update the display
    fname, dialogue, summary, ner, topic, llm_summary, llm_ner, re_summary, re_topic, re_dialogue, _ = get_sample_by_index(dataset_name, 0)
    return 0, fname, f"```\n{dialogue}\n```", f"```\n{summary}\n```", f"```\n{ner}\n```", f"```\n{topic}\n```", f"```\n{llm_summary}\n```", f"```\n{llm_ner}\n```", f"```\n{re_summary}\n```", f"```\n{re_topic}\n```", f"```\n{re_dialogue}\n```"

def get_sample_by_fname(dataset_name, fname):
    df = train_df if dataset_name == "train" else val_df
    
    # Find the index for the given fname
    if fname in df['fname'].values:
        index = df[df['fname'] == fname].index[0]
        # Use the existing function with the found index
        fname, dialogue, summary, ner, topic, llm_summary, llm_ner, re_summary, re_topic, re_dialogue, _ = get_sample_by_index(dataset_name, index)
        return index, fname, f"```\n{dialogue}\n```", f"```\n{summary}\n```", f"```\n{ner}\n```", f"```\n{topic}\n```", f"```\n{llm_summary}\n```", f"```\n{llm_ner}\n```", f"```\n{re_summary}\n```", f"```\n{re_topic}\n```", f"```\n{re_dialogue}\n```"
    else:
        # fname not found
        return -1, f"fname '{fname}' not found in {dataset_name} dataset.", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"

def request_solar_api(api_key, model_name, dataset_name, index):
    if not api_key or not api_key.strip():
        error_msg = "```\n[오류] UPSTAGE_API_KEY가 입력되지 않았습니다. API 키를 입력하고 다시 시도해주세요.\n```"
        return error_msg, error_msg, error_msg, error_msg, error_msg

    df = train_df if dataset_name == "train" else val_df
    index = int(index)
    row_data = df.iloc[index]
    # process_row expects a tuple of (index, series)
    row_for_api = (index, row_data)

    try:
        # retranslate_single returns (fname, results_list)
        fname, results = retranslate_single(row_for_api, api_key, model=model_name)
        
        # results is a list: [summary, topic, en2ko, re_summary, ner]
        summary, topic, re_dialogue, re_summary, ner = results
        
        # Match the output order for the UI: 
        # llm_summary_out, llm_ner_out, re_summary_out, re_topic_out, re_dialogue_out
        return f"```\n{summary}\n```", f"```\n{ner}\n```", f"```\n{re_summary}\n```", f"```\n{topic}\n```", f"```\n{re_dialogue}\n```"
    except Exception as e:
        # It's possible the API key is invalid, causing an error inside retranslate_single
        error_msg = f"```\n[오류] API 호출에 실패했습니다. API 키가 잘못되었거나 네트워크 문제일 수 있습니다.\n{e}\n```"
        return error_msg, error_msg, error_msg, error_msg, error_msg

# --- Block 2: Explore by Topic ---

# Merge dataframes
train_df['source'] = 'train'
val_df['source'] = 'val'
merged_df = pd.concat([train_df, val_df], ignore_index=True)
merged_df['topic'] = merged_df['topic'].astype(str)
topics = sorted(merged_df['topic'].unique().tolist())
topic_df_grouped = merged_df.groupby('topic')

def get_topic_data(topic_name):
    if topic_name not in topic_df_grouped.groups:
        return "Topic not found.", "Topic not found."

    topic_df = topic_df_grouped.get_group(topic_name)
    
    # Clean and split topic name into keywords
    keywords = re.sub(r"[,\\.]", " ", topic_name, flags=re.IGNORECASE)
    keywords = re.sub(r"\\s+", " ", topic_name, flags=re.IGNORECASE)
    keywords = keywords.strip().split(" ")

    def highlight_keywords(text, keywords):
        for keyword in keywords:
            text = re.sub(f"({re.escape(keyword)})", rf'<span style="background-color: yellow;">{keyword}</span>', text, flags=re.IGNORECASE)
        return text

    train_output = ""
    val_output = ""

    for _, row in topic_df.iterrows():
        dialogue = highlight_keywords(row['dialogue'], keywords)
        summary = highlight_keywords(row['summary'], keywords)
        
        formatted_output = f"fname: {row['fname']}\n\n"
        formatted_output += f"Dialogue:\n{dialogue}\n\n"
        formatted_output += f"Summary:\n{summary}\n\n"
        formatted_output += ("-" * 20) + "\n"

        if row['source'] == 'train':
            train_output += formatted_output
        elif row['source'] == 'val':
            val_output += formatted_output
            
    return train_output, val_output

def update_topic_display_by_name(topic_name):
    return get_topic_data(topic_name)

def change_topic(change, current_topic_name):
    current_index = topics.index(current_topic_name)
    new_index = (current_index + change + len(topics)) % len(topics)
    new_topic = topics[new_index]
    train_output, val_output = get_topic_data(new_topic)
    return new_topic, train_output, val_output


# --- Block 3: Validation Inference Exploration ---

def get_rouge_highlighted_html(text_a, text_b, tokenizer_name):
    # 두 개의 텍스트(text_a, text_b)를 비교하여 ROUGE 점수를 기반으로 HTML 하이라이팅을 적용합니다.
    # ROUGE-1 (unigram)은 노란색, ROUGE-2 (bigram)는 연두색, ROUGE-L (LCS)은 하늘색으로 표시됩니다.
    # 토크나이저는 Hugging Face의 AutoTokenizer를 사용하여 로드하며, SPECIAL_TOKENS를 추가합니다.
    try:
        # 지정된 이름의 토크나이저를 로드합니다.
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        # 사전에 정의된 SPECIAL_TOKENS를 토크나이저에 추가합니다.
        tokenizer.add_special_tokens({'additional_special_tokens': SPECIAL_TOKENS})
    except Exception as e:
        # 토크나이저 로딩에 실패하면 오류 메시지를 반환합니다.
        error_msg = f"<p>Error loading tokenizer '{tokenizer_name}': {e}</p>"
        return error_msg, error_msg, None

    # 입력 텍스트를 토큰화합니다.
    tokens_a = tokenizer.tokenize(text_a)
    tokens_b = tokenizer.tokenize(text_b)

    # --- ROUGE Score Calculation ---
    # ROUGE-1
    unigrams_a = set(tokens_a)
    unigrams_b = set(tokens_b)
    common_unigrams_count = len(unigrams_a & unigrams_b)
    r1_precision = common_unigrams_count / len(unigrams_b) if len(unigrams_b) > 0 else 0
    r1_recall = common_unigrams_count / len(unigrams_a) if len(unigrams_a) > 0 else 0
    r1_f1 = 2 * (r1_precision * r1_recall) / (r1_precision + r1_recall) if (r1_precision + r1_recall) > 0 else 0

    # ROUGE-2
    bigrams_a = set(ngrams(tokens_a, 2))
    bigrams_b = set(ngrams(tokens_b, 2))
    common_bigrams_count = len(bigrams_a & bigrams_b)
    r2_precision = common_bigrams_count / len(bigrams_b) if len(bigrams_b) > 0 else 0
    r2_recall = common_bigrams_count / len(bigrams_a) if len(bigrams_a) > 0 else 0
    r2_f1 = 2 * (r2_precision * r2_recall) / (r2_precision + r2_recall) if (r2_precision + r2_recall) > 0 else 0

    # ROUGE-L
    matcher_for_score = SequenceMatcher(None, tokens_a, tokens_b, autojunk=False)
    lcs_length = sum(block.size for block in matcher_for_score.get_matching_blocks())
    rl_precision = lcs_length / len(tokens_b) if len(tokens_b) > 0 else 0
    rl_recall = lcs_length / len(tokens_a) if len(tokens_a) > 0 else 0
    rl_f1 = 2 * (rl_precision * rl_recall) / (rl_precision + rl_recall) if (rl_precision + rl_recall) > 0 else 0

    rouge_scores = {
        "rouge-1": r1_f1,
        "rouge-2": r2_f1,
        "rouge-l": rl_f1,
    }
    # --- End ROUGE Score Calculation ---


    # ROUGE 메트릭별 하이라이트 색상을 정의합니다.
    colors = {
        'rouge-1': 'yellow',      # 1-gram (단일 토큰) 일치
        'rouge-2': 'lightgreen',  # 2-gram (연속된 두 토큰) 일치
        'rouge-l': 'lightcoral'   # Longest Common Subsequence (최장 공통 부분 서열) 일치
    }

    # 각 텍스트의 토큰에 대한 하이라이트 정보를 저장할 맵을 생성합니다.
    # 초기에는 하이라이트가 없는 상태(None)로 초기화합니다.
    highlight_map_a = [None] * len(tokens_a)
    highlight_map_b = [None] * len(tokens_b)

    # ROUGE-1 (unigrams) 하이라이팅
    # 두 텍스트에 공통으로 나타나는 모든 단일 토큰(unigram)을 찾습니다.
    common_unigrams = unigrams_a & unigrams_b
    # 공통 unigram에 해당하는 토큰 위치에 'rouge-1' 색상을 매핑합니다.
    for i, token in enumerate(tokens_a):
        if token in common_unigrams:
            highlight_map_a[i] = colors['rouge-1']
    for i, token in enumerate(tokens_b):
        if token in common_unigrams:
            highlight_map_b[i] = colors['rouge-1']

    # ROUGE-2 (bigrams) 하이라이팅
    # 두 텍스트에 공통으로 나타나는 모든 연속된 두 토큰(bigram)을 찾습니다.
    common_bigrams = bigrams_a & bigrams_b
    # 공통 bigram에 해당하는 토큰 위치에 'rouge-2' 색상을 덮어씁니다.
    # ROUGE-1 및 ROUGE-L보다 우선순위가 높게 적용됩니다.
    for i in range(len(tokens_a) - 1):
        if (tokens_a[i], tokens_a[i+1]) in common_bigrams:
            highlight_map_a[i] = colors['rouge-2']
            highlight_map_a[i+1] = colors['rouge-2']
    for i in range(len(tokens_b) - 1):
        if (tokens_b[i], tokens_b[i+1]) in common_bigrams:
            highlight_map_b[i] = colors['rouge-2']
            highlight_map_b[i+1] = colors['rouge-2']

    # ROUGE-L (LCS) 하이라이팅
    # Longest Common Subsequence (최장 공통 부분 서열)를 찾습니다.
    # SequenceMatcher를 사용하여 두 토큰 시퀀스 간의 일치하는 블록을 찾습니다.
    matcher = SequenceMatcher(None, tokens_a, tokens_b, autojunk=False)
    for block in matcher.get_matching_blocks():
        if block.size > 0:
            # 일치하는 블록의 모든 토큰 위치에 'rouge-l' 색상을 덮어씁니다.
            # ROUGE-1보다 우선순위가 높게 적용됩니다.
            for i in range(block.size):
                highlight_map_a[block.a + i] = colors['rouge-l']
                highlight_map_b[block.b + i] = colors['rouge-l']

    def build_html_from_map(tokens, h_map, tokenizer):
        # 하이라이트 맵을 기반으로 HTML을 생성합니다.
        # 연속된 동일한 색상의 토큰들을 하나의 <span> 태그로 묶어 효율성을 높입니다.
        if not tokens:
            return ""
        
        result_html = []
        if not tokens:
            return ""
        
        # 첫 번째 토큰부터 시작합니다.
        current_color = h_map[0]
        token_buffer = [tokens[0]]

        # 두 번째 토큰부터 순회하면서 색상이 바뀌는 지점을 찾습니다.
        for i in range(1, len(tokens)):
            if h_map[i] != current_color:
                # 색상이 바뀌면, 버퍼에 쌓인 토큰들을 문자열로 변환하고 <span> 태그로 감쌉니다.
                text = tokenizer.convert_tokens_to_string(token_buffer)
                if current_color:
                    result_html.append(f'<span style="background-color: {current_color};">{text}</span>')
                else:
                    result_html.append(text)
                
                # 버퍼를 비우고 새로운 토큰과 색상으로 다시 시작합니다.
                token_buffer = [tokens[i]]
                current_color = h_map[i]
            else:
                # 색상이 같으면 버퍼에 토큰을 추가합니다.
                token_buffer.append(tokens[i])
        
        # 마지막 버퍼에 남은 토큰들을 처리합니다.
        text = tokenizer.convert_tokens_to_string(token_buffer)
        if current_color:
            result_html.append(f'<span style="background-color: {current_color};">{text}</span>')
        else:
            result_html.append(text)
            
        return "".join(result_html)

    # 각 텍스트에 대해 하이라이트가 적용된 HTML을 생성합니다.
    html_a = build_html_from_map(tokens_a, highlight_map_a, tokenizer)
    html_b = build_html_from_map(tokens_b, highlight_map_b, tokenizer)

    # 최종적으로 생성된 HTML을 <p> 태그로 감싸서 반환합니다.
    return f"<p>{html_a}</p>", f"<p>{html_b}</p>", rouge_scores


def get_validation_data(fname, inference_file, tokenizer_name):
    # 1. Get original data
    if fname not in val_df['fname'].values:
        msg = f"fname '{fname}' not found in validation dataset."
        return msg, msg, msg, msg, "N/A"

    original_sample = val_df[val_df['fname'] == fname].iloc[0]
    dialogue_orig = f"```\n{original_sample['dialogue']}\n```"
    topic_orig = f"```\n{original_sample['topic']}\n```"
    summary_orig_text = original_sample['summary']

    # 2. Defaults for outputs
    summary_orig_out = f"<p>{summary_orig_text}</p>"
    summary_infer_out = "Please upload an inference CSV file and select a sample."
    rouge_scores_out = "Upload an inference file to see ROUGE scores."

    # 3. Process inference file if available
    if inference_file is not None:
        try:
            inference_df = pd.read_csv(inference_file.name)
            if fname in inference_df['fname'].values:
                inference_sample = inference_df[inference_df['fname'] == fname].iloc[0]
                summary_infer_text = inference_sample['summary']
                
                summary_orig_out, summary_infer_out, rouge_scores = get_rouge_highlighted_html(
                    summary_orig_text, summary_infer_text, tokenizer_name
                )
                if rouge_scores:
                    rouge_scores_out = (
                        f"**ROUGE-1 F1:** {rouge_scores['rouge-1']:.4f}  \n"
                        f"**ROUGE-2 F1:** {rouge_scores['rouge-2']:.4f}  \n"
                        f"**ROUGE-L F1:** {rouge_scores['rouge-l']:.4f}"
                    )
                else:
                    rouge_scores_out = "Could not calculate ROUGE scores."

            else:
                msg = f"fname '{fname}' not found in the uploaded file."
                summary_infer_out = msg
        except Exception as e:
            tb_str = traceback.format_exc()
            msg = f"<pre>Error reading or processing the file: {e}\n\nTraceback:\n{tb_str}</pre>"
            summary_infer_out = msg
            rouge_scores_out = "Error calculating scores."
            
    return dialogue_orig, summary_orig_out, topic_orig, summary_infer_out, rouge_scores_out

def get_val_sample_by_fname(fname, inference_file, tokenizer_name):
    if fname not in val_df['fname'].values:
        msg = f"fname '{fname}' not found."
        return -1, fname, msg, msg, msg, msg, "N/A"
    
    index = val_df[val_df['fname'] == fname].index[0]
    d_orig, s_orig, t_orig, s_infer, scores = get_validation_data(fname, inference_file, tokenizer_name)
    return index, fname, d_orig, s_orig, t_orig, s_infer, scores

def change_val_sample(index_change, current_index, inference_file, tokenizer_name):
    current_index = int(current_index)
    new_index = current_index + index_change
    
    if not (0 <= new_index < len(val_df)):
        new_index = current_index

    fname = val_df.iloc[new_index]['fname']
    d_orig, s_orig, t_orig, s_infer, scores = get_validation_data(fname, inference_file, tokenizer_name)
    return new_index, fname, d_orig, s_orig, t_orig, s_infer, scores


# --- Gradio Interface ---

with gr.Blocks() as demo:
    gr.Markdown("# Data Explorer for Dialogue Summarization")

    with gr.Tab("Explore by Sample (fname)"):
        # State to hold the current index
        current_index = gr.State(0)
        
        with gr.Row():
            split_select = gr.Radio(["train", "val"], label="Dataset", value="train")
            with gr.Column():
                prev_btn = gr.Button("Previous")
                next_btn = gr.Button("Next")
            fname_out = gr.Textbox(label="fname", interactive=True)
            search_btn = gr.Button("Search")    
        
        with gr.Row():
            with gr.Column():
                origin_out = gr.Markdown(label="original dataset")
                dialogue_out = gr.Markdown(label="Dialogue")
                summary_out = gr.Markdown(label="Summary")
                ner_out = gr.Markdown(label='NER')
                topic_out = gr.Markdown(label="Topic")
            with gr.Column():
                with gr.Row():
                    UPSTAGE_API_KEY = gr.Textbox(label="Your Upstage API Key", interactive=True)
                    solar_model_select = gr.Dropdown(["solar-pro2", "solar-pro", "solar-mini"], label="Solar Model", value="solar-pro2")
                    solar_btn = gr.Button("Request")
                llm_summary_desc = gr.Markdown(label="desc1")
                llm_summary_out = gr.Markdown(label="LLM Summary")
                llm_ner_desc = gr.Markdown(label="desc2")
                llm_ner_out = gr.Markdown(label='LLM_NER')
                re_summary_desc = gr.Markdown(label="desc3")
                re_summary_out = gr.Markdown(label="LLM Retranslate Summary")
                re_topic_desc = gr.Markdown(label="desc4")
                re_topic_out = gr.Markdown(label="LLM Retranslate Topic")
                re_dialogue_desc = gr.Markdown(label="desc5")
                re_dialogue_out = gr.Markdown(label="LLM Retranslate dialogue")

        # Initial load
        initial_fname, initial_dialogue, initial_summary, initial_ner, initial_topic, initial_llm_summary_out,initial_llm_ner_out,initial_re_summary_out,initial_re_topic_out,initial_re_dialogue_out, _ = get_sample_by_index("train", 0)
        
        # Set initial values
        demo.load(
            lambda:
                (
                "### Original Dataset", 
                'Solar Summary result','Solar NER result','Back-translated(ko>en>ko) Solar Summary','Back-translated(ko>en>ko) Solar topic','Back-translated(ko>en>ko) Dialogue',
                initial_fname, f"```\n{initial_dialogue}\n```", f"```\n{initial_summary}\n```", f"```\n{initial_ner}\n```", f"```\n{initial_topic}\n```",
                f"```\n{initial_llm_summary_out}\n```", f"```\n{initial_llm_ner_out}\n```", f"```\n{initial_re_summary_out}\n```", f"```\n{initial_re_topic_out}\n```",f"```\n{initial_re_dialogue_out}\n```"
                ),
            inputs=None,
            outputs=[
                origin_out, 
                llm_summary_desc,llm_ner_desc,re_summary_desc,re_topic_desc,re_dialogue_desc,
                fname_out, dialogue_out, summary_out, ner_out, topic_out,
                llm_summary_out,llm_ner_out,re_summary_out,re_topic_out,re_dialogue_out
            ]
        )

        # Event handlers for Block 1
        search_btn.click(
            get_sample_by_fname,
            inputs=[split_select, fname_out],
            outputs=[current_index, fname_out, dialogue_out, summary_out, ner_out, topic_out, llm_summary_out,llm_ner_out,re_summary_out,re_topic_out,re_dialogue_out]
        )
        next_btn.click(
            next_sample,
            inputs=[split_select, current_index],
            outputs=[current_index, fname_out, dialogue_out, summary_out, ner_out, topic_out, llm_summary_out,llm_ner_out,re_summary_out,re_topic_out,re_dialogue_out]
        )
        prev_btn.click(
            prev_sample,
            inputs=[split_select, current_index],
            outputs=[current_index, fname_out, dialogue_out, summary_out, ner_out, topic_out, llm_summary_out,llm_ner_out,re_summary_out,re_topic_out,re_dialogue_out]
        )
        split_select.change(
            reset_index_on_split_change,
            inputs=[split_select],
            outputs=[current_index, fname_out, dialogue_out, summary_out, ner_out, topic_out, llm_summary_out,llm_ner_out,re_summary_out,re_topic_out,re_dialogue_out]
        )
        solar_btn.click(
            request_solar_api,
            inputs=[UPSTAGE_API_KEY, solar_model_select, split_select, current_index],
            outputs=[llm_summary_out, llm_ner_out, re_summary_out, re_topic_out, re_dialogue_out]
        )


    with gr.Tab("Explore by Topic"):
        with gr.Row():
            topic_select = gr.Dropdown(topics, label="Select Topic", value=topics[0])
            prev_topic_btn = gr.Button("Previous Topic")
            next_topic_btn = gr.Button("Next Topic")

        with gr.Row():
            train_topic_display = gr.Markdown(label="Train Data")
            val_topic_display = gr.Markdown(label="Validation Data")

        # Initial load for Block 2
        demo.load(
            lambda: get_topic_data(topics[0]),
            inputs=None,
            outputs=[train_topic_display, val_topic_display]
        )

        # Event handlers for Block 2
        topic_select.change(
            update_topic_display_by_name,
            inputs=[topic_select],
            outputs=[train_topic_display, val_topic_display]
        )
        next_topic_btn.click(
            lambda current_topic: change_topic(1, current_topic),
            inputs=[topic_select],
            outputs=[topic_select, train_topic_display, val_topic_display]
        )
        prev_topic_btn.click(
            lambda current_topic: change_topic(-1, current_topic),
            inputs=[topic_select],
            outputs=[topic_select, train_topic_display, val_topic_display]
        )
    
    with gr.Tab("Explore Validation Inference"):
        # State
        current_index_2 = gr.State(0)
        
        with gr.Row():
            inference_file_input = gr.File(label="Upload Inferred Validation CSV", value="./data/dev_sample.csv")
            tokenizer_input = gr.Textbox(label="Tokenizer", value="digit82/kobart-summarization")
        
        with gr.Row():
            prev_btn_2 = gr.Button("Previous")
            next_btn_2 = gr.Button("Next")
            fname_out_2 = gr.Textbox(label="fname", interactive=True)
            search_btn_2 = gr.Button("Search")

        with gr.Row():
            val_dialogue_out = gr.Markdown(label="Dialogue")
            val_topic_out = gr.Markdown(label="Topic")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Original Validation Data")
                val_summary_out = gr.HTML(label="Summary")
            with gr.Column():
                gr.Markdown("### Inference Data")
                val_inference_summary_out = gr.HTML(label="Summary")
        
        rouge_scores_out = gr.Markdown(label="ROUGE F1 Scores")

        # Outputs for the new tab
        outputs_2 = [
            current_index_2, fname_out_2,
            val_dialogue_out, val_summary_out, val_topic_out,
            val_inference_summary_out, rouge_scores_out
        ]

        # Event handlers
        search_btn_2.click(
            get_val_sample_by_fname,
            inputs=[fname_out_2, inference_file_input, tokenizer_input],
            outputs=outputs_2
        )
        next_btn_2.click(
            lambda idx, file, tok: change_val_sample(1, idx, file, tok),
            inputs=[current_index_2, inference_file_input, tokenizer_input],
            outputs=outputs_2
        )
        prev_btn_2.click(
            lambda idx, file, tok: change_val_sample(-1, idx, file, tok),
            inputs=[current_index_2, inference_file_input, tokenizer_input],
            outputs=outputs_2
        )
        
        # Trigger update when file or tokenizer changes, using the current fname
        inference_file_input.change(
            get_val_sample_by_fname,
            inputs=[fname_out_2, inference_file_input, tokenizer_input],
            outputs=outputs_2
        )
        tokenizer_input.change(
            get_val_sample_by_fname,
            inputs=[fname_out_2, inference_file_input, tokenizer_input],
            outputs=outputs_2
        )

        # Initial load for the tab
        def initial_load_tab3():
            fname = val_df.iloc[0]['fname']
            d_orig, s_orig, t_orig, s_infer, scores = get_validation_data(fname, None, "digit82/kobart-summarization")
            return 0, fname, d_orig, s_orig, t_orig, s_infer, scores

        demo.load(initial_load_tab3, None, outputs_2)


if __name__ == "__main__":
    # To run this script, use the command:
    # python src/app/app_gradio.py
    demo.launch()
