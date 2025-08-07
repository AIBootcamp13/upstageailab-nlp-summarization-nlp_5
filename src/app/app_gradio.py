import gradio as gr
import pandas as pd
import os
import re
from dotenv import load_dotenv


# --- Configuration ---
# Assuming the script is run from the root of the project
load_dotenv()
PROJECT_DIR = "/mnt/c/SKH/ai_lab_13/projects/nlp-text-summarization/song"
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
DEV_CSV = os.path.join(DATA_DIR, "dev.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")

# --- Load Data ---
try:
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(DEV_CSV)
    test_df = pd.read_csv(TEST_CSV)
    llm_train_df = None
    llm_val_df = None
    if os.path.exists(os.path.join(DATA_DIR, "train_solar_results.csv")):
        llm_train_df = pd.read_csv(os.path.join(DATA_DIR, "train_solar_results.csv"))
    if os.path.exists(os.path.join(DATA_DIR, "val_solar_results.csv")):
        llm_val_df = pd.read_csv(os.path.join(DATA_DIR, "val_solar_results.csv"))

except FileNotFoundError:
    print(f"⚠️  Error: Make sure '{TRAIN_CSV}' and '{DEV_CSV}' exist.")
    # Create dummy dataframes for development if files are not found
    train_df = pd.DataFrame({
        'fname': [f'train_{i}' for i in range(5)],
        'dialogue': [f'Train dialogue {i}' for i in range(5)],
        'summary': [f'Train summary {i}' for i in range(5)],
        'topic': [f'Topic_{i%2}' for i in range(5)]
    })
    val_df = pd.DataFrame({
        'fname': [f'val_{i}' for i in range(3)],
        'dialogue': [f'Val dialogue {i}' for i in range(3)],
        'summary': [f'Val summary {i}' for i in range(3)],
        'topic': [f'Topic_{i%2}' for i in range(3)]
    })

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

        try:
            llm_sample = llm_df.iloc[index]
            if 'ner_solar' not in llm_sample.columns: # not implemented yet.
                llm_sample['ner_solar'] = 'N/A'
        except:
            llm_sample = {'dialogue_en2ko': 'N/A','re_summary_solar': 'N/A','ner_solar': 'N/A','topic_solar': 'N/A'}

        return (sample['fname'], sample['dialogue'], sample['summary'], ner, sample['topic'],
                llm_sample['dialogue_en2ko'], llm_sample['re_summary_solar'], llm_sample['ner_solar'], llm_sample['topic_solar'], index)
    return "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", index

def update_outputs(dataset_name, index):
    fname, dialogue, summary, ner, topic, llm_dialogue, llm_summary, llm_ner, llm_topic, _ = get_sample_by_index(dataset_name, index)
    return fname, f"```\n{dialogue}\n```", f"```\n{summary}\n```", f"```\n{ner}\n```", f"```\n{topic}\n```", f"```\n{llm_dialogue}\n```", f"```\n{llm_summary}\n```", f"```\n{llm_ner}\n```", f"```\n{llm_topic}\n```"

def next_sample(dataset_name, index):
    df = train_df if dataset_name == "train" else val_df
    new_index = min(int(index) + 1, len(df) - 1)
    fname, dialogue, summary, ner, topic, llm_dialogue, llm_summary, llm_ner, llm_topic, _ = get_sample_by_index(dataset_name, new_index)
    return new_index, fname, f"```\n{dialogue}\n```", f"```\n{summary}\n```", f"```\n{ner}\n```", f"```\n{topic}\n```", f"```\n{llm_dialogue}\n```", f"```\n{llm_summary}\n```", f"```\n{llm_ner}\n```", f"```\n{llm_topic}\n```"

def prev_sample(dataset_name, index):
    new_index = max(int(index) - 1, 0)
    fname, dialogue, summary, ner, topic, llm_dialogue, llm_summary, llm_ner, llm_topic, _ = get_sample_by_index(dataset_name, new_index)
    return new_index, fname, f"```\n{dialogue}\n```", f"```\n{summary}\n```", f"```\n{ner}\n```", f"```\n{topic}\n```", f"```\n{llm_dialogue}\n```", f"```\n{llm_summary}\n```", f"```\n{llm_ner}\n```", f"```\n{llm_topic}\n```"

def reset_index_on_split_change(dataset_name):
    # When the dataset changes, reset the index to 0 and update the display
    fname, dialogue, summary, ner, topic, llm_dialogue, llm_summary, llm_ner, llm_topic, _ = get_sample_by_index(dataset_name, 0)
    return 0, fname, f"```\n{dialogue}\n```", f"```\n{summary}\n```", f"```\n{ner}\n```", f"```\n{topic}\n```", f"```\n{llm_dialogue}\n```", f"```\n{llm_summary}\n```", f"```\n{llm_ner}\n```", f"```\n{llm_topic}\n```"

def get_sample_by_fname(dataset_name, fname):
    df = train_df if dataset_name == "train" else val_df
    
    # Find the index for the given fname
    if fname in df['fname'].values:
        index = df[df['fname'] == fname].index[0]
        # Use the existing function with the found index
        fname, dialogue, summary, ner, topic, llm_dialogue, llm_summary, llm_ner, llm_topic, _ = get_sample_by_index(dataset_name, index)
        return index, fname, f"```\n{dialogue}\n```", f"```\n{summary}\n```", f"```\n{ner}\n```", f"```\n{topic}\n```", f"```\n{llm_dialogue}\n```", f"```\n{llm_summary}\n```", f"```\n{llm_ner}\n```", f"```\n{llm_topic}\n```"
    else:
        # fname not found
        return -1, f"fname '{fname}' not found in {dataset_name} dataset.", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"

# --- Block 2: Explore by Topic ---

# Merge dataframes
train_df['source'] = 'train'
val_df['source'] = 'val'
merged_df = pd.concat([train_df, val_df], ignore_index=True)
topics = sorted(merged_df['topic'].unique().tolist())
topic_df_grouped = merged_df.groupby('topic')

def get_topic_data(topic_name):
    if topic_name not in topic_df_grouped.groups:
        return "Topic not found.", "Topic not found."

    topic_df = topic_df_grouped.get_group(topic_name)
    
    # Clean and split topic name into keywords
    keywords = re.sub(r"[,\\.]", " ", topic_name, flags=re.IGNORECASE)
    keywords = re.sub(r"\s+", " ", topic_name, flags=re.IGNORECASE)
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


# --- Gradio Interface ---

with gr.Blocks() as demo:
    gr.Markdown("# Data Explorer for Dialogue Summarization")

    with gr.Tab("Explore by Sample (fname)"):
        # State to hold the current index
        current_index = gr.State(0)
        
        with gr.Row():
            split_select = gr.Radio(["train", "val"], label="Dataset", value="train")
            prev_btn = gr.Button("Previous")
            next_btn = gr.Button("Next")
        
        with gr.Row():
            fname_out = gr.Textbox(label="fname", interactive=True)
            search_btn = gr.Button("Search")

        with gr.Row():
            origin_out = gr.Markdown(label="original dataset")
            solar_out = gr.Markdown(label="Solar API results")
        with gr.Row():
            dialogue_out = gr.Markdown(label="Dialogue")
            re_dialogue_out = gr.Markdown(label="LLM_Dialogue")
        with gr.Row():
            summary_out = gr.Markdown(label="Summary")
            re_summary_out = gr.Markdown(label="LLM_Summary")
        with gr.Row():
            ner_out = gr.Markdown(label='NER')
            re_ner_out = gr.Markdown(label='LLM_NER')
        with gr.Row():
            topic_out = gr.Markdown(label="Topic")
            re_topic_out = gr.Markdown(label="LLM_Topic")

        # Initial load
        initial_fname, initial_dialogue, initial_summary, initial_ner, initial_topic, initial_llm_dialogue, initial_llm_summary, initial_llm_ner, initial_llm_topic, _ = get_sample_by_index("train", 0)
        
        # Set initial values
        demo.load(
            lambda:
                (
                "### Original Dataset", "### Solar API results",
                initial_fname, f"```\n{initial_dialogue}\n```", f"```\n{initial_summary}\n```", f"```\n{initial_ner}\n```", f"```\n{initial_topic}\n```",
                f"```\n{initial_llm_dialogue}\n```", f"```\n{initial_llm_summary}\n```", f"```\n{initial_llm_ner}\n```", f"```\n{initial_llm_topic}\n```",
                ),
            inputs=None,
            outputs=[
                origin_out, solar_out,
                fname_out, dialogue_out, summary_out, ner_out, topic_out,
                re_dialogue_out, re_summary_out, re_ner_out, re_topic_out
            ]
        )

        # Event handlers for Block 1
        search_btn.click(
            get_sample_by_fname,
            inputs=[split_select, fname_out],
            outputs=[current_index, fname_out, dialogue_out, summary_out, ner_out, topic_out, re_dialogue_out, re_summary_out, re_ner_out, re_topic_out]
        )
        next_btn.click(
            next_sample,
            inputs=[split_select, current_index],
            outputs=[current_index, fname_out, dialogue_out, summary_out, ner_out, topic_out, re_dialogue_out, re_summary_out, re_ner_out, re_topic_out]
        )
        prev_btn.click(
            prev_sample,
            inputs=[split_select, current_index],
            outputs=[current_index, fname_out, dialogue_out, summary_out, ner_out, topic_out, re_dialogue_out, re_summary_out, re_ner_out, re_topic_out]
        )
        split_select.change(
            reset_index_on_split_change,
            inputs=[split_select],
            outputs=[current_index, fname_out, dialogue_out, summary_out, ner_out, topic_out, re_dialogue_out, re_summary_out, re_ner_out, re_topic_out]
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
    
    with gr.Tab("Explore Submission file"):
        
        pass


if __name__ == "__main__":
    # To run this script, use the command:
    # python src/app/app_gradio.py
    demo.launch()