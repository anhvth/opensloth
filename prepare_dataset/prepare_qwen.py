

import argparse
import os
from typing import List
from transformers import AutoTokenizer
from speedy_utils import *
import datasets
from utils import train_on_target_text_only

# Add torch and debug function import
import torch
from torch.utils.data import DataLoader
from src.opensloth._debug_dataloader import debug_chat_dataloader_for_training

def main():
    parser = argparse.ArgumentParser(description="Prepare Qwen dataset with tokenization and formatting.")
    parser.add_argument('--tok_name', type=str, default='/data/hf-models/Qwen/Qwen3-30B-A3B-Instruct-2507-FP8/', help='Path to the tokenizer/model directory')
    parser.add_argument('--input_file', '-i',type=str, default='../../TRANSLATE_UI/processed_messages.json', help='Input JSON file with messages')
    parser.add_argument('--num_samples', type=int, default=-1, help='Number of samples to process (use -1 for all)')
    parser.add_argument('--num_proc', type=int, default=8, help='Number of processes for mapping')
    parser.add_argument('--instruction_part', type=str, default='<|im_start|>user\n', help='Instruction part string')
    parser.add_argument('--response_part', type=str, default='<|im_start|>assistant\n', help='Response part string')
    parser.add_argument('--output_dir', '-o',type=str, default='qwen_dataset', help='Output directory for the processed dataset')

    parser.add_argument('--debug_html', type=int, default=0, help='If >0, dump this many samples as HTML to output_dir/debug_samples.html')
    args = parser.parse_args()

    instruction_part = args.instruction_part
    response_part = args.response_part
    tokenizer = AutoTokenizer.from_pretrained(args.tok_name)

    all_messages = load_by_ext(args.input_file, do_memoize=True)
    if args.num_samples > 0:
        all_messages = all_messages[:args.num_samples]

    def process_one(messages):
        text = tokenizer.apply_chat_template(messages['messages'], tokenize=False)
        # for 2507 instruct model 
        if 'Instruct-2507' in args.tok_name:
            text = text.replace('<think>\n\n</think>\n', '')
        input_ids = tokenizer(text)['input_ids']
        labels = train_on_target_text_only(input_ids, tokenizer, instruction_part, response_part)
        return {
            'text': text,
            'input_ids': input_ids,
            'labels': labels,
        }

    data = datasets.Dataset.from_list(all_messages).map(process_one, num_proc=args.num_proc)

    if args.debug_html > 0:
        # Convert HuggingFace dataset to PyTorch format
        data.set_format(type="torch", columns=["input_ids", "labels"])
        dataloader = DataLoader(data, batch_size=1, shuffle=False)
        debug_chat_dataloader_for_training(dataloader, tokenizer, n_example=args.debug_html)

    data.save_to_disk(args.output_dir)

def _escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

if __name__ == "__main__":
    main()