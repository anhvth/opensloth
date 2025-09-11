"""
Utility functions for Qwen dataset preparation.
"""

import json
import os
import warnings
from typing import cast

from tabulate import tabulate
from transformers import AutoTokenizer


def print_config_table(config_dict):
    """Print configuration as a formatted table."""
    table_data = [[key, value] for key, value in config_dict.items()]
    print("\n" + "="*60)
    print("DATASET PREPARATION CONFIGURATION")
    print("="*60)
    print(tabulate(table_data, headers=["Parameter", "Value"], tablefmt="grid"))
    print("="*60 + "\n")


def train_on_target_text_only(ids, tokenizer, instruction_part, response_part):
    """
    Keep tokens that belong to the assistant response and mask everything else
    with -100 so the language-model loss is only computed on answers.
    """
    # Tokenize the two markers (no special tokens added)
    q_tokens = tokenizer(instruction_part, add_special_tokens=False).input_ids
    a_tokens = tokenizer(response_part, add_special_tokens=False).input_ids
    len_q, len_a = len(q_tokens), len(a_tokens)

    labels = [-100] * len(ids)  # start fully masked
    pos = 0

    found_any = False
    while pos < len(ids):
        # Find the next assistant marker
        try:
            a_start = ids.index(a_tokens[0], pos)  # first-token match
        except ValueError:  # not found
            break
        # verify full subsequence match
        if ids[a_start : a_start + len_a] != a_tokens:
            pos = a_start + 1
            continue

        # region after the marker is the answer text
        span_start = a_start + len_a  # exclude the marker

        # Look for the next user/instruction marker
        try:
            q_start = ids.index(q_tokens[0], span_start)
            # ensure full subsequence match
            while ids[q_start : q_start + len_q] != q_tokens:
                q_start = ids.index(q_tokens[0], q_start + 1)
            span_end = q_start  # stop before user marker
        except ValueError:  # no further user turn
            span_end = len(ids)

        # Copy answer tokens into labels
        if span_end > span_start:
            found_any = True
            labels[span_start : span_end] = ids[span_start : span_end]

        # continue search after this span
        pos = span_end

    if not found_any:
        warnings.warn("No assistant response found to train on in this sequence.", stacklevel=2)

    return labels


def load_local_file(filepath):
    """Load data from local file based on extension."""
    try:
        from speedy_utils import load_by_ext
        return load_by_ext(filepath, do_memoize=True)
    except ImportError:
        # Fallback implementation for common formats
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            if filepath.endswith('.json'):
                return json.load(f)
            elif filepath.endswith('.jsonl'):
                return [json.loads(line) for line in f]
            else:
                raise ValueError(f"Unsupported file format: {filepath}")


def sanitize_name(name):
    """Sanitize a name for use in file paths."""
    name = name.split("/")[-1] if "/" in name else name
    name = name.replace(" ", "_").replace("|", "_")
    return name[:32]  # limit length for sanity


def compute_output_dir(args):
    """Auto-compute output directory if not specified."""
    if args.output_dir:
        return args.output_dir

    model_name = sanitize_name(args.tokenizer_name)
    dataset_name = sanitize_name(args.dataset_name)
    split = sanitize_name(args.split)

    if args.num_samples > 0:
        count = f"n{args.num_samples}"
    elif args.debug > 0:
        count = f"debug{args.debug}"
    else:
        count = "all"

    # Include max sequence length in the directory name
    length_tag = f"l{args.max_seq_length}"

    return f"{model_name}-{dataset_name}-{split}-{count}-{length_tag}-processed"


def compute_output_dir_from_args(tokenizer_name, dataset_name, split, num_samples, max_seq_length, debug=0):
    """Compute output directory name based on arguments."""
    # Sanitize model name
    model_name = tokenizer_name.split('/')[-1]  # Take last part
    if model_name.startswith('unsloth/'):
        model_name = model_name[8:]
    
    # Sanitize dataset name
    dataset_name_clean = dataset_name.replace('/', '-').replace('\\', '-')
    
    split_clean = split
    num_samples_str = 'all' if num_samples <= 0 else str(num_samples)
    if debug > 0:
        num_samples_str = f"debug{debug}"
    max_seq_length_str = str(max_seq_length)
    
    dir_name = f"{model_name}-{dataset_name_clean}-{split_clean}-n{num_samples_str}-l{max_seq_length_str}-processed"
    return dir_name


def post_process_text(text, tokenizer_name):
    """Post-process text for Qwen models."""
    # Remove thinking tokens for instruct models
    if 'instruct-2507' in tokenizer_name.lower():
        text = text.replace('<think>\n\n</think>\n\n', '')
    return text


def save_dataset_metadata(output_dir, config):
    """Save metadata for reproducibility and GUI auto-match."""
    import hashlib
    
    meta = {
        "config": {
            "tokenizer_name": config["tokenizer_name"],
            "chat_template": config["chat_template"],
            "dataset_name": config["dataset_name"],
            "split": config["split"],
            "max_seq_length": config["max_seq_length"],
            "num_samples": config["num_samples"],
            "train_on_target_only": config["train_on_target_only"],
            "instruction_part": config["instruction_part"] if config["train_on_target_only"] else None,
            "response_part": config["response_part"] if config["train_on_target_only"] else None,
            "preparer_class": "QwenDatasetPreparer",
        },
        "size": config.get("dataset_size", 0),
    }
    payload = json.dumps(meta["config"], sort_keys=True, ensure_ascii=False)
    meta["config_hash"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def save_dataset_config(output_dir, config):
    """Save complete config for debugging and reuse."""
    complete_config = {
        'tokenizer_name': config['tokenizer_name'],
        'chat_template': config['chat_template'],
        'dataset_name': config['dataset_name'],
        'split': config['split'],
        'num_samples': config['num_samples'],
        'num_proc': config['num_proc'],
        'gpus': config['gpus'],
        'output_dir': output_dir,
        'train_on_target_only': config['train_on_target_only'],
        'instruction_part': config['instruction_part'] if config['train_on_target_only'] else None,
        'response_part': config['response_part'] if config['train_on_target_only'] else None,
        'max_seq_length': config['max_seq_length'],
        'debug': config['debug'],
        'hf_token': None,
        'num_shards': config['gpus'],
        'model_family': 'Qwen'
    }

    with open(os.path.join(output_dir, "dataset_config.json"), "w", encoding="utf-8") as f:
        json.dump(complete_config, f, ensure_ascii=False, indent=2)


def save_training_config(output_dir, config, training_config_template):
    """Save complete training configuration for train.py with schema validation."""
    training_config = training_config_template.copy()
    training_config["opensloth_config"]["data_cache_path"] = output_dir
    training_config["dataset_prep_config"] = config.copy()

    with open(os.path.join(output_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(training_config, f, ensure_ascii=False, indent=2)


def create_formatting_func(tokenizer, train_on_target_only, instruction_part, response_part, tokenizer_name=None):
    """Create a formatting function for dataset mapping."""
    if tokenizer_name is None:
        tokenizer_name = getattr(tokenizer, 'name_or_path', '')
        
    def formatting_prompts_func(examples):
        if "conversations" in examples:
            # HuggingFace format with conversations
            convos = examples["conversations"]
            texts = []
            missing_assistant = 0
            for convo in convos:
                text = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
                text = text.removeprefix('<bos>')
                if train_on_target_only and response_part not in text:
                    missing_assistant += 1
                texts.append(text)
            return {"text": texts, "missing_assistant": [missing_assistant] * len(texts)}
        elif "messages" in examples:
            # Local format with messages
            all_messages = examples["messages"]
            texts = []
            missing_assistant = 0
            for messages in all_messages:
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                text = post_process_text(text, tokenizer_name)
                if train_on_target_only and response_part not in text:
                    missing_assistant += 1
                texts.append(text)
            return {"text": texts, "missing_assistant": [missing_assistant] * len(texts)}
        else:
            raise ValueError("Dataset must have either 'conversations' or 'messages' column")
    
    return formatting_prompts_func


def create_processing_func(tokenizer, train_on_target_only, instruction_part, response_part):
    """Create a processing function for tokenization and labeling."""
    if train_on_target_only:
        def process_one(example):
            text = example["text"]
            input_ids = tokenizer(text)["input_ids"]
            labels = train_on_target_text_only(
                input_ids, tokenizer, 
                instruction_part, response_part
            )
            return {
                "text": text,
                "input_ids": input_ids,
                "labels": labels,
                "all_masked": int(all(l == -100 for l in labels)),
            }
    else:
        def process_one(example):
            text = example["text"]
            input_ids = tokenizer(text)["input_ids"]
            return {
                "text": text,
                "input_ids": input_ids,
                "labels": input_ids.copy(),  # Use same as input_ids for full training
                "all_masked": 0,
            }
    
    return process_one


def create_filter_func(max_seq_length, train_on_target_only):
    """Create a filter function for dataset cleaning."""
    def should_keep(example):
        # Filter out examples that are too long
        if len(example["input_ids"]) > max_seq_length:
            return False
        
        # Filter out examples with no training labels (all -100)
        if train_on_target_only and "all_masked" in example:
            if example["all_masked"] == 1:  # All labels are -100
                return False
        
        return True
    
    return should_keep