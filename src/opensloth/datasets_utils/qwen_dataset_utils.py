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


def post_process_text(text, tokenizer_name):
    """Post-process text for Qwen models."""
    # Remove thinking tokens for instruct models
    if 'instruct-2507' in tokenizer_name.lower():
        text = text.replace('<think>\n\n</think>\n\n', '')
    return text