from typing import List
from transformers import AutoTokenizer
from speedy_utils import *
import datasets
import warnings

def train_on_target_text_only(
    ids: List[int],
    tokenizer,
    instruction_part: str,
    response_part: str,
) -> List[int]:
    """
    Keep tokens that belong to the *assistant response* and mask everything else
    with -100 so the language-model loss is only computed on answers.

    Args
    ----
    ids : List[int]
        A single, already-encoded sequence (e.g. conversation turned into token-ids).
    tokenizer : PreTrainedTokenizerBase
        The tokenizer that produced `ids`.
    instruction_part : str
        Text that marks the *start of the user/instruction turn*
        (e.g. "<|user|>" or "<|im_start|>user").
    response_part : str
        Text that marks the *start of the assistant/response turn*
        (e.g. "<|assistant|>" or "<|im_start|>assistant").

    Returns
    -------
    List[int]
        Same length as `ids`, but every token **outside** assistant answers
        is replaced by -100.
    """
    # Tokenize the two markers (no special tokens added)
    q_tokens = tokenizer(instruction_part, add_special_tokens=False).input_ids
    a_tokens = tokenizer(response_part,    add_special_tokens=False).input_ids
    len_q, len_a = len(q_tokens), len(a_tokens)

    labels = [-100] * len(ids)          # start fully masked
    pos = 0

    found_any = False
    while pos < len(ids):
        # --- find the next assistant marker -------------------------------
        try:
            a_start = ids.index(a_tokens[0], pos)          # first-token match
        except ValueError:                                  # not found
            break
        # verify full subsequence match
        if ids[a_start : a_start + len_a] != a_tokens:
            pos = a_start + 1
            continue

        # region *after* the marker is the answer text
        span_start = a_start + len_a                       # exclude the marker

        # --- look for the next user/instruction marker --------------------
        try:
            q_start = ids.index(q_tokens[0], span_start)
            # ensure full subsequence match;
            # if it isn’t, keep searching forward
            while ids[q_start : q_start + len_q] != q_tokens:
                q_start = ids.index(q_tokens[0], q_start + 1)
            span_end = q_start                              # stop *before* user marker
        except ValueError:                                  # no further user turn
            span_end = len(ids)

        # --- copy answer tokens into labels -------------------------------
        if span_end > span_start:
            found_any = True
            labels[span_start : span_end] = ids[span_start : span_end]

        # continue search after this span
        pos = span_end

    if not found_any:
        warnings.warn("No assistant response found to train on in this sequence.")

    return labels
    