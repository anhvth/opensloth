README — GRPO-only Implementation for Qwen3-4B-Base (Unsloth)

Purpose

Demonstrate pure GRPO training with Unsloth on Qwen3-4B-Base without SFT.
Covers:
	•	Model loading with LoRA
	•	Chat template for reasoning
	•	Dataset preparation (Open-R1 DAPO Math)
	•	Reward functions (format + correctness)
	•	Prompt length filtering
	•	GRPO training config + run
	•	LoRA save + inference

⸻

1. Load Model + LoRA Config

from unsloth import FastLanguageModel
import torch

max_seq_length = 2048
lora_rank = 32
gpu_util = 0.7

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-4B-Base",
    max_seq_length = max_seq_length,
    load_in_4bit = False,
    fast_inference = True,
    max_lora_rank = lora_rank,
    gpu_memory_utilization = gpu_util,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank,
    target_modules = [
        "q_proj","k_proj","v_proj","o_proj",
        "gate_proj","up_proj","down_proj",
    ],
    lora_alpha = lora_rank * 2,
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)


⸻

2. Define Chat Template

reasoning_start = "<start_working_out>"
reasoning_end   = "<end_working_out>"
solution_start  = "<SOLUTION>"
solution_end    = "</SOLUTION>"

system_prompt = f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""

chat_template = (
    "{% if messages[0]['role'] == 'system' %}"
        "{{ messages[0]['content'] + eos_token }}"
        "{% set loop_messages = messages[1:] %}"
    "{% else %}"
        "{{ '" + system_prompt + "' + eos_token }}"
        "{% set loop_messages = messages %}"
    "{% endif %}"
    "{% for message in loop_messages %}"
        "{% if message['role'] == 'user' %}"
            "{{ message['content'] }}"
        "{% elif message['role'] == 'assistant' %}"
            "{{ message['content'] + eos_token }}"
        "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '" + reasoning_start + "' }}{% endif %}"
)
tokenizer.chat_template = chat_template


⸻

3. Dataset Preparation

from datasets import load_dataset
dataset = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split="train")

def extract_hash_answer(text): return text

dataset = dataset.map(lambda x: {
    "prompt": [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": x["prompt"]},
    ],
    "answer": extract_hash_answer(x["solution"]),
})


⸻

4. Reward Functions

import re
solution_end_regex = r"</SOLUTION>[\s]{0,}(?:" + re.escape(tokenizer.eos_token) + ")?"
match_format = re.compile(
    rf"{reasoning_end}.*?{solution_start}(.+?){solution_end_regex}[\s]{{0,}}$",
    flags=re.MULTILINE|re.DOTALL
)

def match_format_exactly(completions, **kwargs):
    return [3.0 if match_format.search(c[0]["content"]) else 0.0 for c in completions]

def match_format_approximately(completions, **kwargs):
    scores = []
    for c in completions:
        r = c[0]["content"]
        s = 0
        s += 0.5 if r.count(reasoning_end)  == 1 else -1.0
        s += 0.5 if r.count(solution_start) == 1 else -1.0
        s += 0.5 if r.count(solution_end)   == 1 else -1.0
        scores.append(s)
    return scores

def check_answer(prompts, completions, answer, **kwargs):
    responses = [c[0]["content"] for c in completions]
    extracted = [m.group(1) if (m:=match_format.search(r)) else None for r in responses]
    scores = []
    for guess, true in zip(extracted, answer):
        if guess is None: scores.append(-2.0); continue
        if guess == true: scores.append(5.0)
        elif guess.strip() == true.strip(): scores.append(3.5)
        else:
            try:
                ratio = float(guess)/float(true)
                if 0.9 <= ratio <= 1.1: scores.append(2.0)
                elif 0.8 <= ratio <= 1.2: scores.append(1.5)
                else: scores.append(-2.5)
            except: scores.append(-4.5)
    return scores

match_numbers = re.compile(
    solution_start + r".*?[\s]{0,}([-]?[\d\.\,]{1,})",
    flags=re.MULTILINE|re.DOTALL
)

PRINTED_TIMES = 0
PRINT_EVERY_STEPS = 5
def check_numbers(prompts, completions, answer, **kwargs):
    global PRINTED_TIMES
    responses = [c[0]["content"] for c in completions]
    extracted = [m.group(1) if (m:=match_numbers.search(r)) else None for r in responses]
    if PRINTED_TIMES % PRINT_EVERY_STEPS == 0:
        print(f"Q:\n{prompts[0][-1]['content']}\nA:\n{answer[0]}\nResp:\n{responses[0]}\nExtracted:\n{extracted[0]}")
    PRINTED_TIMES += 1
    scores = []
    for guess, true in zip(extracted, answer):
        if guess is None: scores.append(-2.5); continue
        try:
            if float(guess.replace(",","")) == float(true.strip()):
                scores.append(3.5)
            else: scores.append(-1.5)
        except: scores.append(0)
    return scores


⸻

5. Filter Long Prompts

tokenized = dataset.map(
    lambda x: {"tokens": tokenizer.apply_chat_template(x["prompt"], add_generation_prompt=True, tokenize=True)},
    batched=True,
)
tokenized = tokenized.map(lambda x: {"L": len(x["tokens"])})
import numpy as np
maximum_length = int(np.quantile(tokenized["L"], 0.9))
dataset = dataset.select(np.where(np.array(tokenized["L"]) <= maximum_length)[0])


⸻

6. GRPO Trainer

from vllm import SamplingParams
vllm_sampling_params = SamplingParams(
    min_p = 0.1,
    top_p = 1.0,
    top_k = -1,
    seed = 3407,
    stop = [tokenizer.eos_token],
    include_stop_str_in_output = True,
)

max_prompt_length     = maximum_length + 1
max_completion_length = max_seq_length - max_prompt_length

from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    vllm_sampling_params      = vllm_sampling_params,
    temperature               = 1.0,
    learning_rate             = 5e-6,
    weight_decay              = 0.01,
    warmup_ratio              = 0.1,
    lr_scheduler_type         = "linear",
    optim                     = "adamw_8bit",
    logging_steps             = 1,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1,
    num_generations           = 4,
    max_prompt_length         = max_prompt_length,
    max_completion_length     = max_completion_length,
    max_steps                 = 100,
    save_steps                = 100,
    report_to                 = "none",
    output_dir                = "outputs",
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        match_format_exactly,
        match_format_approximately,
        check_answer,
        check_numbers,
    ],
    args = training_args,
    train_dataset = dataset,
)
trainer.train()


⸻

7. Save & Validate LoRA

model.save_lora("grpo_saved_lora")

from safetensors import safe_open
with safe_open("grpo_saved_lora/adapter_model.safetensors", framework="pt") as f:
    for k in f.keys():
        t = f.get_tensor(k)
        assert (t == 0).sum().item() != t.numel()


⸻

8. Inference

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user",   "content": "What is the sqrt of 101?"},
]
text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

sampling_params = SamplingParams(temperature=1.0, top_k=50, max_tokens=1024)

output = model.fast_generate(
    text,
    sampling_params = sampling_params,
    lora_request = model.load_lora("grpo_saved_lora"),
)[0].outputs[0].text

print(output)


⸻

If you want, I can now also make you a flow diagram of this GRPO-only pipeline so the AI can learn the high-level logic visually. This would make it easier for an LLM to reason about the sequence of operations. Would you like me to do that?