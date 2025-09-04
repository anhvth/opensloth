# -*- coding: utf-8 -*-
"""
Simplified GRPO Trainer Setup for GSM8K
Based on Unsloth GRPO notebook with minimal dependencies
"""
import os
import re
import torch


def get_trainer():
    """
    Create a GRPO trainer for GSM8K mathematical reasoning.
    
    Args:
        device_id: CUDA device ID (default: 0)
        
    Returns:
        Configured GRPOTrainer instance
    """
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import GRPOConfig, GRPOTrainer
    from vllm import SamplingParams
    
    
    # ==================== Model Setup ====================
    max_seq_length = 4096*2
    lora_rank = 16
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="/data/hf-models/Qwen/Qwen3-1.7B-Base",  # Using 0.6B as per project
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        fast_inference=os.getenv("FAST_INFERENCE", "1") == "1",
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.7,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_rank * 2,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    
    # ==================== Template Setup ====================
    reasoning_start = "<start_working_out>"
    reasoning_end = "<end_working_out>"
    solution_start = "<SOLUTION>"
    solution_end = "</SOLUTION>"
    
    system_prompt = (
        f"You are given a problem. "
        f"Think about the problem and provide your working out. "
        f"Place it between {reasoning_start} and {reasoning_end}. "
        f"Then, provide your solution between {solution_start} and {solution_end}."
    )
    
    # Setup chat template
    chat_template = (
        "{% if messages[0]['role'] == 'system' %}"
            "{{ messages[0]['content'] + eos_token }}"
            "{% set loop_messages = messages[1:] %}"
        "{% else %}"
            f"{{ '{system_prompt}' + eos_token }}"
            "{% set loop_messages = messages %}"
        "{% endif %}"
        "{% for message in loop_messages %}"
            "{% if message['role'] == 'user' %}"
                "{{ message['content'] }}"
            "{% elif message['role'] == 'assistant' %}"
                "{{ message['content'] + eos_token }}"
            "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
            f"{{ '{reasoning_start}' }}"
        "{% endif %}"
    )
    
    tokenizer.chat_template = chat_template
    
    # ==================== Dataset Setup ====================
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    
    def extract_answer(text):
        """Extract answer after #### marker."""
        if "####" not in text:
            return text.strip()
        return text.split("####")[-1].strip()
    
    def format_prompt(example):
        """Format GSM8K example for inference."""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["question"]},
        ]
    
    dataset = dataset.map(lambda x: {
        "prompt": format_prompt(x),
        "answer": extract_answer(x["answer"]),
    })
    
    # ==================== Reward Functions ====================
    
    # Regex patterns for reward functions
    solution_end_regex = r"</SOLUTION>[\s]{0,}" + f"(?:{re.escape(tokenizer.eos_token)})?"
    
    match_format = re.compile(
        rf"{reasoning_end}.*?"
        rf"{solution_start}(.+?){solution_end_regex}"
        rf"[\s]{{0,}}$",
        flags=re.MULTILINE | re.DOTALL
    )
    
    match_numbers = re.compile(
        solution_start + r".*?[\s]{0,}([-]?[\d\.\,]{1,})",
        flags=re.MULTILINE | re.DOTALL
    )
    
    global PRINTED_TIMES, PRINT_EVERY_STEPS
    PRINTED_TIMES = 0
    PRINT_EVERY_STEPS = 5
    
    def match_format_exactly(completions, **kwargs):
        """Reward for exact format matching."""
        scores = []
        for completion in completions:
            score = 0
            response = completion[0]["content"]
            if match_format.search(response) is not None:
                score += 3.0
            scores.append(score)
        return scores
    
    def match_format_approximately(completions, **kwargs):
        """Reward for partial format matching."""
        scores = []
        for completion in completions:
            score = 0
            response = completion[0]["content"]
            
            score += 0.5 if response.count(reasoning_end) == 1 else -1.0
            score += 0.5 if response.count(solution_start) == 1 else -1.0
            score += 0.5 if response.count(solution_end) == 1 else -1.0
            scores.append(score)
        return scores
    
    def check_answer(prompts, completions, answer, **kwargs):
        """Reward for correct answer extraction."""
        responses = [completion[0]["content"] for completion in completions]
        
        extracted_responses = [
            guess.group(1) if (guess := match_format.search(r)) is not None else None
            for r in responses
        ]
        
        scores = []
        for guess, true_answer in zip(extracted_responses, answer):
            score = 0
            if guess is None:
                scores.append(-2.0)
                continue
                
            if guess == true_answer:
                score += 5.0
            elif guess.strip() == true_answer.strip():
                score += 3.5
            else:
                try:
                    ratio = float(guess) / float(true_answer)
                    if 0.9 <= ratio <= 1.1:
                        score += 2.0
                    elif 0.8 <= ratio <= 1.2:
                        score += 1.5
                    else:
                        score -= 2.5
                except Exception:
                    score -= 4.5
            scores.append(score)
        return scores
    
    def check_numbers(prompts, completions, answer, **kwargs):
        """Reward for numerical answer matching with debug output."""
        global PRINTED_TIMES, PRINT_EVERY_STEPS
        
        question = prompts[0][-1]["content"]
        responses = [completion[0]["content"] for completion in completions]
        
        extracted_responses = [
            guess.group(1) if (guess := match_numbers.search(r)) is not None else None
            for r in responses
        ]
        
        # Print debug info every few steps
        if PRINTED_TIMES % PRINT_EVERY_STEPS == 0:
            print('*' * 50)
            print(f"Question:\n{question}")
            print(f"Answer:\n{answer[0]}")
            print(f"Response:\n{responses[0]}")
            print(f"Extracted:\n{extracted_responses[0]}")
            print('*' * 50)
        PRINTED_TIMES += 1
        
        scores = []
        for guess, true_answer in zip(extracted_responses, answer):
            if guess is None:
                scores.append(-2.5)
                continue
            
            try:
                true_num = float(str(true_answer).strip().replace(",", ""))
                guess_num = float(guess.strip().replace(",", ""))
                scores.append(3.5 if guess_num == true_num else -1.5)
            except Exception:
                scores.append(0)
                
        return scores
    
    # ==================== GRPO Configuration ====================
    vllm_sampling_params = SamplingParams(
        min_p=0.1,
        top_p=1.0,
        top_k=-1,
        seed=3407,
        stop=[tokenizer.eos_token],
        include_stop_str_in_output=True,
    )
    
    training_args = GRPOConfig(
        vllm_sampling_params=vllm_sampling_params,
    disable_tqdm=True,
    torch_compile=False,
        temperature=1.0,
        learning_rate=5e-6,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=4,
        max_prompt_length=4096,
        max_completion_length=4096,
        max_steps=10000,
        save_steps=100,
        report_to="none",
        output_dir="outputs",
    )
    
    # ==================== Trainer Creation ====================
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            match_format_exactly,
            match_format_approximately,
            check_answer,
            check_numbers,
        ],
        args=training_args,
        train_dataset=dataset,
    )
    
    trainer.model.training = True
    return trainer