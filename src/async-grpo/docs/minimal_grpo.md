```python
# minimal_grpo.py — one reward, no SFT

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams
import re, torch

# ----- Model (LoRA-ready, but no SFT) -----
model, tok = FastLanguageModel.from_pretrained(
    "unsloth/Qwen3-4B-Base",
    max_seq_length=2048,
    load_in_4bit=False,
    fast_inference=True,
    gpu_memory_utilization=0.7,
)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    use_gradient_checkpointing="unsloth",
)

# ----- Data: GSM8K; answers live after '####' -----
ds = load_dataset("openai/gsm8k", "main", split="train")
def gold(x):
    s = x["answer"]
    return s.split("####")[-1].strip() if "####" in s else s.strip()

ds = ds.map(lambda x: {
    "prompt": [{"role":"user","content": x["question"]}],
    "answer": gold(x),
})

# ----- Single reward: numeric/string exact match -----
def reward_correct(prompts, completions, answer, **_):
    scores = []
    for comp, gold_ans in zip(completions, answer):
        text = comp[0]["content"]
        # prefer last number if present; else compare stripped strings
        nums_pred = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
        nums_gold = re.findall(r"-?\d+(?:\.\d+)?", str(gold_ans).replace(",", ""))
        if nums_gold:
            try:
                pred = float(nums_pred[-1]) if nums_pred else None
                gold_v = float(nums_gold[-1])
                scores.append(1.0 if pred == gold_v else -1.0)
                continue
            except: pass
        scores.append(1.0 if str(gold_ans).strip() in text.strip() else -1.0)
    return scores

# ----- GRPO config -----
sp = SamplingParams(
    min_p=0.1, top_p=1.0, top_k=-1, seed=3407,
    stop=[tok.eos_token], include_stop_str_in_output=True,
)
args = GRPOConfig(
    vllm_sampling_params=sp,
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
    max_prompt_length=1024,
    max_completion_length=512,
    max_steps=100, save_steps=100,
    report_to="none", output_dir="outputs",
)

# ----- Train (GRPO only) -----
trainer = GRPOTrainer(
    model=model,
    processing_class=tok,
    reward_funcs=[reward_correct],  # <- only one reward
    args=args,
    train_dataset=ds,
)
trainer.train()

# (optional) save LoRA
# model.save_lora("grpo_lora")
```