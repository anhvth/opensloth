import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TORCHDYNAMO_DISABLE'] = '1'
#====
import unsloth # unsloth must be import before trl
import pandas as pd
from datasets import load_dataset, Dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastModel
print("Unsloth version:", unsloth.__version__)
model, tokenizer = FastModel.from_pretrained(
    model_name = "hf-models/unsloth/Qwen3-0.6B-bnb-4bit/",
    max_seq_length = 2048,
    load_in_4bit = True,
    load_in_8bit = False,
    full_finetuning = False,
    device_map="auto"
)

model = FastModel.get_peft_model(
    model,
    r = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

reasoning_dataset = load_dataset("unsloth/OpenMathReasoning-mini", split = "cot")

def generate_conversation(examples):
    problems  = examples["problem"]
    solutions = examples["generated_solution"]
    conversations = []
    for problem, solution in zip(problems, solutions, strict=False):
        conversations.append([
            {"role" : "user",      "content" : problem},
            {"role" : "assistant", "content" : solution},
        ])
    return { "conversations": conversations, }

reasoning_dataset = reasoning_dataset.select(range(min(300, len(reasoning_dataset))))
reasoning_conversations = tokenizer.apply_chat_template(
    reasoning_dataset.map(generate_conversation, batched = True)["conversations"],
    tokenize = False,
)

data = pd.Series(reasoning_conversations)

data.name = "text"

combined_dataset = Dataset.from_pandas(pd.DataFrame(data))
combined_dataset = combined_dataset.shuffle(seed = 3407)
tokenizer.eos_token = "<|im_end|>"

trainer = SFTTrainer(
    model = model,
    processing_class = tokenizer,
    train_dataset = combined_dataset,
    eval_dataset = None,
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 30,
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none",
    ),
)

trainer.train()
